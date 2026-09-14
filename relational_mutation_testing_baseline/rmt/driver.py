"""The funnel, run over projects and mutants concurrently.

Stage 1 is one call per project; everything after it is per mutant and fully
independent, so mutants fan out inside a project and projects fan out across the run.
The work is I/O bound (API latency, gcc, cargo), so threads are the right tool -- and
they let all workers share one `Project` and one LLM cache.

Resumable at three granularities, because the expensive parts are not only the LLM
calls:
  - a finished project is skipped entirely (`mutants.json`),
  - a finished mutant is skipped and its verdict reloaded (`mutants/<id>/verdicts.json`),
  - Stage 1 is reloaded from `stage1.json`, so a resumed project neither re-proposes
    mutants nor re-runs the compile gate on them,
and under all of that, every LLM call that already succeeded is served from the disk
cache. `--redo` forces a clean run.
"""
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import cbench, llm, rbench
from . import stage1_mutate as _s1_names  # noqa: F401  (kept for readers grepping stages)

#: Outcomes worth re-running when `--topup` is on: they failed on something this code
#: has since changed (interface-file rule, soundness repair, alternate references) or on
#: infrastructure, not on the mutant itself. F1/F4 are the mutant's own failure and are
#: replaced by top-up rather than retried.
RETRY_OUTCOMES = frozenset({"F5_exception", "F5_error", "F3_unsound_on_correct_rust",
                            "F5_no_rust_counterpart"})
#: Exported ground-truth-passing implementations, used as alternate references.
EXPORTS = Path(__file__).resolve().parent.parent.parent / "exports"
ALT_REFERENCES = 4
REFERENCE_EXPORT = "codex_expts__all__initial_codex_output"
MAX_TOPUP_ROUNDS = 4
from . import stage1_mutate as s1
from . import stage2_testgen as s2
from . import stage3_transpile as s3
from . import stage4_relate as s4

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _alt_references(name):
    """Ground-truth-passing implementations of `name` other than the reference.

    Taken from `export_implementations.py`'s `good/` set, whose label is a fresh run of
    the benchmark's own suite. The export of the codex reference itself is skipped (it
    is the M that just failed). Sorted for determinism, capped so an unsound test costs
    a bounded number of cargo runs.
    """
    d = EXPORTS / name / "good"
    if not d.is_dir():
        return []
    return sorted(x for x in d.iterdir()
                  if (x / "Cargo.toml").exists()
                  and x.name != REFERENCE_EXPORT)[:ALT_REFERENCES]


def _rescue_unsound(project, name, mid, tg, tr, rec, retries, log):
    """RT failed the reference M. Decide whether the test or M is wrong, and recover.

    First the cheap, LLM-free triage: run RT on other implementations that pass the
    ground-truth suite. If it passes a majority of them the reference is the outlier,
    and the trial is re-run with one of those as M (recorded as `reference`). Otherwise
    the translation is the likelier culprit and Stage 3.5 re-derives it from the C test
    and the interfaces, after which the ordinary trial on M decides.

    Returns the stage-4 record, or None if nothing rescued the mutant.
    """
    alts = _alt_references(name)
    votes = []
    for alt in alts:
        r = rbench.run_test(alt, tr["test_rs"], tag=f"{name}_{mid}_alt")
        votes.append({"impl": alt.name, "status": r["status"]})
    passing = [v for v in votes if v["status"] == "green"]
    rec["unsound_triage"] = {"alternates": votes,
                             "passes": len(passing), "of": len(votes)}
    if votes and len(passing) * 2 > len(votes):
        alt = alts[[v["impl"] for v in votes].index(passing[0]["impl"])]
        log(f"{name}/{mid}: RT passes {len(passing)}/{len(votes)} other correct impls; "
            f"reference swapped to {alt.name}")
        t = s4.trial(name, tr["test_rs"], mutant_verdict(rec), tag=f"{name}_{mid}_swap",
                     impl_dir=alt)
        if t["verdict"] == "killed":
            return t
        rec["swap_trial"] = t

    fix = s3.repair_unsound(name, tg["test_c"], tr["test_rs"],
                            rec["stage4_first"].get("out", ""), retries=retries,
                            tag=f"{name}_{mid}")
    rec["stage3_unsound_repair"] = {k: v for k, v in fix.items() if k != "test_rs"}
    if fix.get("stands_by_c") or not fix.get("compiles"):
        return None
    t = s4.trial(name, fix["test_rs"], mutant_verdict(rec), tag=f"{name}_{mid}_fix")
    if t["verdict"] in ("killed", "not_killed", "no_counterpart"):
        tr["test_rs"] = fix["test_rs"]
        return t
    rec["fix_trial"] = t
    return None


def mutant_verdict(rec):
    """The stage-1 verdict fields Stage 4b ports from, as stored on the record."""
    keys = ("id", "function", "file", "n_sites", "fault_class", "mutant_kind",
            "reachable_via", "semantic_change", "observable_effect", "rationale")
    return {k: rec.get(k) for k in keys}


def _one_mutant(project, mutant, ctx, outdir, retries, log, redo=False,
                retry_outcomes=frozenset({"F5_exception"})):
    """Stages 2-4 for a single mutant. Returns its result record.

    The record is written to `verdicts.json` on *every* terminal path, so a resumed run
    can skip the mutant without redoing its gcc and cargo work. Outcomes in
    `retry_outcomes` are re-run instead of reloaded; their Stage 2/3 LLM calls are cache
    hits, so a retry re-pays only for what is genuinely new.
    """
    name = ctx["name"]
    mid = mutant["verdict"]["id"]
    d = outdir / "mutants" / str(mid)
    d.mkdir(parents=True, exist_ok=True)
    done = d / "verdicts.json"

    if done.exists() and not redo:
        rec = json.loads(done.read_text())
        # F5_exception is a crash, not a verdict -- always retry it. (The collateral
        # pass persists whatever record it is handed, so one can end up on disk.)
        if rec.get("outcome") and rec["outcome"] not in retry_outcomes | {"F5_exception"}:
            log(f"{name}/{mid}: {rec['outcome']} (cached)")
            return rec
        if rec.get("outcome"):
            log(f"{name}/{mid}: {rec['outcome']} -> retrying")

    rec = {"id": mid, **mutant["verdict"]}

    def finish(outcome, note=""):
        rec["outcome"] = outcome
        done.write_text(json.dumps(rec, indent=2))
        log(f"{name}/{mid}: {outcome}{note}")
        return rec

    diff = s1.unified_diff(project, mutant["patch"])
    (d / "patch.diff").write_text(diff)

    tg = s2.generate(project, mutant, ctx["iface_names"], ctx["sources"], diff,
                     retries=retries, tag=f"{name}_{mid}")
    rec["stage2"] = {k: v for k, v in tg.items() if k != "test_c"}
    if not tg["killed"]:
        return finish(f"F1_{tg['reason']}")
    (d / "test.c").write_text(tg["test_c"])

    tr = s3.transpile(name, tg["test_c"], retries=retries, tag=f"{name}_{mid}")
    rec["stage3"] = {k: v for k, v in tr.items() if k != "test_rs"}
    if not tr["compiles"]:
        (d / "test_failed.rs").write_text(tr["test_rs"] or "")
        return finish("F2_transpile_no_compile")
    (d / "test.rs").write_text(tr["test_rs"])

    t = s4.trial(name, tr["test_rs"], mutant["verdict"], tag=f"{name}_{mid}")
    if t["verdict"] == "unsound":
        rec["stage4_first"] = t
        rescued = _rescue_unsound(project, name, mid, tg, tr, rec, retries, log)
        if rescued is not None:
            t = rescued
            (d / "test.rs").write_text(tr["test_rs"])
    rec["stage4"] = t
    return finish(OUTCOME.get(t["verdict"], "F5_" + t["verdict"]))


OUTCOME = {
    "killed": "KILLED",
    "not_killed": "F4_not_discriminating",
    "unsound": "F3_unsound_on_correct_rust",
    "no_counterpart": "F5_no_rust_counterpart",
    "error": "F5_error",
}


def _stage1(project, ctx, n_mutants, outdir, redo):
    """Propose and gate mutants, or reload a previous run's gated set."""
    saved = outdir / "stage1.json"
    if saved.exists() and not redo:
        blob = json.loads(saved.read_text())
        kept = s1.rehydrate(project, blob["mutants"])
        # An empty saved set is a failed proposal (XOpt, rhbloom), not a result.
        if kept and len(kept) == len(blob["mutants"]):
            return kept, blob["verdicts"], True
    kept, verdicts = s1.generate(project, ctx["iface_names"], n_mutants,
                                 tag=ctx["name"])
    saved.write_text(json.dumps({"verdicts": verdicts,
                                 "mutants": s1.dehydrate(kept)}, indent=2))
    return kept, verdicts, False


def _next_id(verdicts):
    """One past the highest `mNN` already proposed for this project."""
    top = 0
    for v in verdicts:
        try:
            top = max(top, int(str(v.get("id", "m0"))[1:]))
        except ValueError:
            continue
    return top + 1


def _topup(project, ctx, outdir, kept, verdicts, results, need, round_, log):
    """One top-up round: propose `need` new mutants that avoid everything tried so far.

    Appends to `stage1.json` so a resumed run rehydrates the union, and returns the newly
    gated mutants (the caller runs them).
    """
    by_id = {str(r["id"]): r.get("outcome") for r in results}
    tried = [{**v, "outcome": by_id.get(str(v.get("id")))} for v in verdicts]
    new_kept, new_verdicts = s1.generate(
        project, ctx["iface_names"], need, tag=ctx["name"],
        exclude=s1.describe_tried(tried), round_=round_, start_id=_next_id(verdicts))
    kept += new_kept
    verdicts += new_verdicts
    (outdir / "stage1.json").write_text(json.dumps(
        {"verdicts": verdicts, "mutants": s1.dehydrate(kept),
         "topup_rounds": round_}, indent=2))
    log(f"{ctx['name']}: top-up round {round_}: {len(new_kept)}/{len(new_verdicts)} "
        f"new mutants valid")
    return new_kept


def _run_mutants(project, mutants, ctx, outdir, retries, workers, log, redo,
                 retry_outcomes):
    name = ctx["name"]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_mutant, project, m, ctx, outdir, retries, log, redo,
                          retry_outcomes): m for m in mutants}
        for fut in as_completed(futs):
            mid = futs[fut]["verdict"]["id"]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001 - one mutant must not sink a project
                log(f"{name}/{mid}: ERROR {type(e).__name__}: {e}")
                results.append({"id": mid, "outcome": "F5_exception",
                                "error": traceback.format_exc()[-2000:]})
    return results


def run_project(prof, n_mutants, outdir, retries, workers, log=print, redo=False,
                topup_budget=0, retry_outcomes=frozenset({"F5_exception"})):
    """The funnel for one project.

    `topup_budget` > 0 turns on top-up: while fewer than `n_mutants` are KILLED and fewer
    than `topup_budget` mutants have been proposed in total, another round proposes
    replacements for the ones that fell out. Each round asks for the shortfall plus half
    again, since some of any batch will fall out too, capped by the remaining budget.
    """
    name = prof["project"]
    t0 = time.time()
    with cbench.Project(prof["cbench"]) as project:
        ctx = {"name": name,
               "iface_names": "\n".join(f"- `{f}`" for f in rbench.iface_fns(name)),
               "sources": s1.source_bundle(project)}

        kept, verdicts, reused = _stage1(project, ctx, n_mutants, outdir, redo)
        log(f"{name}: stage1 {len(kept)}/{len(verdicts)} mutants valid"
            + (" (reloaded)" if reused else ""))

        results = _run_mutants(project, kept, ctx, outdir, retries, workers, log, redo,
                               retry_outcomes)

        saved = outdir / "stage1.json"
        round_ = (json.loads(saved.read_text()).get("topup_rounds", 0)
                  if saved.exists() else 0)
        while topup_budget and round_ < MAX_TOPUP_ROUNDS:
            killed = sum(1 for r in results if r.get("outcome") == "KILLED")
            room = topup_budget - len(verdicts)
            if killed >= n_mutants or room <= 0:
                break
            need = min(room, (n_mutants - killed) + (n_mutants - killed + 1) // 2)
            round_ += 1
            before = len(verdicts)
            new = _topup(project, ctx, outdir, kept, verdicts, results, need, round_, log)
            if not new:
                if len(verdicts) == before:
                    log(f"{name}: top-up round {round_} proposed nothing; stopping")
                    break
                continue  # a round that gated nothing still spent budget; try once more
            results += _run_mutants(project, new, ctx, outdir, retries, workers, log,
                                    redo, retry_outcomes)

        # T may kill mutants other than its own; that is a diagnostic about mutant
        # redundancy, and it needs every accepted test, so it runs after the fan-out.
        for r in results:
            d = outdir / "mutants" / str(r["id"])
            test = d / "test.c"
            if not test.exists() or r.get("collateral") is not None:
                continue
            r["collateral"] = s2.collateral(project, test.read_text(), kept, r["id"])
            (d / "verdicts.json").write_text(json.dumps(r, indent=2))

    results.sort(key=lambda r: str(r["id"]))
    payload = {"project": name, "profile": prof,
               "reference": str(rbench.reference(name)),
               "stage1_verdicts": verdicts, "mutants": results,
               "seconds": round(time.time() - t0, 1)}
    (outdir / "mutants.json").write_text(json.dumps(payload, indent=2))
    return payload


def _finished(root, prof, n_mutants=10, topup_budget=0, retry_outcomes=frozenset()):
    """A project is done only if no mutant fell over on a transient error.

    `F5_exception` is not a result -- it is an API overload or a crash -- and its mutant
    never wrote a `verdicts.json`, so re-running the project retries exactly those and
    reloads the rest from cache.
    """
    f = root / prof["project"] / "mutants.json"
    if not f.exists():
        return False
    ms = json.loads(f.read_text())["mutants"]
    if any(m["outcome"] == "F5_exception" for m in ms):
        return False
    if topup_budget:
        killed = sum(1 for m in ms if m["outcome"] == "KILLED")
        if killed < n_mutants and (len(ms) < topup_budget or
                                   any(m["outcome"] in retry_outcomes for m in ms)):
            return False
    return True


def run(run_name, projects, n_mutants=10, retries=3, project_workers=4,
        mutant_workers=4, redo=False, topup_budget=0, retry_outcomes=None):
    retry_outcomes = frozenset(retry_outcomes or ()) | {"F5_exception"}
    root = RESULTS / run_name
    root.mkdir(parents=True, exist_ok=True)
    todo = projects if redo else [
        p for p in projects
        if not _finished(root, p, n_mutants, topup_budget, retry_outcomes)]
    skipped = len(projects) - len(todo)
    print(f"run={run_name} projects={len(todo)} (+{skipped} done) "
          f"mutants/project={n_mutants} "
          + (f"topup-budget={topup_budget} retry={sorted(retry_outcomes)} "
             if topup_budget else "")
          + f"workers={project_workers}x{mutant_workers}", flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(prof):
        outdir = root / prof["project"]
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            return run_project(prof, n_mutants, outdir, retries, mutant_workers, log,
                               redo=redo, topup_budget=topup_budget,
                               retry_outcomes=retry_outcomes)
        except Exception:  # noqa: BLE001 - record and keep going
            (outdir / "ERROR.txt").write_text(traceback.format_exc())
            log(f"{prof['project']}: PROJECT ERROR (see ERROR.txt)")
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=project_workers) as ex:
        for n, r in enumerate(ex.map(one, todo), 1):
            if r:
                counts = {}
                for m in r["mutants"]:
                    counts[m["outcome"]] = counts.get(m["outcome"], 0) + 1
                print(f"[{n}/{len(todo)}] {r['project']}: {counts} ({r['seconds']}s)",
                      flush=True)

    # Where the tokens went, per stage. Written even on a partial run: it is the only
    # record of usage, and the disk cache means a rerun would not re-measure it.
    cost = llm.cost_ledger()
    cost["seconds"] = round(time.time() - t0)
    (root / "cost.json").write_text(json.dumps(cost, indent=2))
    print(f"done in {cost['seconds']}s; "
          f"{cost['total']['calls']} LLM calls, ${cost['total']['paid_usd']:.2f} paid "
          f"(${cost['total']['usd']:.2f} incl. cache hits)", flush=True)
