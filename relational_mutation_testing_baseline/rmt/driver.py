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

from . import cbench, rbench
from . import stage1_mutate as s1
from . import stage2_testgen as s2
from . import stage3_transpile as s3
from . import stage4_relate as s4

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _one_mutant(project, mutant, ctx, outdir, retries, log, redo=False):
    """Stages 2-4 for a single mutant. Returns its result record.

    The record is written to `verdicts.json` on *every* terminal path, so a resumed run
    can skip the mutant without redoing its gcc and cargo work.
    """
    name = ctx["name"]
    mid = mutant["verdict"]["id"]
    d = outdir / "mutants" / str(mid)
    d.mkdir(parents=True, exist_ok=True)
    done = d / "verdicts.json"

    if done.exists() and not redo:
        rec = json.loads(done.read_text())
        if rec.get("outcome"):
            log(f"{name}/{mid}: {rec['outcome']} (cached)")
            return rec

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
        if len(kept) == len(blob["mutants"]):
            return kept, blob["verdicts"], True
    kept, verdicts = s1.generate(project, ctx["iface_names"], n_mutants,
                                 tag=ctx["name"])
    saved.write_text(json.dumps({"verdicts": verdicts,
                                 "mutants": s1.dehydrate(kept)}, indent=2))
    return kept, verdicts, False


def run_project(prof, n_mutants, outdir, retries, workers, log=print, redo=False):
    name = prof["project"]
    t0 = time.time()
    with cbench.Project(prof["cbench"]) as project:
        ctx = {"name": name,
               "iface_names": "\n".join(f"- `{f}`" for f in rbench.iface_fns(name)),
               "sources": s1.source_bundle(project)}

        kept, verdicts, reused = _stage1(project, ctx, n_mutants, outdir, redo)
        log(f"{name}: stage1 {len(kept)}/{len(verdicts)} mutants valid"
            + (" (reloaded)" if reused else ""))

        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one_mutant, project, m, ctx, outdir, retries, log,
                              redo): m for m in kept}
            for fut in as_completed(futs):
                mid = futs[fut]["verdict"]["id"]
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001 - one mutant must not sink a project
                    log(f"{name}/{mid}: ERROR {type(e).__name__}: {e}")
                    results.append({"id": mid, "outcome": "F5_exception",
                                    "error": traceback.format_exc()[-2000:]})

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


def run(run_name, projects, n_mutants=10, retries=3, project_workers=4,
        mutant_workers=4, redo=False):
    root = RESULTS / run_name
    root.mkdir(parents=True, exist_ok=True)
    todo = projects if redo else [
        p for p in projects if not (root / p["project"] / "mutants.json").exists()]
    skipped = len(projects) - len(todo)
    print(f"run={run_name} projects={len(todo)} (+{skipped} done) "
          f"mutants/project={n_mutants} "
          f"workers={project_workers}x{mutant_workers}", flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(prof):
        outdir = root / prof["project"]
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            return run_project(prof, n_mutants, outdir, retries, mutant_workers, log,
                               redo=redo)
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
    print(f"done in {round(time.time() - t0)}s", flush=True)
