"""The funnel, run over migrations and mutants concurrently.

Structurally identical to the C->Rust driver, and resumable at the same three
granularities, because the expensive parts are again not only the LLM calls:
  - a finished migration is skipped entirely (`mutants.json`),
  - a finished mutant is skipped and its verdict reloaded (`mutants/<id>/verdicts.json`),
  - Stage 1 is reloaded from `stage1.json`, so a resumed migration neither re-proposes
    mutants nor re-runs the import gate on them,
and under all of that, every LLM call that already succeeded is served from the disk
cache. `--redo` forces a clean run.

One thing is genuinely different: setting a migration up -- fetching the repo, building
two venvs, chasing a decade of missing dependencies -- can cost more wall-clock than the
whole funnel that follows, so it is done once per migration and shared by every mutant.
"""
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import llm
from . import pybench as PB
from . import changes as CH
from . import stage1_mutate as s1
from . import stage2_testgen as s2
from . import stage3_migrate as s3
from . import stage4_relate as s4

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Which Stage 1 prompt this run uses. `mutate` is the original, library-independent one;
# `mutate_mig` anchors every mutant to the recorded migration sites and refuses any edit
# that changes the module's declared surface. Set with RMT_MUTATE_PROMPT, and recorded in
# each `stage1.json` so a cached answer to one question is never reused for the other.
MUTATE_PROMPT = os.environ.get("RMT_MUTATE_PROMPT", "mutate")

OUTCOME = {
    "killed": "KILLED",
    "not_killed": "F4_not_discriminating",
    "unsound": "F3_unsound_on_migrated",
    "no_counterpart": "F5_no_counterpart",
    "error": "F5_error",
}


def _one_mutant(mig, envs, mutant, outdir, retries, log, redo=False, stop_after=4):
    """Stages 2-4 for a single mutant. Returns its result record.

    `stop_after=3` halts with T' written to disk and no counterpart built. That is not a
    truncated run of this experiment -- it is the whole input to a different one, where
    the code M' is replaced by a real migration attempt and T' is the instrument that
    judges it (see `migbaseline`). Stage 4's own question needs M', so it is skipped.
    """
    pre, post = envs
    mid = mutant["verdict"]["id"]
    d = outdir / "mutants" / str(mid)
    d.mkdir(parents=True, exist_ok=True)
    done = d / "verdicts.json"

    if done.exists() and not redo:
        rec = json.loads(done.read_text())
        if rec.get("outcome") and rec["outcome"] != "F5_exception":
            log(f"{mig.id}/{mid}: {rec['outcome']} (cached)")
            return rec

    rec = {"id": mid, **mutant["verdict"]}

    def finish(outcome):
        rec["outcome"] = outcome
        done.write_text(json.dumps(rec, indent=2))
        log(f"{mig.id}/{mid}: {outcome}")
        return rec

    diff = s1.unified_diff(pre, mutant["patch"])
    (d / "patch.diff").write_text(diff)

    tg = s2.generate(mig, pre, mutant, diff, retries=retries, tag=f"{mig.id}_{mid}")
    rec["stage2"] = {k: v for k, v in tg.items() if k != "test_py"}
    if not tg["killed"]:
        return finish(f"F1_{tg['reason']}")
    (d / "test.py").write_text(tg["test_py"])

    mg = s3.migrate(mig, post, tg["test_py"], retries=retries, tag=f"{mig.id}_{mid}")
    rec["stage3"] = {k: v for k, v in mg.items() if k != "test_py"}
    if not mg["loads"]:
        (d / "test_migrated_failed.py").write_text(mg["test_py"] or "")
        return finish("F2_migrate_no_load")
    (d / "test_migrated.py").write_text(mg["test_py"])
    if stop_after <= 3:
        return finish("T3_ready")

    t = s4.trial(mig, post, mg["test_py"], mutant["verdict"], tag=f"{mig.id}_{mid}",
                 run_gt=bool(rec.get("_gt")))
    rec["stage4"] = t
    return finish(OUTCOME.get(t["verdict"], "F5_" + t["verdict"]))


def _stage1(mig, pre, n_mutants, outdir, redo):
    saved = outdir / "stage1.json"
    if saved.exists() and not redo:
        blob = json.loads(saved.read_text())
        # A cached stage 1 is only reusable if it was asked the same question. A run that
        # asked for 10 mutants must not silently inherit a 3-mutant answer from a smoke
        # test, and a run on a different mutate prompt must not inherit any of it.
        if blob.get("prompt", "mutate") == MUTATE_PROMPT \
                and len(blob["verdicts"]) >= n_mutants:
            kept = s1.rehydrate(pre, blob["mutants"])
            if len(kept) == len(blob["mutants"]):
                return kept, blob["verdicts"], True
    extra = ({"changes": CH.render(mig, pre.rel, side="source")}
             if MUTATE_PROMPT == "mutate_mig" else {})
    kept, verdicts = s1.generate(mig, pre, n_mutants, tag=mig.id,
                                 prompt=MUTATE_PROMPT, **extra)
    saved.write_text(json.dumps({"prompt": MUTATE_PROMPT, "n_requested": n_mutants,
                                 "verdicts": verdicts,
                                 "mutants": s1.dehydrate(kept)}, indent=2))
    return kept, verdicts, False


def run_migration(mig, n_mutants, outdir, retries, workers, log=print, redo=False,
                  gt=False, stop_after=4):
    t0 = time.time()
    pre, post = PB.open_migration(mig)
    if gt and stop_after > 3:
        PB.prepare_gt(post)

    kept, verdicts, reused = _stage1(mig, pre, n_mutants, outdir, redo)
    log(f"{mig.id}: stage1 {len(kept)}/{len(verdicts)} mutants valid"
        + (" (reloaded)" if reused else ""))

    for m in kept:
        m["verdict"]["_gt"] = gt
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_mutant, mig, (pre, post), m, outdir, retries, log,
                          redo, stop_after): m for m in kept}
        for fut in as_completed(futs):
            mid = futs[fut]["verdict"]["id"]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001 - one mutant must not sink a migration
                log(f"{mig.id}/{mid}: ERROR {type(e).__name__}: {e}")
                results.append({"id": mid, "outcome": "F5_exception",
                                "error": traceback.format_exc()[-2000:]})

    # T may kill mutants other than its own; that is a diagnostic about mutant
    # redundancy, and it needs every accepted test, so it runs after the fan-out.
    for r in results:
        test = outdir / "mutants" / str(r["id"]) / "test.py"
        if not test.exists() or r.get("collateral") is not None:
            continue
        r["collateral"] = s2.collateral(pre, test.read_text(), kept, r["id"])
        (outdir / "mutants" / str(r["id"]) / "verdicts.json").write_text(
            json.dumps(r, indent=2))

    results.sort(key=lambda r: str(r["id"]))
    payload = {"migration": mig.id, "profile": mig.as_dict(),
               "python": PB.python_for(mig), "gt_arm": gt, "stop_after": stop_after,
               "stage1_verdicts": verdicts, "mutants": results,
               "seconds": round(time.time() - t0, 1)}
    (outdir / "mutants.json").write_text(json.dumps(payload, indent=2))
    return payload


def _finished(root, mig):
    f = root / mig.id / "mutants.json"
    if not f.exists():
        return False
    return not any(m["outcome"] == "F5_exception"
                   for m in json.loads(f.read_text())["mutants"])


def run(run_name, migs, n_mutants=10, retries=3, project_workers=4, mutant_workers=4,
        redo=False, gt_ids=(), stop_after=4):
    root = RESULTS / run_name
    root.mkdir(parents=True, exist_ok=True)
    todo = migs if redo else [m for m in migs if not _finished(root, m)]
    print(f"run={run_name} migrations={len(todo)} (+{len(migs) - len(todo)} done) "
          f"mutants/migration={n_mutants} stages=1-{stop_after} "
          f"workers={project_workers}x{mutant_workers}", flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(mig):
        outdir = root / mig.id
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            return run_migration(mig, n_mutants, outdir, retries, mutant_workers, log,
                                 redo=redo, gt=mig.id in gt_ids, stop_after=stop_after)
        except Exception:  # noqa: BLE001 - record and keep going
            (outdir / "ERROR.txt").write_text(traceback.format_exc())
            log(f"{mig.id}: MIGRATION ERROR (see ERROR.txt)")
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=project_workers) as ex:
        for n, r in enumerate(ex.map(one, todo), 1):
            if r:
                counts = {}
                for m in r["mutants"]:
                    counts[m["outcome"]] = counts.get(m["outcome"], 0) + 1
                print(f"[{n}/{len(todo)}] {r['migration']}: {counts} ({r['seconds']}s)",
                      flush=True)

    cost = llm.cost_ledger()
    cost["seconds"] = round(time.time() - t0)
    (root / "cost.json").write_text(json.dumps(cost, indent=2))
    print(f"done in {cost['seconds']}s; "
          f"{cost['total']['calls']} LLM calls, ${cost['total']['paid_usd']:.2f} paid "
          f"(${cost['total']['usd']:.2f} incl. cache hits)", flush=True)
