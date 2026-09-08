"""Mutate the migrated code, not the code it was migrated from.

The funnel this package was built around mutates P, writes a test that kills the mutant,
and carries that test across the migration. Measuring it turned up a specific failure:
of 25 rejecting test runs on the pmb40 panel, 12 died on `AttributeError` or `ImportError`
reaching for a name -- the test was probing M's internal *structure*, not the behaviour
of the migrated code. That is not an accident of Stage 3. It is what Stage 1 asks for.
`prompts/mutate.md` rule 7 tells the model to describe its fault library-independently,
*because* Stage 4 has to replay the same fault on the other side, and a fault that ports
across a library change is by construction a fault that has nothing to do with the
library change.

So this arm inverts the pipeline:

    stage 1'   mutate M -- the post-migration module -- at the sites the migration
               actually landed on, using PyMigBench's own record of which lines moved
               and which target APIs arrived. The taxonomy is migration defects: the old
               default assumed, the result type read the old way, the failure mode moved,
               an argument whose unit changed, a near-synonym chosen.
    stage 2'   write a test that passes M and fails M'. Unchanged machinery, pointed at
               the post-migration environment.
    stage 3'   there isn't one. The test is already written against the target library,
               so nothing has to be carried anywhere -- and Stage 3 was the other place
               M's declared shape leaked into the test.

What comes out is a test whose assertions are about what the new API *does*. `validate
--arm target` then runs it against the candidate migrations exactly as it runs T'.

This is a diagnostic on a small panel, not a replacement funnel. It is pointed at the
five migrations where the pmb40 result is most in question: the three whose T' falsely
rejected a candidate the repository's own suite accepts, and the two that carry a
genuinely incorrect candidate and must still be caught.
"""
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import changes as CH
from . import llm
from . import pybench as PB
from . import stage1_mutate as s1
from . import stage2_testgen as s2
from .dataset import by_id

RESULTS = Path(__file__).resolve().parent.parent / "results"

# The three false rejections and the two true positives from the pmb40 panel. A panel
# chosen to be answerable rather than representative: if the migration-defect mutants do
# not clear the false rejections here, they do not clear them anywhere, and if they lose
# the true positives here the trade is not worth making.
PANEL = [
    "lockfile__fasteners__samschott@maestral__e4388ee1",
    "bunch__munch__1and1@confluencer__df895ac8",
    "dill__cloudpickle__rapid-design-of-systems-laboratory@beluga__078e3e56",
    "ipaddr__ipaddress__google@capirca__eb768ea7",
    "retrying__tenacity__openstack@ironic-inspector__f4648fac",
]


def render_changes(mig, rel):
    """The post-migration half of the recorded diff. See `changes.render`."""
    return CH.render(mig, rel, side="target")


def one_mutant(mig, post_env, mutant, outdir, retries=3, log=print, redo=False):
    """Stage 2' for a single migration-defect mutant."""
    mid = mutant["verdict"]["id"]
    d = outdir / mid
    d.mkdir(parents=True, exist_ok=True)
    out = d / "result.json"
    if out.exists() and not redo:
        return json.loads(out.read_text())

    t0 = time.time()
    rec = {"id": mid, "migration": mig.id, "verdict": mutant["verdict"],
           "record": mutant["record"]}
    diff = s1.unified_diff(post_env, mutant["patch"])
    (d / "mutant.diff").write_text(diff)

    # `source=mig.target`: the module this test is written against uses the *target*
    # library, and telling the prompt otherwise would invite it to mock the library that
    # is no longer there.
    tg = s2.generate(mig, post_env, mutant, diff, retries=retries,
                     tag=f"tgt_{mig.id}_{mid}", lib=mig.target)
    rec["stage2"] = {k: v for k, v in tg.items() if k != "test_py"}
    if tg["killed"]:
        (d / "test.py").write_text(tg["test_py"])
        rec["outcome"] = "T_ready"
    else:
        rec["outcome"] = f"F2_{tg['reason']}"
    rec["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=2))
    log(f"{mig.id}/{mid}: {rec['outcome']}")
    return rec


def run_migration(mid, run_name, n_mutants=8, retries=3, workers=4, log=print,
                  redo=False):
    root = RESULTS / run_name / mid
    outdir = root / "target_mutants"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "mutants.json"
    if out.exists() and not redo:
        log(f"{mid}: cached")
        return json.loads(out.read_text())

    t0 = time.time()
    mig = by_id(mid)
    _, post = PB.open_migration(mig)
    kept, verdicts = s1.generate(mig, post, n_mutants, f"tgt_{mid}",
                                 prompt="target_mutate",
                                 changes=render_changes(mig, post.rel))
    log(f"{mid}: stage1' {len(kept)}/{len(verdicts)} mutants valid")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one_mutant, mig, post, m, outdir, retries, log, redo): m
                for m in kept}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001 - one mutant must not sink a migration
                log(f"{mid}: mutant ERROR {type(e).__name__}: {e}")
                results.append({"id": futs[f]["verdict"]["id"], "outcome": "ERROR",
                                "error": traceback.format_exc()[-1500:]})

    results.sort(key=lambda r: str(r["id"]))
    payload = {"migration": mid, "source": mig.source, "target": mig.target,
               "file": post.rel, "n_proposed": len(verdicts), "n_valid": len(kept),
               "verdicts": verdicts, "mutants": results,
               "n_tests": sum(1 for r in results if r["outcome"] == "T_ready"),
               "seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(payload, indent=2))
    log(f"{mid}: {payload['n_tests']}/{len(kept)} tests in {payload['seconds']}s")
    return payload


def run(run_name, ids=None, n_mutants=8, retries=3, project_workers=2,
        mutant_workers=4, redo=False):
    root = RESULTS / run_name
    if not root.is_dir():
        raise SystemExit(f"no run at {root} -- this arm reuses its candidates")
    ids = ids or PANEL
    print(f"target-side mutation: run={run_name} migrations={len(ids)} "
          f"mutants={n_mutants} workers={project_workers}x{mutant_workers}", flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(mid):
        try:
            return run_migration(mid, run_name, n_mutants, retries, mutant_workers,
                                 log, redo)
        except Exception:  # noqa: BLE001
            (root / mid).mkdir(parents=True, exist_ok=True)
            (root / mid / "TARGET_ARM_ERROR.txt").write_text(traceback.format_exc())
            log(f"{mid}: MIGRATION ERROR (see TARGET_ARM_ERROR.txt)")
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=project_workers) as ex:
        rows = [r for r in ex.map(one, ids) if r]

    cost = llm.cost_ledger()
    cost["seconds"] = round(time.time() - t0)
    (root / "cost_target_arm.json").write_text(json.dumps(cost, indent=2))
    print(f"\ndone in {cost['seconds']}s; {sum(r['n_tests'] for r in rows)} tests from "
          f"{sum(r['n_valid'] for r in rows)} valid mutants "
          f"({sum(r['n_proposed'] for r in rows)} proposed); "
          f"${cost['total']['paid_usd']:.2f} paid", flush=True)
    return rows
