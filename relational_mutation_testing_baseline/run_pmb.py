#!/usr/bin/env python3
"""Entry point for the PyMigBench relational mutation-testing experiment.

    python run_pmb.py phase0                      # measure the executable panel
    python run_pmb.py select --n 10               # choose the pilot panel
    python run_pmb.py run --run pilot10 --mutants 10
    python run_pmb.py naive --run pilot10         # the no-mutant baseline arm
    python run_pmb.py report --run pilot10
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("phase0", help="probe every migration for executability")
    q.add_argument("--workers", type=int, default=10)
    q.add_argument("--limit", type=int)
    q.add_argument("--redo", action="store_true")

    q = sub.add_parser("select", help="choose the panel from the phase0 ladder")
    q.add_argument("--n", type=int, default=10)
    q.add_argument("--strata", action="store_true",
                   help="rotate over (LOC band, library pair) rather than pair alone")

    q = sub.add_parser("run", help="the four-stage funnel")
    q.add_argument("--run", required=True)
    q.add_argument("--mutants", type=int, default=10)
    q.add_argument("--retries", type=int, default=3)
    q.add_argument("--project-workers", type=int, default=3)
    q.add_argument("--mutant-workers", type=int, default=4)
    q.add_argument("--panel")
    q.add_argument("--redo", action="store_true")
    q.add_argument("--stop-after", type=int, default=4, choices=(3, 4),
                   help="3 = stop with T' on disk, for the migration-validation study")

    q = sub.add_parser("naive", help="the no-mutant baseline arm")
    q.add_argument("--run", required=True)
    q.add_argument("--retries", type=int, default=3)
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--redo", action="store_true")
    q.add_argument("--suite-only", action="store_true",
                   help="write the suite but do not score it on M' (stop-after-3 runs)")

    q = sub.add_parser("migbaseline", help="LLM does the migration (arXiv:2504.13272)")
    q.add_argument("--run", required=True)
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--panel")
    q.add_argument("--redo", action="store_true")
    q.add_argument("--allow-static", action="store_true",
                   help="fall back to a parse-only candidate where no venv will build")

    q = sub.add_parser("validate", help="judge the candidate migrations with T'")
    q.add_argument("--run", required=True)
    q.add_argument("--model", help="candidate arm to judge; default the current RMT_MODEL")
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--report-only", action="store_true")
    q.add_argument("--arm", default="rt", choices=("rt", "naive", "target"))
    q.add_argument("--redo", action="store_true")

    q = sub.add_parser("targetarm", help="mutate M at the migration sites, not P")
    q.add_argument("--run", required=True)
    q.add_argument("--mutants", type=int, default=8)
    q.add_argument("--retries", type=int, default=3)
    q.add_argument("--project-workers", type=int, default=2)
    q.add_argument("--mutant-workers", type=int, default=4)
    q.add_argument("--id", action="append", help="repeatable; default the 5-migration panel")
    q.add_argument("--redo", action="store_true")

    q = sub.add_parser("changedev", help="log the developer's edits vs the model's")
    q.add_argument("--run", required=True)
    q.add_argument("--model", action="append",
                   help="repeatable; default every model with candidates on disk")
    q.add_argument("--report-only", action="store_true")

    q = sub.add_parser("ceiling", help="tier 0 (compiles) and tier 2 (post-side imports)")
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--id", action="append", help="repeatable; default every migration")
    q.add_argument("--report-only", action="store_true")
    q.add_argument("--redo", action="store_true")

    q = sub.add_parser("guided", help="repair rejected candidates, guided vs blind")
    q.add_argument("--run", required=True)
    q.add_argument("--model", help="candidate arm to repair; default the current RMT_MODEL")
    q.add_argument("--arm", action="append", choices=("guided", "blind"),
                   help="repeatable; default both")
    q.add_argument("--id", action="append", help="repeatable; default every rejected one")
    q.add_argument("--workers", type=int, default=2)
    q.add_argument("--report-only", action="store_true")
    q.add_argument("--redo", action="store_true")

    q = sub.add_parser("report", help="aggregate a run")
    q.add_argument("--run", required=True)

    a = p.parse_args()

    if a.cmd == "phase0":
        from pmb import phase0
        phase0.run(limit=a.limit, workers=a.workers, redo=a.redo)
    elif a.cmd == "select":
        from pmb import select
        rows = select.save(select.stratified(a.n) if a.strata else select.select(a.n))
        for r in rows:
            print(f"{r['id']}  {r['source']}->{r['target']}  {r['loc']} LOC  "
                  f"band={select.band(r)}  gt={r.get('gt')}")
        print(json.dumps({"n": len(rows),
                          "pairs": len({(r['source'], r['target']) for r in rows}),
                          "gt_green": len(select.gt_ids(rows))}, indent=2))
    elif a.cmd == "run":
        from pmb import driver, select
        rows = select.load(a.panel) if a.panel else select.load()
        driver.run(a.run, select.migrations(rows), n_mutants=a.mutants,
                   retries=a.retries, project_workers=a.project_workers,
                   mutant_workers=a.mutant_workers, redo=a.redo,
                   gt_ids=select.gt_ids(rows), stop_after=a.stop_after)
    elif a.cmd == "naive":
        from pmb import baseline_naive, select
        baseline_naive.run(a.run, select.load(), retries=a.retries,
                           workers=a.workers, redo=a.redo,
                           suite_only=a.suite_only)
    elif a.cmd == "migbaseline":
        from pmb import migbaseline, select
        rows = select.load(a.panel) if a.panel else select.load()
        migbaseline.run(a.run, select.migrations(rows), workers=a.workers, redo=a.redo,
                        allow_static=a.allow_static)
    elif a.cmd == "validate":
        from pmb import validate
        from pmb.llm import MODEL
        model = a.model or MODEL
        if not a.report_only:
            validate.run(a.run, model, workers=a.workers, redo=a.redo,
                         arm=a.arm)
        validate.report(a.run, model, arm=a.arm)
    elif a.cmd == "targetarm":
        from pmb import target_arm
        target_arm.run(a.run, ids=a.id, n_mutants=a.mutants, retries=a.retries,
                       project_workers=a.project_workers,
                       mutant_workers=a.mutant_workers, redo=a.redo)
    elif a.cmd == "changedev":
        from pmb import changed_dev
        if a.report_only:
            changed_dev.report(a.run)
        else:
            changed_dev.run(a.run, models=a.model)
    elif a.cmd == "ceiling":
        from pmb import ceiling
        if a.report_only:
            ceiling.report()
        else:
            ceiling.run(workers=a.workers, ids=a.id, redo=a.redo)
    elif a.cmd == "guided":
        from pmb import guided
        from pmb.llm import MODEL
        model = a.model or MODEL
        if not a.report_only:
            guided.run(a.run, model, arms=tuple(a.arm or ("guided", "blind")),
                       ids=a.id, workers=a.workers, redo=a.redo)
        guided.report(a.run, model)
    elif a.cmd == "report":
        from pmb import report
        report.run(a.run)


if __name__ == "__main__":
    main()
