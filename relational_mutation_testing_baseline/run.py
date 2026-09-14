#!/usr/bin/env python
"""Relational mutation testing, C <-> Rust. The only entrypoint.

    ./run.py select                     choose the project set
    ./run.py run --mutants 10           run the funnel
    ./run.py report                     summarise a finished run
    ./run.py naive --run <run>          the naive baseline on that run's mutants
    ./run.py rustnative --run <run>     the Rust-native oracle arm on the same mutants
    ./run.py naive-report --run <run>   pipeline vs naive vs GT vs rust-native

Run it with the project venv: `.venv/bin/python run.py ...`
"""
import argparse
import sys

from rmt import (baseline_naive, baseline_report, baseline_rustnative, driver, pilot,
                 report)


def cmd_select(a):
    profs, picked = pilot.select_all() if a.all else pilot.select(a.n)
    print(f"{len(profs)} eligible projects -> selected {len(picked)}\n")
    print(f"{'project':30} {'mode':12} {'loc':>6} {'files':>5} {'iface':>5} {'gt':>3}")
    for p in picked:
        print(f"{p['project']:30} {p['mode']:12} {p['c_loc']:6} "
              f"{p['c_files']:5} {p['n_iface_fns']:5} "
              f"{'yes' if p.get('gt_green') else 'NO':>3}")


def _projects(a):
    projects = pilot.load(pilot.PANEL if a.all else None)
    if a.projects:
        projects = [p for p in projects if p["project"] in a.projects]
        missing = set(a.projects) - {p["project"] for p in projects}
        if missing:
            sys.exit(f"not in the selected set: {', '.join(sorted(missing))} "
                     f"(run `select` first, or widen it with --n)")
    return projects


def cmd_run(a):
    projects = _projects(a)
    driver.run(a.run, projects, n_mutants=a.mutants, retries=a.retries,
               project_workers=a.project_workers, mutant_workers=a.mutant_workers,
               redo=a.redo, topup_budget=a.topup,
               retry_outcomes=driver.RETRY_OUTCOMES if a.topup else None)


def cmd_widen(a):
    from rmt import widen
    widen.run(a.run, _projects(a), retries=a.retries, project_workers=a.project_workers,
              mutant_workers=a.mutant_workers, redo=a.redo)


def cmd_report(a):
    text = report.summarise(a.run)
    out = driver.RESULTS / a.run / "REPORT.md"
    if out.parent.is_dir():
        out.write_text(text)
        print(f"wrote {out}\n")
    print(text)


def cmd_naive(a):
    baseline_naive.run(a.run, _projects(a), retries=a.retries,
                       project_workers=a.project_workers,
                       mutant_workers=a.mutant_workers, redo=a.redo)


def cmd_rustnative(a):
    baseline_rustnative.run(a.run, _projects(a), retries=a.retries,
                            project_workers=a.project_workers,
                            mutant_workers=a.mutant_workers, redo=a.redo,
                            max_mutants=a.max_mutants)


def cmd_naive_report(a):
    text = baseline_report.summarise(a.run)
    out = driver.RESULTS / a.run / "NAIVE_BASELINE.md"
    if out.parent.is_dir():
        out.write_text(text)
        print(f"wrote {out}\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="choose and freeze the project set")
    s.add_argument("-n", type=int, default=10, help="how many projects")
    s.add_argument("--all", action="store_true",
                   help="the whole panel (results/panel.json), soft filters off")
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("run", help="run the funnel")
    s.add_argument("--run", default="pilot1", help="run name (results/<run>/)")
    s.add_argument("--mutants", type=int, default=10, help="mutants per project")
    s.add_argument("--retries", type=int, default=3, help="repair attempts per stage")
    s.add_argument("--project-workers", type=int, default=4)
    s.add_argument("--mutant-workers", type=int, default=4)
    s.add_argument("--projects", nargs="*", help="restrict to these projects")
    s.add_argument("--all", action="store_true", help="use the full panel set")
    s.add_argument("--redo", action="store_true",
                   help="ignore saved results and rerun from scratch")
    s.add_argument("--topup", type=int, default=0, metavar="BUDGET",
                   help="keep proposing replacement mutants until --mutants are KILLED "
                        "or BUDGET have been proposed in total; also re-runs mutants "
                        "whose failure this code has since addressed "
                        "(F3 unsound, F5 no counterpart / error / exception)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("widen", help="stage 5: add a differential oracle (C-probe "
                                     "replay) to every KILLED mutant's test, in place")
    s.add_argument("--run", required=True, help="run name (results/<run>/), a COPY of "
                                                "a finished run")
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--project-workers", type=int, default=4)
    s.add_argument("--mutant-workers", type=int, default=4)
    s.add_argument("--projects", nargs="*")
    s.add_argument("--all", action="store_true")
    s.add_argument("--redo", action="store_true")
    s.set_defaults(fn=cmd_widen)

    s = sub.add_parser("report", help="summarise a run")
    s.add_argument("--run", default="pilot1")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("naive", help="the naive baseline: Rust tests from the C source")
    s.add_argument("--run", default="pilot1",
                   help="the finished funnel run whose mutants M' it is scored on")
    s.add_argument("--retries", type=int, default=3, help="compile-repair attempts")
    s.add_argument("--project-workers", type=int, default=4)
    s.add_argument("--mutant-workers", type=int, default=4)
    s.add_argument("--projects", nargs="*", help="restrict to these projects")
    s.add_argument("--all", action="store_true", help="use the full panel set")
    s.add_argument("--redo", action="store_true", help="ignore saved naive.json")
    s.set_defaults(fn=cmd_naive)

    s = sub.add_parser("rustnative",
                       help="the rust-native oracle arm: a test per M' from the M->M' diff")
    s.add_argument("--run", default="pilot1",
                   help="the finished funnel run whose mutants M' it is scored on")
    s.add_argument("--retries", type=int, default=3, help="gate/repair attempts")
    s.add_argument("--project-workers", type=int, default=4)
    s.add_argument("--mutant-workers", type=int, default=4)
    s.add_argument("--projects", nargs="*", help="restrict to these projects")
    s.add_argument("--all", action="store_true", help="use the full panel set")
    s.add_argument("--max-mutants", type=int,
                   help="cap mutants per project (a cheap pilot; the arm pays per mutant)")
    s.add_argument("--redo", action="store_true", help="ignore saved rustnative results")
    s.set_defaults(fn=cmd_rustnative)

    s = sub.add_parser("naive-report", help="pipeline vs naive vs GT vs rust-native")
    s.add_argument("--run", default="pilot1")
    s.set_defaults(fn=cmd_naive_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
