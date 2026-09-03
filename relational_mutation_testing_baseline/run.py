#!/usr/bin/env python
"""Relational mutation testing, C <-> Rust. The only entrypoint.

    ./run.py select                     choose the project set
    ./run.py run --mutants 10           run the funnel
    ./run.py report                     summarise a finished run

Run it with the project venv: `.venv/bin/python run.py ...`
"""
import argparse
import sys

from rmt import driver, pilot, report


def cmd_select(a):
    profs, picked = pilot.select(a.n)
    print(f"{len(profs)} eligible projects -> selected {len(picked)}\n")
    print(f"{'project':30} {'mode':12} {'loc':>6} {'files':>5} {'iface':>5}")
    for p in picked:
        print(f"{p['project']:30} {p['mode']:12} {p['c_loc']:6} "
              f"{p['c_files']:5} {p['n_iface_fns']:5}")


def cmd_run(a):
    projects = pilot.load()
    if a.projects:
        projects = [p for p in projects if p["project"] in a.projects]
        missing = set(a.projects) - {p["project"] for p in projects}
        if missing:
            sys.exit(f"not in the selected set: {', '.join(sorted(missing))} "
                     f"(run `select` first, or widen it with --n)")
    driver.run(a.run, projects, n_mutants=a.mutants, retries=a.retries,
               project_workers=a.project_workers, mutant_workers=a.mutant_workers,
               redo=a.redo)


def cmd_report(a):
    text = report.summarise(a.run)
    out = driver.RESULTS / a.run / "REPORT.md"
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
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("run", help="run the funnel")
    s.add_argument("--run", default="pilot1", help="run name (results/<run>/)")
    s.add_argument("--mutants", type=int, default=10, help="mutants per project")
    s.add_argument("--retries", type=int, default=3, help="repair attempts per stage")
    s.add_argument("--project-workers", type=int, default=4)
    s.add_argument("--mutant-workers", type=int, default=4)
    s.add_argument("--projects", nargs="*", help="restrict to these projects")
    s.add_argument("--redo", action="store_true",
                   help="ignore saved results and rerun from scratch")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="summarise a run")
    s.add_argument("--run", default="pilot1")
    s.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
