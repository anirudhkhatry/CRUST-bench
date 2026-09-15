#!/usr/bin/env python
"""Good-vs-bad migration benchmarks. The only entrypoint.

    ./run_additional_datasets.py select --benchmark refactorbench_js
    ./run_additional_datasets.py run --benchmark refactorbench_js --tasks 3
    ./run_additional_datasets.py report --run pilot

Run it with the project venv: `.venv/bin/python run_additional_datasets.py ...`
"""
import argparse
import json
import sys
from pathlib import Path

# Runs are namespaced, matching the existing `results/<run>/<project>/` convention in
# rmt/ and pmb/, so a task's tree is results/additional_datasets/<run>/<benchmark>/<task>.
RESULTS = Path(__file__).resolve().parent / "results/additional_datasets"
BENCHMARKS = ("refactorbench_js",)


def _module(name):
    if name == "refactorbench_js":
        from additional_datasets.benchmarks import refactorbench_js
        return refactorbench_js
    sys.exit(f"unknown benchmark {name!r}; known: {', '.join(BENCHMARKS)}")


def cmd_select(a):
    _module(a.benchmark)
    from additional_datasets.adapters import js_vitest as JS
    names = JS.pure_logic_fixtures()
    print(f"{len(names)} pilot fixtures")
    for n in names:
        print(" ", n)


def cmd_run(a):
    mod = _module(a.benchmark)
    from additional_datasets.adapters import js_vitest as JS
    names = (a.fixtures or JS.pure_logic_fixtures())[:a.tasks]
    root = RESULTS / a.run if a.run else RESULTS
    rows = []
    for n in names:
        try:
            rows += mod.run_task(n, root, n_good=a.good, n_bad=a.bad)
        except Exception as e:  # noqa: BLE001 - one task must not sink the run
            print(f"  {n}: ERROR {type(e).__name__}: {e}", flush=True)
    out = root / "rows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out} ({len(rows)} rows)")


def cmd_report(a):
    from additional_datasets.score import report
    root = RESULTS / a.run if a.run else RESULTS
    rows = json.loads((root / "rows.json").read_text())
    text = report(rows, title=a.run or "pilot")
    (root / "REPORT.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="list the benchmark's task pool")
    s.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("run", help="run the protocol over the task pool")
    s.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    s.add_argument("--run", default="pilot", help="run name (results/additional_datasets/<run>)")
    s.add_argument("--tasks", type=int, default=3, help="how many tasks")
    s.add_argument("--fixtures", nargs="*", help="restrict to these task ids")
    s.add_argument("--good", type=int, default=3, help="GOOD candidates per task")
    s.add_argument("--bad", type=int, default=3, help="seeded BAD candidates per task")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="the spec §5 metrics table")
    s.add_argument("--run", default="pilot")
    s.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
