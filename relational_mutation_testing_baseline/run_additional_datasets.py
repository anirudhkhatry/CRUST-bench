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

from additional_datasets.arms.assemble import N_DRAWS
from additional_datasets.runner import LeakageError

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
    failures = []
    for n in names:
        try:
            rows += mod.run_task(n, root, n_good=a.good, n_bad=a.bad, draws=a.draws)
        except LeakageError as e:
            # A leakage control fired: the run is not scoreable, full stop (spec §15).
            # It must not be caught with the per-task handler below, dropped to a
            # stdout line, and then have rows.json/REPORT.md written as if nothing
            # happened -- so it aborts the whole run rather than sinking one task.
            sys.exit(f"ABORT: leakage control fired on {n}: {e}\n"
                     f"This run is not scoreable. No rows.json was written.")
        except Exception as e:  # noqa: BLE001 - one task must not sink the run
            print(f"  {n}: ERROR {type(e).__name__}: {e}", flush=True)
            failures.append({"task": n, "error": f"{type(e).__name__}: {e}"})
    out = root / "rows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))
    print(f"wrote {out} ({len(rows)} rows, {len(failures)} failure(s))")


def cmd_report(a):
    from additional_datasets.score import report
    root = RESULTS / a.run if a.run else RESULTS
    blob = json.loads((root / "rows.json").read_text())
    # Older rows.json files (before failures were tracked) were a bare list of rows.
    rows, failures = (blob["rows"], blob.get("failures", [])) if isinstance(blob, dict) else (blob, [])
    text = report(rows, title=a.run or "pilot", failures=failures)
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
    # ONE knob for both arms. ARM N's draw count and ARM M's mutant count must stay equal
    # -- that equality is the matched control -- so there is deliberately no --mutants to
    # set alongside it. Lower it for a cheap pilot; `run_task` asserts the two arms were
    # given the same number.
    s.add_argument("--draws", type=int, default=N_DRAWS,
                   help=f"draws per arm: ARM N suites AND ARM M mutants (default "
                        f"{N_DRAWS}); they are always equal")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="the spec §5 metrics table")
    s.add_argument("--run", default="pilot")
    s.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
