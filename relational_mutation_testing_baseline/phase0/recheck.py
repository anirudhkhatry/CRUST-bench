"""Re-probe every CBench project through the *live* harness (`rmt.cbench.Project`).

`charness.py` was a standalone reimplementation; this drives the real thing, so a
project marked usable here is one Stages 1/2 can actually build and run. Used as the
regression gate for harness changes: run it before and after, diff the usable sets.

  .venv/bin/python phase0/recheck.py            # probe all 100, write recheck.csv/.json
  .venv/bin/python phase0/recheck.py --out X    # write results/phase0/X.csv/.json
  .venv/bin/python phase0/recheck.py --baseline results/phase0/recheck_before.json
        # ...and diff against a previous run (regression check)
  .venv/bin/python phase0/recheck.py gfc tisp   # probe a subset

Usability is exactly the pipeline's requirement: compile all library sources, then link
them (minus the project's own main) against a generated test main, and run it.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmt.cbench import CBENCH, ROOT, Project  # noqa: E402

OUT = ROOT / "relational_mutation_testing_baseline/results/phase0"

# A real (if trivial) test main: it must compile, link and run, not just link.
PROBE_MAIN = """
#include <stdio.h>
int main(void) { printf("rmt-probe\\n"); return 0; }
"""


def probe(name):
    try:
        with Project(name) as p:
            r = p.run(PROBE_MAIN, timeout=30)
            return {
                "project": name,
                "usable": r.ok and r.returncode == 0,
                "stage": r.stage,
                "returncode": r.returncode,
                "n_lib_sources": len(p.lib_sources),
                "n_mains": len(p.mains),
                "header_only": p.header_only,
                "link_libs": " ".join(p.link_libs or []),
                "detail": (r.detail or r.stderr or "")[:1500],
            }
    except Exception as e:  # noqa: BLE001
        return {"project": name, "usable": False, "stage": "error", "returncode": -1,
                "n_lib_sources": 0, "n_mains": 0, "header_only": False,
                "link_libs": "", "detail": f"{type(e).__name__}: {e}"[:1500]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="*", help="default: all CBench projects")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="recheck", help="basename under results/phase0")
    ap.add_argument("--baseline", help="a previous recheck json to diff against")
    a = ap.parse_args(argv)

    names = a.projects or sorted(d.name for d in CBENCH.iterdir() if d.is_dir())
    print(f"probing {len(names)} projects with {a.jobs} workers", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(probe, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0 or i == len(names):
                print(f"  {i}/{len(names)}", flush=True)
    rows.sort(key=lambda r: r["project"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.out}.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / f"{a.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    usable = {r["project"] for r in rows if r["usable"]}
    print(f"\nusable: {len(usable)}/{len(rows)}")
    print("stages:", dict(Counter(r["stage"] for r in rows)))
    for r in rows:
        if not r["usable"]:
            print(f"  FAIL {r['project']:28s} {r['stage']}")

    if a.baseline:
        base = json.load(open(a.baseline))
        was = {r["project"] for r in base if r["usable"]}
        print(f"\nbaseline usable: {len(was)}")
        print(f"  recovered ({len(usable - was)}): {sorted(usable - was)}")
        print(f"  REGRESSED ({len(was - usable)}): {sorted(was - usable)}")
        return 1 if was - usable else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
