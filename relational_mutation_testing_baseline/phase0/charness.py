"""Phase 0b -- verify the C build harness (design spec S5).

What matters for Stages 1/2 is not "does the project build" but: can we compile the
library sources and link them, minus the project's own main, against a *generated test
main* that then runs. That is exactly `rmt.cbench.Project.run`, so this probes the live
harness rather than re-implementing it -- an earlier standalone reimplementation drifted
from the real thing and reported 89 usable where the harness managed 100.

Rows are keyed on the **RBench** project name, with the resolved CBench directory in the
`cbench` column, so `charness.csv` and `eligible.csv` join without ambiguity. (The two
names differ for `proj_2DPartInt` and `proj_42_Kocaeli_Printf`; joining on the wrong one
is what made §2.3 of the report disagree with `eligible.csv`.)

The project's own `make` is recorded alongside, for information only: a `make` success
gives no way to link a generated test main, so it is never the build path.

Writes charness.json + charness.csv.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmt.cbench import CBENCH, ROOT, Project, resolve_name  # noqa: E402
from rmt.rbench import RBENCH  # noqa: E402

OUT = ROOT / "relational_mutation_testing_baseline/results/phase0"
JOBS = int(os.environ.get("PHASE0_JOBS", "12"))
TIMEOUT = 180

# A real (if trivial) test main: it must compile, link and run, not merely link.
PROBE_MAIN = """
#include <stdio.h>
int main(void) { printf("rmt-probe\\n"); return 0; }
"""


def try_make(proj):
    """Attempt the project's own build. Non-destructive: build in a copy."""
    if not any((proj / n).exists() for n in ("Makefile", "makefile", "GNUmakefile")):
        return "no_makefile", ""
    with tempfile.TemporaryDirectory(prefix="rmt_mk_") as td:
        copy = Path(td) / proj.name
        try:
            shutil.copytree(proj, copy, symlinks=True)
        except OSError as e:
            return "copy_failed", str(e)[:300]
        try:
            r = subprocess.run(["make", "-j", "2"], cwd=copy, capture_output=True,
                               text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    return ("ok" if r.returncode == 0 else "failed"), (r.stdout + r.stderr)[-1200:]


def probe(name):
    """`name` is an RBench project name; the harness resolves it to a CBench directory."""
    cdir = resolve_name(name)
    mk_status, mk_detail = try_make(CBENCH / cdir)
    row = {"project": name, "cbench": cdir, "make": mk_status}
    try:
        with Project(name) as p:
            r = p.run(PROBE_MAIN, timeout=30)
            row.update({
                "gcc": r.stage if not r.passed else "ok",
                "usable": r.passed,
                "mode": p.mutation_target,
                "n_lib_sources": len(p.lib_sources),
                "n_mutable": len(p.mutable_files),
                "n_mains": len(p.mains),
                "link_libs": " ".join(p.link_libs or []),
                "cflags": " ".join(p.cflags or []),
                "detail": (r.detail or r.stderr or "")[:1200],
            })
    except Exception as e:  # noqa: BLE001
        row.update({"gcc": "error", "usable": False, "mode": "", "n_lib_sources": 0,
                    "n_mutable": 0, "n_mains": 0, "link_libs": "", "cflags": "",
                    "detail": f"{type(e).__name__}: {e}"[:1200]})
    row["build_path"] = "gcc" if row["usable"] else ("make_only" if mk_status == "ok"
                                                     else "none")
    row["make_detail"] = mk_detail[:600] if mk_status not in ("ok", "no_makefile") else ""
    return row


def main():
    names = sorted(d.name for d in RBENCH.iterdir() if (d / "Cargo.toml").exists())
    print(f"{len(names)} RBench projects, {JOBS} workers")
    orphans = sorted({d.name for d in CBENCH.iterdir() if d.is_dir()}
                     - {resolve_name(n) for n in names})
    if orphans:
        print(f"CBench dirs with no RBench counterpart: {orphans}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(probe, n): n for n in names}
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 20 == 0 or n == len(names):
                print(f"  {n}/{len(names)}", flush=True)

    rows.sort(key=lambda r: r["project"])
    with open(OUT / "charness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "charness.json").write_text(json.dumps(rows, indent=2))

    print("\nharness status:", dict(Counter(r["gcc"] for r in rows)))
    print("make status:   ", dict(Counter(r["make"] for r in rows)))
    print("mutation target:", dict(Counter(r["mode"] for r in rows)))
    print("usable by pipeline:", sum(r["usable"] for r in rows), "/", len(rows))
    for r in rows:
        if not r["usable"]:
            print(f"  FAIL {r['project']:28s} {r['gcc']}")
    print(f"wrote {OUT/'charness.csv'}")


if __name__ == "__main__":
    sys.exit(main())
