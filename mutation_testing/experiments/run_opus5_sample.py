"""Mutation-test the 5 hardest Opus-5 projects, one after another, unattended.

The workspaces under ``results/claude/opus5_sample/_workspaces`` are already assembled by
``split_crates.py`` and already pruned to their green suites, so none of ``run_mutation``'s
LLM phases have anything to do here. This drives phase 4 directly, via ``mutate_fast``.

Settings and why
----------------
``WORKERS x JOBS`` is the load knob: 8 workers each allowed 4 cargo build jobs is up to 32
concurrent compile processes on a 56-core box (~57% subscription), leaving real headroom.
``mutate_fast`` also pins ``RUST_TEST_THREADS=1`` when workers > 1, so libtest does not add
a third multiplier on top -- unbounded fan-out is what has crashed this machine before.

``MIN_TIMEOUT = 120`` is a deliberate reduction from the 300 s originally chosen, and the
reason is that the measurements moved: timeouts were assumed to be the bottleneck, but the
real cost is a ~16-19 s recompile of `core` plus its 11 dependent test binaries on EVERY
mutant (cargo's freshness check is 2 s of it; lld, incremental compilation and the parallel
frontend were each measured and do not help). Against a suite that runs in 0.4 s, 120 s is
already 300x headroom, and the evidence says these hangs are genuine non-terminating loops
rather than slow-but-terminating ones -- ulidgen reports ~3 timeouts per 11 mutants at a
10 s cap and ~41 per 122 at 20 s, i.e. the count does not fall as the cap rises. At 300 s
the projected timeout budget alone is ~6.4 h on top of ~2.5 h of compute, which puts
finishing overnight at risk; at 120 s it is ~2.6 h, which does not. Raise it back if you
would rather have the margin than the certainty.

Order is smallest mutant set first (ulidgen 122, clog 231, emlang 238, clhash 297,
cset 306) so that an overnight failure costs the least finished work, and every project's
results are written the moment it completes rather than at the end of the run.
"""

import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Scripts live one level down; put the study root on sys.path so ``harness`` imports
# work when this file is run directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.paths import RESULTS
from harness.mutate_fast import run_fast_mutation
from harness.split_crates import TEST_CRATE_NAMES
from harness.run_mutation import SETTING_DIRS

ROOT = RESULTS / "claude/opus5_sample"
WS = ROOT / "_workspaces"
OUT = ROOT / "_mutants_out"
TARGET = ROOT / "_shared_target"
FROZEN = ROOT / "_frozen_source"
REPORT = ROOT / "rerun_report.json"

PROJECTS = ["ulidgen", "clog", "emlang", "clhash", "cset"]

MIN_TIMEOUT = 120.0
TIMEOUT_MULT = 2.0
WORKERS = 8
JOBS = 4


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def verify_clean(project):
    """Assert core/ matches the frozen source before mutating it.

    A run that dies without unwinding leaves a mutant spliced into core/, and cloning a
    dirty core would silently apply that mutant on top of every other one. This has
    actually happened twice, so it is checked rather than assumed.
    """
    fz, core = FROZEN / project / "src", WS / project / "core" / "src"
    dirty = []
    for f in fz.rglob("*.rs"):
        if "bin" in f.relative_to(fz).parts:
            continue
        dst = core / f.relative_to(fz)
        if dst.is_file() and dst.read_bytes() != f.read_bytes():
            dirty.append(str(dst.relative_to(WS)))
    return dirty


def restore(project):
    fz, core = FROZEN / project / "src", WS / project / "core" / "src"
    n = 0
    for f in fz.rglob("*.rs"):
        if "bin" in f.relative_to(fz).parts:
            continue
        dst = core / f.relative_to(fz)
        if dst.is_file() and dst.read_bytes() != f.read_bytes():
            shutil.copyfile(f, dst)
            n += 1
    return n


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")] or PROJECTS
    resume = "--resume" in sys.argv
    report = json.loads(REPORT.read_text()) if (resume and REPORT.exists()) else {}

    log(f"projects={only} workers={WORKERS} jobs={JOBS} "
        f"min_timeout={MIN_TIMEOUT:.0f}s resume={resume}")
    t_all = time.time()

    for i, name in enumerate(only, 1):
        if resume and report.get(name, {}).get("settings"):
            log(f"[{i}/{len(only)}] {name}: already done, skipping")
            continue

        ws = WS / name
        dirty = verify_clean(name)
        if dirty:
            log(f"[{i}/{len(only)}] {name}: core was DIRTY ({dirty}), restoring")
            restore(name)

        crates = {s: c for s, c in TEST_CRATE_NAMES.items()
                  if (ws / c / "tests").is_dir() and any((ws / c / "tests").iterdir())}
        log(f"[{i}/{len(only)}] {name}: {len(crates)} suites")
        t0 = time.time()
        try:
            res = run_fast_mutation(
                ws, package=name, test_crates=crates, out_base=OUT, project_name=name,
                target_dir=(TARGET / name).resolve(), jobs=JOBS, shard=None,
                timeout_mult=TIMEOUT_MULT, min_timeout=MIN_TIMEOUT,
                workers=WORKERS, runner="direct")
        except Exception:
            log(f"{name}: FAILED\n{traceback.format_exc()}")
            report[name] = {"error": traceback.format_exc()[-2000:]}
            REPORT.write_text(json.dumps(report, indent=2))
            restore(name)
            continue

        for setting, summary in res["settings"].items():
            summary = {**summary, "project": name, "package": name,
                       "test_package": crates[setting], "sharded": False, "shard": None,
                       "shared_mutants": True, "min_test_timeout": MIN_TIMEOUT,
                       "harness": "mutate_fast", "workers": WORKERS}
            md = ws / "metadata" / SETTING_DIRS[setting]
            md.mkdir(parents=True, exist_ok=True)
            (md / "mutation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        report[name] = {"shared": res["shared"], "settings": res["settings"],
                        "wall_s": round(time.time() - t0, 1)}
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

        left = restore(name)
        log(f"{name}: done in {(time.time() - t0) / 60:.1f} min"
            + (f" (restored {left} files)" if left else ""))

    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min")
    for name, r in report.items():
        if "settings" not in r:
            log(f"  {name}: ERROR")
            continue
        for s, sm in sorted(r["settings"].items()):
            log(f"  {name:8} {s:9} killed {sm['killed']}/{sm['viable_mutants']} "
                f"({sm['mutation_score']:.1%}) timeout {sm['timeout']} "
                f"unviable {sm['unviable']}")


if __name__ == "__main__":
    main()
