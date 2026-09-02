"""Score the transpilation operators on the same 5 projects, unattended.

This is ``run_opus5_sample.py`` with one substitution: mutants come from
``list_transpile_mutants`` instead of cargo-mutants. Everything downstream -- the shared
build, the baseline gate, the per-setting verdicts, ``outcomes.json`` -- is untouched, so
the two runs are directly comparable and the analysis reads both with the same code.

Why the knobs match the cargo-mutants run exactly
-------------------------------------------------
``WORKERS x JOBS = 8 x 4`` on a 56-core box (~57% subscription) is the load ceiling that
was measured safe; unbounded fan-out has crashed this machine. ``MIN_TIMEOUT = 120`` is
300x headroom over a 0.4 s suite, and the evidence from the cargo-mutants run was that
hangs here are genuine non-terminating loops rather than slow-but-terminating ones, so a
higher cap buys wall-clock rather than verdicts. Changing any of these would make the two
runs incomparable, which is the whole point of the exercise.

Expected scale: 722 mutants against the cargo-mutants run's 1101, so roughly 2 hours.
Order is smallest mutant set first (ulidgen 63, clog ~140, emlang ~160, clhash ~170,
cset ~190) so an overnight failure costs the least finished work, and each project's
results are written the moment it completes.
"""

import json
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
from harness.list_transpile_mutants import list_transpile_mutants
from harness.split_crates import TEST_CRATE_NAMES
from harness.run_mutation import SETTING_DIRS

ROOT = RESULTS / "claude/opus5_sample"
WS = ROOT / "_workspaces"
OUT = ROOT / "_transpile_out"
TARGET = ROOT / "_shared_target"
FROZEN = ROOT / "_frozen_source"
REPORT = ROOT / "transpile_report.json"

PROJECTS = ["ulidgen", "clog", "emlang", "clhash", "cset"]

MIN_TIMEOUT = 120.0
TIMEOUT_MULT = 2.0
WORKERS = 8
JOBS = 4


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _mutable(frozen_root):
    """Every frozen .rs path that the harness is allowed to mutate.

    Mirrors list_transpile_mutants' own exclusions so the cleanliness check does not
    report a file as dirty that no mutant could have touched.
    """
    for f in frozen_root.rglob("*.rs"):
        parts = f.relative_to(frozen_root).parts
        if "bin" in parts:
            continue
        yield f


def verify_clean(project):
    """Assert core/ matches the frozen source before mutating it.

    A run that dies without unwinding leaves a mutant spliced into core/, and cloning a
    dirty core would silently apply that mutant on top of every other one. This has
    actually happened, so it is checked rather than assumed.
    """
    fz, core = FROZEN / project / "src", WS / project / "core" / "src"
    dirty = []
    for f in _mutable(fz):
        dst = core / f.relative_to(fz)
        if dst.is_file() and dst.read_bytes() != f.read_bytes():
            dirty.append(str(dst.relative_to(WS)))
    return dirty


def restore(project):
    fz, core = FROZEN / project / "src", WS / project / "core" / "src"
    n = 0
    for f in _mutable(fz):
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
        f"min_timeout={MIN_TIMEOUT:.0f}s resume={resume} mutants=transpile")
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
        n = len(list_transpile_mutants(ws, name))
        log(f"[{i}/{len(only)}] {name}: {len(crates)} suites, {n} transpilation mutants")
        t0 = time.time()
        try:
            res = run_fast_mutation(
                ws, package=name, test_crates=crates, out_base=OUT, project_name=name,
                target_dir=(TARGET / f"{name}_transpile").resolve(), jobs=JOBS, shard=None,
                timeout_mult=TIMEOUT_MULT, min_timeout=MIN_TIMEOUT,
                workers=WORKERS, runner="direct",
                mutant_lister=list_transpile_mutants)
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
                       "harness": "mutate_fast", "workers": WORKERS,
                       "mutants": "transpile"}
            md = ws / "metadata" / SETTING_DIRS[setting]
            md.mkdir(parents=True, exist_ok=True)
            (md / "mutation_transpile.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")

        report[name] = {"shared": res["shared"], "settings": res["settings"],
                        "n_mutants": n, "wall_s": round(time.time() - t0, 1)}
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
