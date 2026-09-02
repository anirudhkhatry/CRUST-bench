"""Score the transpilation operators over the whole 4-settings Claude set, unattended.

Same harness as ``run_transpile_sample.py`` -- mutants from ``list_transpile_mutants``,
everything downstream untouched -- pointed at ``results_opus5_4settings`` instead of the
5-project sample, so the two runs remain directly comparable and the analysis reads both
with the same code.

Prerequisite: ``build_full_set_workspaces.py`` must have run. It assembles
``_workspaces/<project>`` in the split layout and records which projects have four green
baselines; this script mutates exactly that corpus, read from ``full_set_corpus.json``.
Running against the raw per-setting trees would put the tests back inside the mutated
crate and make viability -- hence the denominator -- setting-dependent again.

Why the knobs match the sample run exactly
------------------------------------------
``WORKERS x JOBS = 8 x 4`` on a 56-core box is the load ceiling measured safe; unbounded
fan-out has crashed this machine. ``MIN_TIMEOUT = 120`` is large headroom over suites that
run in well under a second, and the sample run's evidence was that hangs are genuine
non-terminating loops rather than slow-but-terminating ones, so a higher cap buys
wall-clock rather than verdicts. Changing any of these would make the runs incomparable.

Ordering is smallest mutant set first, so an overnight failure costs the least finished
work, and each project's results are written the moment it completes. ``--resume`` skips
projects that already have settings in the report.
"""

import argparse
import json
import os
import shutil
import time
import traceback
from datetime import datetime
import sys
from pathlib import Path

# ``mutate_fast`` reaches into ``run_mutation`` for the setting -> directory map, and
# importing that module pulls in the whole translation pipeline, whose OpenAI client
# constructs itself at import time and refuses to exist without a key. Nothing in a
# mutation run ever calls an endpoint, so a placeholder satisfies the import without
# putting a live credential in the process. A real key already in the environment wins.
os.environ.setdefault("OPENAI_API_KEY", "unused-by-mutation-runs")

# Scripts live one level down; put the study root on sys.path so ``harness`` imports
# work when this file is run directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.paths import RESULTS, DATA  # noqa: E402
from harness.mutate_fast import run_fast_mutation  # noqa: E402
from harness.list_transpile_mutants import list_transpile_mutants  # noqa: E402
from harness.split_crates import TEST_CRATE_NAMES  # noqa: E402
from harness.run_mutation import SETTING_DIRS  # noqa: E402

ROOT = RESULTS / "claude/results_opus5_4settings"
WS = ROOT / "_workspaces"
OUT = ROOT / "_transpile_out"
TARGET = ROOT / "_transpile_target"
FROZEN = ROOT / "_frozen_source"
REPORT = ROOT / "transpile_report.json"
CORPUS = DATA / "full_set_corpus.json"

MIN_TIMEOUT = 120.0
TIMEOUT_MULT = 2.0
WORKERS = 8
JOBS = 4


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _mutable(frozen_root):
    """Every frozen .rs path the harness could have touched.

    ``core`` is the frozen source minus ``src/bin``, so everything else -- including
    ``interfaces``, which the mutant lister skips -- belongs in the cleanliness check.
    Checking a file no mutant can edit costs nothing; skipping one that a mutant did edit
    silently poisons every later mutant of that project.
    """
    for f in frozen_root.rglob("*.rs"):
        if "bin" in f.relative_to(frozen_root).parts:
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


def corpus(include_partial=False):
    """Projects to run, smallest mutant set first.

    Default is the four-green set: only there is the four-way comparison over one shared
    denominator meaningful. ``include_partial`` adds the projects where the split left one
    or two suites red -- their mutants still carry per-setting and per-genre information,
    they just cannot enter the cross-config intersection.
    """
    c = json.loads(CORPUS.read_text())
    sel = list(c["all4"]) + (list(c["partial"]) if include_partial else [])
    counts = json.loads((DATA / "full_set_readiness.json").read_text())
    return sorted(sel, key=lambda p: counts[p]["n_mutants"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="*", help="default: the whole four-green corpus")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-target", action="store_true",
                    help="keep each project's target dir (default: delete after, ~1 GB each)")
    ap.add_argument("--include-partial", action="store_true",
                    help="also run projects where the split left 1-2 suites red")
    args = ap.parse_args()

    only = args.projects or corpus(args.include_partial)
    report = json.loads(REPORT.read_text()) if (args.resume and REPORT.exists()) else {}

    log(f"{len(only)} project(s) workers={WORKERS} jobs={JOBS} "
        f"min_timeout={MIN_TIMEOUT:.0f}s resume={args.resume} mutants=transpile")
    t_all = time.time()

    for i, name in enumerate(only, 1):
        if args.resume and report.get(name, {}).get("settings"):
            log(f"[{i}/{len(only)}] {name}: already done, skipping")
            continue

        ws = WS / name
        dirty = verify_clean(name)
        if dirty:
            log(f"[{i}/{len(only)}] {name}: core was DIRTY ({dirty[:3]}), restoring")
            restore(name)

        crates = {s: c for s, c in TEST_CRATE_NAMES.items()
                  if (ws / c / "tests").is_dir() and any((ws / c / "tests").iterdir())}
        n = len(list_transpile_mutants(ws, name))
        log(f"[{i}/{len(only)}] {name}: {len(crates)} suites, {n} transpilation mutants")
        t0 = time.time()
        td = (TARGET / f"{name}_transpile").resolve()
        try:
            res = run_fast_mutation(
                ws, package=name, test_crates=crates, out_base=OUT, project_name=name,
                target_dir=td, jobs=JOBS, shard=None,
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
        # 85 projects x ~1 GB of build artefacts is the one resource this run can actually
        # exhaust, and nothing downstream reads a target dir once outcomes.json is written.
        if not args.keep_target and td.is_dir():
            shutil.rmtree(td, ignore_errors=True)
        log(f"{name}: done in {(time.time() - t0) / 60:.1f} min"
            + (f" (restored {left} files)" if left else ""))

    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min")
    errs = [n for n, r in report.items() if "settings" not in r]
    log(f"{len(report) - len(errs)} ok, {len(errs)} failed"
        + (f": {errs}" if errs else ""))


if __name__ == "__main__":
    main()
