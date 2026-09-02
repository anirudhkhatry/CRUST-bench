"""Assemble split-crate workspaces for the whole 4-settings set, then probe each baseline.

``results_opus5_4settings`` predates the split-crates fix: each setting there is a full
copy of the project with that setting's tests inside ``src/bin``, i.e. inside the crate
that gets mutated. The transpilation harness needs the split layout instead -- one
workspace per project, ``core`` holding the frozen translation and one ``tests-<setting>``
crate per suite -- which is what ``split_crates.build_workspace`` produces and what
``results/claude/opus5_sample/_workspaces`` already contains for the 5-project sample.

Two things can go wrong in the split, and both are cheap to find out here rather than
five hours into a mutation run:

api offenders
    A test that reaches past the public API compiles fine while it lives inside the
    crate and stops compiling once it moves out. ``build_workspace`` reports these.

red baselines
    A suite that fails on the unmutated workspace fails on every mutant too, so every
    mutant reads as caught and the setting scores a flattering 100%. ``mutate_fast``
    drops such settings at run time, but a project that loses a setting is no longer
    usable for the four-way comparison at all -- so the count is needed up front.

Writes ``full_set_readiness.json``: per project the api offenders, the per-setting
baseline verdict and test time, and the transpilation-mutant count. The projects with
four green baselines are the corpus for the run.

Usage::

    python build_full_set_workspaces.py            # build + baseline everything
    python build_full_set_workspaces.py --workers 6
    python build_full_set_workspaces.py --resume   # keep projects already probed
    python build_full_set_workspaces.py cset clog  # only these
"""

import argparse
import json
import os
import queue
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from mutate_shared import baseline
from split_crates import TEST_CRATE_NAMES, build_workspace
from list_transpile_mutants import list_transpile_mutants

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results/claude/results_opus5_4settings"
WS = ROOT / "_workspaces"
READINESS = HERE / "full_set_readiness.json"

SETTING_DIRS = {"src": "src", "src+test": "src+test",
                "src/test": "src_over_test", "c+rust": "c_and_rust"}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_one(project):
    """Assemble ``_workspaces/<project>``. Returns (ws, tests_by_setting, offenders)."""
    frozen = ROOT / "_frozen_source" / project
    tests_by_setting = {}
    for setting, sdir in SETTING_DIRS.items():
        binp = ROOT / sdir / project / "src" / "bin"
        if binp.is_dir() and any(binp.glob("*.rs")):
            tests_by_setting[setting] = binp
    ws = WS / project
    info = build_workspace(project, frozen, tests_by_setting, ws)
    return ws, info


def probe_one(project, env, target_dir):
    """Build the workspace and run every suite unmutated."""
    t0 = time.time()
    ws, info = build_one(project)
    crates = {s: TEST_CRATE_NAMES[s] for s in info["tests"]}
    n_mutants = len(list_transpile_mutants(ws, project))
    env = {**env, "CARGO_TARGET_DIR": str(target_dir)}
    base = baseline(ws, crates, env)
    return {
        "project": project,
        "n_mutants": n_mutants,
        "api_offenders": [[s, Path(p).name, w] for s, p, w in info["api_offenders"]],
        "settings": {s: {"ok": bool(base[s]["ok"]),
                         "test_s": round(base[s]["test_s"], 3),
                         "output": base[s]["output"][-1200:] if not base[s]["ok"] else ""}
                     for s in sorted(crates)},
        "missing_settings": sorted(set(SETTING_DIRS) - set(crates)),
        "wall_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--workers", type=int, default=6,
                    help="projects probed concurrently; each gets its own target dir")
    ap.add_argument("--jobs", type=int, default=4, help="CARGO_BUILD_JOBS per worker")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    sel = args.projects or json.loads(
        (HERE / "full_set_projects.json").read_text())["all4"]
    done = json.loads(READINESS.read_text()) if (args.resume and READINESS.exists()) else {}
    todo = [p for p in sel if p not in done]

    env = os.environ.copy()
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_BUILD_JOBS"] = str(args.jobs)
    # Mutation testing never reads a backtrace, so debuginfo is generated and discarded.
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -Cdebuginfo=0").strip()

    WS.mkdir(parents=True, exist_ok=True)
    pool = queue.Queue()
    for i in range(args.workers):
        td = ROOT / f"_probe_target_{i}"
        td.mkdir(parents=True, exist_ok=True)
        pool.put(td)

    log(f"probing {len(todo)} of {len(sel)} project(s), workers={args.workers} "
        f"jobs={args.jobs}")
    t_all = time.time()
    n = 0

    def work(project):
        td = pool.get()
        try:
            return probe_one(project, env, td)
        except Exception:
            return {"project": project, "error": traceback.format_exc()[-2000:]}
        finally:
            pool.put(td)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(work, todo):
            n += 1
            p = res["project"]
            done[p] = res
            READINESS.write_text(json.dumps(done, indent=2))
            if "error" in res:
                log(f"[{n}/{len(todo)}] {p}: FAILED")
                continue
            green = [s for s, d in res["settings"].items() if d["ok"]]
            red = [s for s, d in res["settings"].items() if not d["ok"]]
            log(f"[{n}/{len(todo)}] {p}: {len(green)}/4 green, {res['n_mutants']} mutants"
                + (f", RED={red}" if red else "")
                + (f", missing={res['missing_settings']}" if res["missing_settings"] else "")
                + (f", {len(res['api_offenders'])} api offender(s)"
                   if res["api_offenders"] else "")
                + f"  ({res['wall_s']:.0f}s)")

    log(f"done in {(time.time() - t_all) / 60:.1f} min")
    summarise(done)


def summarise(done):
    all4 = [p for p, r in done.items()
            if "error" not in r and len(r["settings"]) == 4
            and all(d["ok"] for d in r["settings"].values())]
    partial = [p for p, r in done.items()
               if "error" not in r and p not in all4
               and any(d["ok"] for d in r["settings"].values())]
    dead = [p for p, r in done.items() if "error" in r
            or not any(d["ok"] for d in r["settings"].values())]
    n_mut = sum(done[p]["n_mutants"] for p in all4)
    print()
    print(f"4 green baselines : {len(all4):3}  ({n_mut} transpilation mutants)")
    print(f"1-3 green         : {len(partial):3}  {partial}")
    print(f"unusable          : {len(dead):3}  {dead}")
    out = HERE / "full_set_corpus.json"
    out.write_text(json.dumps({"all4": sorted(all4), "partial": sorted(partial),
                               "dead": sorted(dead), "n_mutants": n_mut}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    if "--summarise" in sys.argv:
        summarise(json.loads(READINESS.read_text()))
    else:
        main()
