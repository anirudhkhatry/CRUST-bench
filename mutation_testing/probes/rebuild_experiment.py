"""Can the ~19 s per-mutant rebuild be made cheaper?

Established by measurement so far (ulidgen): the rebuild after touching `core` is ~60% of
a mutant's cost; it is NOT the cargo freshness check (2 s) and NOT linking (lld: 0.97x).
Dropping debuginfo gives 1.32x. What remains is genuine compilation of `core` plus the 11
test binaries that depend on it.

Three untested levers, measured here against a like-for-like baseline:

  incremental   `mutate_shared` sets CARGO_INCREMENTAL=0 and mutate_fast inherited it
                unexamined. Each mutant edits exactly one function body, which is the
                case rustc's red/green tracking exists for -- the other CGUs should be
                reusable. Nobody has checked whether that holds across the *dependent*
                crates, which is where most of the 19 s lives.

  -Zthreads     nightly's parallel rustc frontend. The dependency graph here is a wide
                fan-out (core, then 11 independent binaries), so cargo's own -j should
                already cover it; this tests whether per-crate frontend parallelism adds
                anything on top.

  opt-level=0   already the dev default, included only as a control -- if it moves, the
                profile is not what we think it is.

Realism note: a mutant edits ONE function body, so `src.touch()` (which invalidates the
whole file but changes no content) understates incremental's advantage. This script edits
a real function body instead, appending a no-op statement, to match what splicing does.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Scripts live one level down; put the study root on sys.path so ``harness`` imports
# work when this file is run directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.paths import RESULTS


ROOT = RESULTS / "claude/opus5_sample"
CRATES = ["tests-src", "tests-src_and_test", "tests-src_over_test", "tests-c_and_rust"]


def pick_source(ws):
    """The largest core source file -- most likely to hold a mutable function body."""
    return max((ws / "core" / "src").glob("*.rs"), key=lambda p: p.stat().st_size)


def rebuild_times(ws, build, env, n=4):
    """Rebuild after a REAL edit to a function body, n times. Always restores."""
    src = pick_source(ws)
    original = src.read_text()
    # Append a distinct statement inside the first `{` of a fn -- a body-only change, the
    # same shape as a spliced mutant.
    m = re.search(r"\n(pub )?fn [a-zA-Z_0-9]+\s*\([^)]*\)[^{]*\{", original)
    if not m:
        raise RuntimeError(f"no fn found in {src}")
    cut = m.end()
    out = []
    try:
        for i in range(n):
            edited = original[:cut] + f"\n    let _mutprobe{i} = {i};" + original[cut:]
            src.write_text(edited)
            t0 = time.monotonic()
            r = subprocess.run(build, cwd=ws, capture_output=True, text=True, env=env)
            out.append(time.monotonic() - t0)
            if r.returncode != 0:
                return out, r.stderr[-1200:]
    finally:
        src.write_text(original)
    return out, None


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else "ulidgen"
    ws = ROOT / "_workspaces" / project
    crates = [c for c in CRATES if (ws / c / "tests").is_dir()]
    build = ["cargo", "test", "--no-run", "--package", project]
    for c in crates:
        build += ["--package", c]

    def env_for(tag, incremental, rustflags):
        e = os.environ.copy()
        e["CARGO_INCREMENTAL"] = "1" if incremental else "0"
        e["CARGO_BUILD_JOBS"] = "8"
        e["CARGO_TARGET_DIR"] = str((ROOT / "_shared_target" / f"{project}_x_{tag}").resolve())
        e["RUSTFLAGS"] = rustflags
        return e

    DBG = "-Cdebuginfo=0"
    configs = [
        ("base(inc=0)",        False, DBG),
        ("inc=1",              True,  DBG),
        ("inc=1 +Zthreads=8",  True,  DBG + " -Zthreads=8"),
        ("inc=0 +Zthreads=8",  False, DBG + " -Zthreads=8"),
    ]

    print(f"project={project} crates={len(crates)} binaries=? (see build)\n")
    results = {}
    for tag, inc, flags in configs:
        env = env_for(re.sub(r"\W", "", tag), inc, flags)
        print(f"--- {tag} ---", flush=True)
        w = subprocess.run(build, cwd=ws, capture_output=True, text=True, env=env)
        if w.returncode != 0:
            print(f"  SETUP FAILED: {w.stderr[-600:]}\n")
            continue
        ts, err = rebuild_times(ws, build, env)
        if err:
            print(f"  FAILED: {err}\n")
            continue
        # First edit after a cold build still populates the incremental cache; the steady
        # state is what a 122-mutant loop actually pays.
        steady = min(ts[1:]) if len(ts) > 1 else min(ts)
        results[tag] = steady
        print(f"  {[f'{t:.2f}s' for t in ts]}  steady-state best={steady:.2f}s\n", flush=True)

    if results:
        base = results.get("base(inc=0)")
        print("=== rebuild per mutant (real function-body edit) ===")
        for k, v in results.items():
            sp = f"  ({base / v:.2f}x)" if base else ""
            print(f"  {k:20} {v:6.2f}s{sp}")


if __name__ == "__main__":
    main()
