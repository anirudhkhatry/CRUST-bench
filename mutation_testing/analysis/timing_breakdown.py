"""Where does a mutant's wall time actually go?

Every full run so far was killed before it could print the ``build_s``/``test_s`` split,
so the "cargo overhead dominates" claim has been inference from ``plan_shared_mutant_set``
plus arithmetic, not measurement. This measures it directly on one project, with a warm
target dir, by timing each step a mutant pays for:

  no-op build      cargo test --no-run, nothing changed -- pure freshness check
  rebuild          same, after touching core -- freshness check + real recompile + relink
  cargo test run   cargo test --package tests-<X>, nothing changed
  direct exec      the test binary invoked straight, no cargo at all

The last two are the phase-3 vs. mutate_fast comparison for the per-setting step; the
first two say how much of a mutant is unavoidable compilation.
"""

import json
import os
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


def timed(cmd, cwd, env, n=3):
    """Median-ish: run n times, report each and the best."""
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        times.append(time.monotonic() - t0)
    return times, r.returncode


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else "ulidgen"
    ws = ROOT / "_workspaces" / project
    env = os.environ.copy()
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_TARGET_DIR"] = str((ROOT / "_shared_target" / project).resolve())
    env["CARGO_BUILD_JOBS"] = "8"

    crates = [c for c in CRATES if (ws / c / "tests").is_dir()]
    build = ["cargo", "test", "--no-run", "--package", project]
    for c in crates:
        build += ["--package", c]

    print(f"project={project} crates={len(crates)}")
    print("warming...", flush=True)
    subprocess.run(build, cwd=ws, capture_output=True, env=env)

    print("\n--- 1. no-op batched build (pure freshness check) ---", flush=True)
    ts, rc = timed(build, ws, env)
    print(f"  rc={rc}  {[f'{t:.2f}s' for t in ts]}  best={min(ts):.2f}s")
    noop = min(ts)

    print("\n--- 2. batched build after touching core (real rebuild) ---", flush=True)
    src = next((ws / "core" / "src").glob("*.rs"))
    rebuilds = []
    for _ in range(3):
        src.touch()
        t0 = time.monotonic()
        subprocess.run(build, cwd=ws, capture_output=True, env=env)
        rebuilds.append(time.monotonic() - t0)
    print(f"  touched {src.name}  {[f'{t:.2f}s' for t in rebuilds]}  best={min(rebuilds):.2f}s")

    print("\n--- 3. cargo test --package <crate> (no changes) ---", flush=True)
    cargo_total = 0.0
    for c in crates:
        ts, rc = timed(["cargo", "test", "--package", c], ws, env)
        cargo_total += min(ts)
        print(f"  {c:24} rc={rc} best={min(ts):.2f}s")
    print(f"  all {len(crates)} suites: {cargo_total:.2f}s")

    print("\n--- 4. direct test-binary exec (no cargo) ---", flush=True)
    r = subprocess.run(build + ["--message-format=json"], cwd=ws,
                       capture_output=True, text=True, env=env)
    exes = {}
    for line in r.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") \
                and msg.get("profile", {}).get("test"):
            pid = msg["package_id"]
            head, _, tail = pid.partition("#")
            name = tail.split("@")[0] if "@" in tail else (
                tail if tail and not tail[0].isdigit() else head.rstrip("/").split("/")[-1])
            exes.setdefault(name, []).append(msg["executable"])

    direct_total = 0.0
    for c in crates:
        best = 0.0
        for exe in exes.get(c, []):
            ts, rc = timed([exe], ws / c, env)
            best += min(ts)
        print(f"  {c:24} {len(exes.get(c, []))} binaries best={best:.2f}s")
        direct_total += best
    print(f"  all {len(crates)} suites: {direct_total:.2f}s")

    print("\n=== per-mutant cost model ===")
    print(f"  phase 3  (1 build + 4 cargo test) = {min(rebuilds):.2f} + {cargo_total:.2f} "
          f"= {min(rebuilds) + cargo_total:.2f}s")
    print(f"  fast     (1 build + direct exec)  = {min(rebuilds):.2f} + {direct_total:.2f} "
          f"= {min(rebuilds) + direct_total:.2f}s")
    print(f"  cargo no-op overhead per invocation ~ {noop:.2f}s "
          f"x 4 saved per mutant = {noop * 4:.2f}s")


if __name__ == "__main__":
    main()
