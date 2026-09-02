"""Is the 17.8 s per-mutant rebuild compilation, or linking?

`timing_breakdown.py` showed the rebuild after touching `core` dominates a mutant's cost
(17.8 s of 31.6 s on ulidgen) -- far larger than the cargo freshness check the plan doc
blamed. `core` is a few hundred lines, so 17.8 s is not compiling it; the suspicion is
linking 11 test binaries, each of which statically links libstd and a pile of debuginfo.

Two cheap, independent knobs, measured here against an untouched baseline:

  debug=0   stop emitting debuginfo for the test profile. Mutation testing never reads a
            backtrace, so the information is generated, linked, and discarded.
  lld       rust-lld ships inside the toolchain (rustlib/.../bin/gcc-ld/ld.lld) and is
            typically several times faster than GNU ld at exactly this shape of work.

Each config touches core and rebuilds, three times, and reports the best.
"""

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
GCC_LD = ("/usr/local/rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/"
          "x86_64-unknown-linux-gnu/bin/gcc-ld")

PROFILE = """
[profile.test]
debug = 0
[profile.dev]
debug = 0
"""


def rebuild_times(ws, build, env, n=3):
    src = next((ws / "core" / "src").glob("*.rs"))
    out = []
    for _ in range(n):
        src.touch()
        t0 = time.monotonic()
        r = subprocess.run(build, cwd=ws, capture_output=True, text=True, env=env)
        out.append(time.monotonic() - t0)
        if r.returncode != 0:
            return out, r.stderr[-1500:]
    return out, None


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else "ulidgen"
    ws = ROOT / "_workspaces" / project
    manifest = ws / "Cargo.toml"
    original = manifest.read_text()

    crates = [c for c in CRATES if (ws / c / "tests").is_dir()]
    build = ["cargo", "test", "--no-run", "--package", project]
    for c in crates:
        build += ["--package", c]

    def env_for(target_sub, rustflags=None):
        e = os.environ.copy()
        e["CARGO_INCREMENTAL"] = "0"
        e["CARGO_BUILD_JOBS"] = "8"
        e["CARGO_TARGET_DIR"] = str((ROOT / "_shared_target" / f"{project}_{target_sub}").resolve())
        if rustflags:
            e["RUSTFLAGS"] = rustflags
        return e

    configs = [
        ("baseline", False, None),
        ("debug=0", True, None),
        ("lld", False, f"-Clink-arg=-fuse-ld=lld -Clink-arg=-B{GCC_LD}"),
        ("debug=0 + lld", True, f"-Clink-arg=-fuse-ld=lld -Clink-arg=-B{GCC_LD}"),
    ]

    print(f"project={project} crates={len(crates)}\n")
    results = {}
    try:
        for name, no_debug, flags in configs:
            manifest.write_text(original + (PROFILE if no_debug else ""))
            env = env_for(name.replace(" ", "").replace("=", "").replace("+", "_"), flags)
            print(f"--- {name}: warming cold target dir ---", flush=True)
            w = subprocess.run(build, cwd=ws, capture_output=True, text=True, env=env)
            if w.returncode != 0:
                print(f"  SETUP FAILED: {w.stderr[-800:]}\n")
                continue
            ts, err = rebuild_times(ws, build, env)
            if err:
                print(f"  BUILD FAILED: {err}\n")
                continue
            results[name] = min(ts)
            print(f"  rebuilds {[f'{t:.2f}s' for t in ts]}  best={min(ts):.2f}s\n", flush=True)
    finally:
        manifest.write_text(original)

    if results:
        base = results.get("baseline")
        print("=== rebuild cost per mutant ===")
        for k, v in results.items():
            sp = f"  ({base / v:.2f}x)" if base and v else ""
            print(f"  {k:16} {v:6.2f}s{sp}")


if __name__ == "__main__":
    main()
