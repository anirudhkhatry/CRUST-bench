"""A/B the fast harness against the phase-3 reference, mutant for mutant.

`mutate_shared.run_shared_mutation` is the validated reference (116 verdicts matched
cargo-mutants exactly, per plan_shared_mutant_set.md). `mutate_fast.run_fast_mutation`
changes three things that could plausibly alter a verdict:

  * test binaries are executed directly instead of through `cargo test`, so cwd and
    CARGO_MANIFEST_DIR are set by us rather than by cargo;
  * mutants run concurrently in workspace clones rather than serially in the original;
  * debuginfo is off, which changes the binaries that actually run.

None of those *should* move a verdict. This checks that they don't, on a shard small
enough to run in minutes, with an identical timeout floor on both sides so a borderline
Timeout cannot masquerade as a disagreement.

    python validate_fast.py ulidgen --shard 1/12 --cap 10 --workers 6
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Scripts live one level down; put the study root on sys.path so ``harness`` imports
# work when this file is run directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.paths import RESULTS
from harness.mutate_fast import run_fast_mutation, validate_against_shared
from harness.mutate_shared import run_shared_mutation
from harness.split_crates import TEST_CRATE_NAMES

ROOT = RESULTS / "claude/opus5_sample"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("project", nargs="?", default="ulidgen")
    p.add_argument("--shard", default="1/12")
    p.add_argument("--cap", type=float, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()

    ws = ROOT / "_workspaces" / a.project
    crates = {s: c for s, c in TEST_CRATE_NAMES.items()
              if (ws / c / "tests").is_dir() and any((ws / c / "tests").iterdir())}

    ref_out, fast_out = Path("/tmp/val_ref"), Path("/tmp/val_fast")
    for d in (ref_out, fast_out):
        shutil.rmtree(d, ignore_errors=True)

    print(f"=== reference: mutate_shared, serial, cargo test ===", flush=True)
    ref = run_shared_mutation(
        ws, package=a.project, test_crates=crates, out_base=ref_out,
        project_name=a.project, target_dir=(ROOT / "_shared_target" / a.project).resolve(),
        jobs=8, shard=a.shard, timeout_mult=2.0, min_timeout=a.cap)

    print(f"\n=== fast: mutate_fast, {a.workers} workers, direct exec ===", flush=True)
    fast = run_fast_mutation(
        ws, package=a.project, test_crates=crates, out_base=fast_out,
        project_name=a.project, target_dir=(ROOT / "_shared_target" / f"{a.project}_val").resolve(),
        jobs=a.jobs, shard=a.shard, timeout_mult=2.0, min_timeout=a.cap,
        workers=a.workers, runner="direct")

    n, mismatches = validate_against_shared(ref_out, fast_out, a.project)
    print(f"\n=== comparison: {n} verdicts compared ===")
    if not mismatches:
        print("  IDENTICAL - every mutant got the same verdict in both harnesses")
    else:
        print(f"  {len(mismatches)} MISMATCHES:")
        for setting, key, ref_v, fast_v in mismatches[:40]:
            print(f"    {setting:9} {key[:70]:70} ref={ref_v} fast={fast_v}")

    rw, fw = ref["shared"]["wall_s"], fast["shared"]["wall_s"]
    print(f"\n  wall: reference {rw:.0f}s, fast {fw:.0f}s  ({rw / fw:.1f}x)"
          if fw else f"\n  wall: reference {rw:.0f}s")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
