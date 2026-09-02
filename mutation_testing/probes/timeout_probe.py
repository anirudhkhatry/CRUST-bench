"""Probe how many mutants time out at a given cap, for one project.

The question this answers: are the ``Timeout`` outcomes in the stored runs genuine
non-terminating mutants, or slow-but-terminating ones that a bigger cap would resolve
into a real kill/survival? Run the same project at two caps and compare -- if the counts
match, the cap is irrelevant and raising it only buys wall-clock.

    python timeout_probe.py ulidgen --cap 10 --out /tmp/ulidgen_10s
"""

import argparse
import json
import sys
from pathlib import Path

# Scripts live one level down; put the study root on sys.path so ``harness`` imports
# work when this file is run directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.paths import RESULTS
from harness.mutate_shared import run_shared_mutation
from harness.split_crates import TEST_CRATE_NAMES

ROOT = RESULTS / "claude/opus5_sample"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("project")
    p.add_argument("--cap", type=float, required=True, help="min per-mutant test timeout, seconds")
    p.add_argument("--out", required=True, help="absolute output dir for outcomes.json")
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()

    ws = ROOT / "_workspaces" / a.project
    crates = {s: c for s, c in TEST_CRATE_NAMES.items()
              if (ws / c / "tests").is_dir() and any((ws / c / "tests").iterdir())}

    res = run_shared_mutation(
        ws, package=a.project, test_crates=crates, out_base=Path(a.out).resolve(),
        project_name=a.project, target_dir=(ROOT / "_shared_target" / a.project).resolve(),
        jobs=a.jobs, shard=None, timeout_mult=2.0, min_timeout=a.cap)

    print(json.dumps({"cap": a.cap, "shared": res["shared"],
                      "settings": res["settings"]}, indent=2))


if __name__ == "__main__":
    main()
