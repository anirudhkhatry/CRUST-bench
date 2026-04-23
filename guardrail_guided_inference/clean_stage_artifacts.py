"""Remove stage-A-only artifacts from a dataset directory.

Complements `prepare_for_agent.py`, which already drops AGENTS.md,
rust/metadata, and rust/guardrails.txt but does not know about codex
log files, cargo build output, or the agent's scratch workspace.

CLI:
    python clean_stage_artifacts.py --dataset_dir <path>

Library:
    from clean_stage_artifacts import clean_stage_artifacts
    counts = clean_stage_artifacts(Path(...))
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BENCH_ROOT_FILES = ("codex_stdout.log", "codex_stderr.log", "codex_run_summary.json")
BENCH_ROOT_DIRS = ("stage_a_workspace",)
RUST_SUBDIRS = ("target",)


def _remove_if_exists(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def clean_stage_artifacts(dataset_dir: Path) -> dict[str, int]:
    """Remove stage-A artifacts from every benchmark under dataset_dir.

    Returns a dict of counts keyed by artifact name.
    """
    if not dataset_dir.is_dir():
        raise ValueError(f"dataset_dir is not a directory: {dataset_dir}")

    counts = {
        "codex_stdout_log": 0,
        "codex_stderr_log": 0,
        "codex_run_summary_json": 0,
        "rust_target": 0,
        "stage_a_workspace": 0,
    }

    for benchmark_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for fname in BENCH_ROOT_FILES:
            key = fname.replace(".", "_")
            if _remove_if_exists(benchmark_dir / fname):
                counts[key] += 1

        for dname in BENCH_ROOT_DIRS:
            if _remove_if_exists(benchmark_dir / dname):
                counts[dname] += 1

        rust_dir = benchmark_dir / "rust"
        if rust_dir.is_dir():
            for dname in RUST_SUBDIRS:
                key = f"rust_{dname}"
                if _remove_if_exists(rust_dir / dname):
                    counts[key] += 1

    return counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove stage-A-only artifacts (codex logs, rust/target, stage_a_workspace) "
            "from each benchmark under a dataset dir. Idempotent."
        )
    )
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory to clean in place.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    counts = clean_stage_artifacts(dataset_dir)

    print(f"Cleaned: {dataset_dir}")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
