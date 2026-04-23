"""Standalone tests for clean_stage_artifacts.clean_stage_artifacts.

Run with: python guardrail_guided_inference/tests/test_clean_stage_artifacts.py
Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from clean_stage_artifacts import clean_stage_artifacts


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_fake_benchmark(root: Path, name: str) -> Path:
    bench = root / name
    _touch(bench / "AGENTS.md", "translation agents")
    _touch(bench / "c" / "main.c", "int main(){}")
    _touch(bench / "rust" / "Cargo.toml", "[package]\nname=\"x\"\nversion=\"0.1.0\"\n")
    _touch(bench / "rust" / "src" / "lib.rs", "")
    _touch(bench / "rust" / "src" / "bin" / "generated_guardrails.rs", "pub fn main() {}")
    _touch(bench / "rust" / "target" / "debug" / "build.log", "cargo output")
    _touch(bench / "codex_stdout.log", "stage A stdout")
    _touch(bench / "codex_stderr.log", "stage A stderr")
    _touch(bench / "codex_run_summary.json", "{}")
    _touch(bench / "stage_a_workspace" / "oracle_bin", "binary")
    return bench


def test_removes_all_stage_a_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_benchmark(root, "proj_a")
        _make_fake_benchmark(root, "proj_b")

        counts = clean_stage_artifacts(root)

        for name in ("proj_a", "proj_b"):
            bench = root / name
            assert not (bench / "codex_stdout.log").exists(), f"{name}: codex_stdout.log not removed"
            assert not (bench / "codex_stderr.log").exists(), f"{name}: codex_stderr.log not removed"
            assert not (bench / "codex_run_summary.json").exists(), f"{name}: codex_run_summary.json not removed"
            assert not (bench / "rust" / "target").exists(), f"{name}: rust/target not removed"
            assert not (bench / "stage_a_workspace").exists(), f"{name}: stage_a_workspace not removed"

            # Preserved.
            assert (bench / "AGENTS.md").is_file(), f"{name}: AGENTS.md should survive"
            assert (bench / "c" / "main.c").is_file(), f"{name}: c/ should survive"
            assert (bench / "rust" / "Cargo.toml").is_file(), f"{name}: rust/Cargo.toml should survive"
            assert (bench / "rust" / "src" / "lib.rs").is_file(), f"{name}: rust/src/lib.rs should survive"
            assert (
                bench / "rust" / "src" / "bin" / "generated_guardrails.rs"
            ).is_file(), f"{name}: guardrail file must be preserved"

        # Reported counts: 2 benchmarks * 1 each of the five removable things = 10.
        assert counts["codex_stdout_log"] == 2
        assert counts["codex_stderr_log"] == 2
        assert counts["codex_run_summary_json"] == 2
        assert counts["rust_target"] == 2
        assert counts["stage_a_workspace"] == 2


def test_idempotent_on_clean_dir() -> None:
    """Running against an already-clean dataset must not error and must report zeros."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bench = root / "proj_c"
        _touch(bench / "AGENTS.md", "")
        _touch(bench / "rust" / "Cargo.toml", "")

        counts = clean_stage_artifacts(root)

        assert counts["codex_stdout_log"] == 0
        assert counts["codex_stderr_log"] == 0
        assert counts["codex_run_summary_json"] == 0
        assert counts["rust_target"] == 0
        assert counts["stage_a_workspace"] == 0


def test_skips_non_benchmark_entries() -> None:
    """Files and non-benchmark-like dirs at dataset root are ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("dataset readme", encoding="utf-8")
        (root / ".DS_Store").write_text("", encoding="utf-8")
        _make_fake_benchmark(root, "proj_d")

        clean_stage_artifacts(root)

        # Dataset-level files survive.
        assert (root / "README.md").is_file()
        # Benchmark was still cleaned.
        assert not (root / "proj_d" / "codex_stdout.log").exists()


if __name__ == "__main__":
    test_removes_all_stage_a_artifacts()
    test_idempotent_on_clean_dir()
    test_skips_non_benchmark_entries()
    print("OK: all clean_stage_artifacts tests passed")
