# Agent-Driven Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel, agent-driven pipeline that generates guardrails, translates C→Rust using codex, and finishes with a ground-truth evaluation overlay — without modifying any file of the existing `generate_guardrails.py` / `prepare_for_agent.py` pipeline. Both pipelines run independently.

**Architecture:** All deliverables are new files. The orchestrator chains the existing (unmodified) `prepare_for_agent.py` with a new `clean_stage_artifacts.py` helper that removes the stage-A-only artifacts the existing script doesn't know about (`codex_stdout.log`, `codex_stderr.log`, `codex_run_summary.json`, `rust/target/`, `stage_a_workspace/`). After stage B, the orchestrator chains `copy_parallel.py` and `paste_bin_files.py` (both pre-existing in `src/codex_expts/`) to produce a `stageB_codex_output_gt_tests/` directory with the RBench ground-truth tests overlaid. Audit directories (`initial/`, `stageA/`, `initial_s_B/`, `stageB/`) live alongside existing run outputs; the `*_codex_output/` and `*_codex_output_gt_tests/` dirs are auto-produced or explicitly produced by the final two steps.

**Tech Stack:** Python 3 (new helper), Bash (new orchestrator), codex CLI (via the existing `src/codex_expts/run_codex.py`), cargo.

**Spec:** `docs/superpowers/specs/2026-04-23-agent-driven-guardrails-design.md`

**Constraint (from user):** The existing pipeline must remain runnable as-is so the old and new pipelines can be compared. Every change is additive; no existing file is touched.

---

## File Structure

**New files only:**
- `guardrail_guided_inference/prompts/guardrail_agent.AGENTS.md` — stage A spec for codex
- `guardrail_guided_inference/prompts/translation_agent.AGENTS.md` — stage C spec for codex
- `guardrail_guided_inference/clean_stage_artifacts.py` — post-copy cleanup the existing prepare_for_agent.py doesn't do
- `guardrail_guided_inference/run_agent_pipeline.sh` — end-to-end orchestrator
- `guardrail_guided_inference/tests/test_clean_stage_artifacts.py` — standalone test (no pytest dependency)

**Modified files:** none.

**Reuse:**
- `guardrail_guided_inference/prepare_for_agent.py` — unchanged, called by the orchestrator for the copy + `AGENTS.md` install.
- `src/codex_expts/run_codex.py` — unchanged, called by the orchestrator for both codex runs.

---

## Task 1: clean_stage_artifacts.py helper + tests

**Files:**
- Create: `guardrail_guided_inference/clean_stage_artifacts.py`
- Create: `guardrail_guided_inference/tests/test_clean_stage_artifacts.py`

This helper runs AFTER `prepare_for_agent.py` copies `stageA/` → `initial_s_B/`. `prepare_for_agent.py` already drops `AGENTS.md`, `rust/metadata/`, `rust/guardrails.txt` — this helper fills the gap for stage-A-specific artifacts.

- [ ] **Step 1: Write the failing test**

Create `guardrail_guided_inference/tests/test_clean_stage_artifacts.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/tests/test_clean_stage_artifacts.py
```

Expected: `ModuleNotFoundError: No module named 'clean_stage_artifacts'` (the helper hasn't been written yet).

- [ ] **Step 3: Implement the helper**

Create `guardrail_guided_inference/clean_stage_artifacts.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/tests/test_clean_stage_artifacts.py
```

Expected stdout: `OK: all clean_stage_artifacts tests passed`, exit 0.

- [ ] **Step 5: Smoke-test the CLI**

Run:
```bash
python /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/clean_stage_artifacts.py --help
```

Expected: argparse help text, no traceback.

- [ ] **Step 6: Commit**

```bash
git add guardrail_guided_inference/clean_stage_artifacts.py \
        guardrail_guided_inference/tests/test_clean_stage_artifacts.py
git commit -m "clean_stage_artifacts: new helper for agent-driven pipeline

Removes per-benchmark codex_stdout.log, codex_stderr.log,
codex_run_summary.json, rust/target/, and stage_a_workspace/.
Complements prepare_for_agent.py (which drops AGENTS.md,
rust/metadata, rust/guardrails.txt) so the agent-driven orchestrator
can chain two codex runs without the second run skipping benchmarks
whose codex_stdout.log was left behind by the first.

Existing prepare_for_agent.py is untouched — the old pipeline still
runs unchanged."
```

---

## Task 2: Translation AGENTS.md template

**Files:**
- Create: `guardrail_guided_inference/prompts/translation_agent.AGENTS.md`

Spec document for codex. Acceptance test: file exists and is non-empty.

- [ ] **Step 1: Write the content**

Create `guardrail_guided_inference/prompts/translation_agent.AGENTS.md`:

```markdown
# Objective

Translate the C project in `c/` into an equivalent safe Rust implementation in `rust/`, satisfying the fixed guardrail contract in `rust/src/bin/generated_guardrails.rs`.

# Required process

1. Inspect and understand the C source files in `c/` before making changes.
2. Read the Rust project structure and the entry point interface definitions in:
   - `rust/`
   - `rust/src/interfaces/`
3. Read the guardrail contract at `rust/src/bin/generated_guardrails.rs`. This file is the fixed contract you must satisfy. You MUST NOT modify it.
4. The interface files are the **entry points** — the functions, structs, enums, and types that the guardrails exercise. Treat them as the primary contract your implementation must satisfy.
5. Implement the required functionality in safe Rust inside `rust/src/`, strictly conforming to the entry point definitions provided for each file in `rust/src/interfaces/`.
6. Each file in `rust/src/interfaces/` should be implemented in `rust/src/` with the same name but without the `unimplemented!()` macro, and instead with a concrete implementation that reproduces the behavior of the C code AND satisfies the guardrail assertions.
7. You may define additional internal types, helper functions, or modules — but the entry point signatures and types must be preserved exactly, and the guardrail file must not be modified.

# Requirements

- Preserve the existing Rust entry point interfaces exactly.
- Implement concrete Rust code in `rust/src/` based on the behavior of the C code.
- Reproduce the C implementation's behavior faithfully.
- Prefer safe, idiomatic Rust abstractions where possible, while preserving correctness.
- Every assertion in `rust/src/bin/generated_guardrails.rs` must pass against your implementation.

# Constraints

- Use only safe Rust.
- Do not use `unsafe`.
- Do not modify:
  - `rust/src/interfaces/` (interface definitions)
  - `rust/src/bin/generated_guardrails.rs` (guardrail contract)
  - source files in `c/`
  - `Cargo.toml`

# Dispute release valve

If, while translating, you become convinced that a specific guardrail assertion contradicts the C semantics — for example, the guardrail asserts on internal plumbing that the C code legitimately leaves unspecified, or the guardrail's own expected value disagrees with what the C code demonstrably produces — do NOT modify the guardrail file to work around it. Instead:

1. Continue implementing the rest of the contract as faithfully as you can.
2. Write a file at the benchmark root named `.guardrail_dispute.md` containing:
   - The specific assertion you believe is wrong (file name + line range).
   - The C semantics you observed.
   - The value the C code produces vs. what the guardrail expects.
3. Terminate the task with that dispute written; a human will review.

This valve exists because the guardrail is machine-generated and occasionally wrong. Silent workarounds (editing the guardrail, weakening assertions, stubbing the failing probe) defeat the purpose of guardrail-guided translation and are not acceptable.

# Validation

After implementation, run:

1. `cargo build` — must succeed without errors.
2. `cargo test --bin generated_guardrails` — must succeed (all guardrail assertions pass).

Iterate on your implementation until both succeed, or a `.guardrail_dispute.md` has been written per the dispute valve above.

# Repair

If `cargo build` fails, debug and fix the implementation until build succeeds without errors.

If `cargo test --bin generated_guardrails` fails, the failure is (in almost all cases) a real semantic divergence between your Rust implementation and the C code. Re-read the C, locate the behavior the failing assertion is probing, and correct the Rust. Do NOT modify the guardrail file.

# Completion criteria

The task is complete only when:

- The Rust implementation matches the expected behavior of the C code.
- All required entry point interfaces are implemented without modification.
- The guardrail file at `rust/src/bin/generated_guardrails.rs` has not been modified.
- `cargo build` succeeds.
- `cargo test --bin generated_guardrails` succeeds — OR a `.guardrail_dispute.md` has been written at the benchmark root documenting why a specific guardrail contradicts the observed C semantics.
- No disallowed files were changed.
```

- [ ] **Step 2: Verify file is present and non-empty**

Run:
```bash
test -s /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/prompts/translation_agent.AGENTS.md && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add guardrail_guided_inference/prompts/translation_agent.AGENTS.md
git commit -m "prompts: add translation_agent.AGENTS.md for stage B

Derived from dataset/*/AGENTS.md. Adds: guardrail target required
(cargo test --bin generated_guardrails), guardrail file in do-not-modify
list, dispute release valve via .guardrail_dispute.md to prevent silent
contract gaming."
```

---

## Task 3: Guardrail-generation AGENTS.md template

**Files:**
- Create: `guardrail_guided_inference/prompts/guardrail_agent.AGENTS.md`

- [ ] **Step 1: Write the content**

Create `guardrail_guided_inference/prompts/guardrail_agent.AGENTS.md`:

```markdown
# Objective

Generate a Rust guardrail harness at `rust/src/bin/generated_guardrails.rs` that captures the public contract of the C project in `c/` and is self-consistent against an oracle built from the C code itself. This file will be a fixed contract for a later translation agent; it must be correct before you terminate.

# Inputs available to you

- `c/` — the C source and headers you are building a guardrail for.
- `rust/src/interfaces/*.rs` — the authoritative entry-point signatures (pub fn / pub struct / pub enum / pub const). These are the contract surface.
- `rust/Cargo.toml` — the crate definition. Do not modify.
- You may write scratch files anywhere under `stage_a_workspace/` at the benchmark root. A downstream step will delete that directory wholesale.

# Do not modify

- `c/` — the C source is ground truth.
- `rust/src/interfaces/` — interface definitions are the fixed surface.
- `rust/Cargo.toml` — crate configuration is out of scope.

# Required process (follow in order)

## 1. Interface anchoring

Read every `*.rs` file under `rust/src/interfaces/`. Extract every `pub fn`, `pub struct`, `pub enum`, and `pub const`. This set is the non-negotiable `INTERFACE_FNS` / `INTERFACE_TYPES` for your guardrail. You do not get to decide which interface symbols are "important" — the interfaces file decides for you. Every `pub fn` in an interfaces file must either be covered by a `validate_semantics_*` probe OR be listed in an explicit exclusion set with a one-line reason.

## 2. C contract reading

For each interface `pub fn`, locate the corresponding C implementation in `c/`. Record its observable contract: return value, output buffers that cross the public API boundary (parameters the caller passes in and inspects afterward), and caller-visible state mutations.

You may assert on:
- return values,
- the contents of output buffers that the public C API writes to,
- state the caller can observe through subsequent public API calls.

You may NOT assert on:
- byte layouts of buffers consumed only internally within the library,
- intermediate tables, key schedules, or other plumbing not exposed by the C header,
- ordering of writes to internal scratch memory.

Test internal structures transitively through their consumers. If the C header does not expose a getter for a piece of state, the guardrail does not get to make claims about its layout.

## 3. Build the C and capture authoritative fixtures

Build the C project. Try in order:
1. `make` at the benchmark root (if a Makefile exists).
2. `cc c/*.c -o stage_a_workspace/oracle_bin` (direct compile, for single-file projects).
3. A one-off build script you write at `stage_a_workspace/build.sh` — use this for projects with custom layouts or multiple TUs.

Then write a small C driver at `stage_a_workspace/driver.c` (or similar) that:
- calls each interface function over a deterministic seed list (hardcoded inputs — NOT random per run, so reruns are reproducible),
- prints each `input → output` pair, with byte buffers hex-encoded.

Run the driver. Capture its stdout. These `(input, output)` pairs are the **authoritative fixtures**. Bake them as literals into your Rust guardrail's `validate_semantics_*` probes — compare crate outputs directly against these captured values. Do NOT write a synthesised Rust reference implementation and compare the crate against it; the C is the reference.

**If the C cannot be built** after trying the three approaches above (missing system deps, exotic toolchain, non-portable code), skip the oracle-derived fixtures, add a leading comment inside the guardrail file documenting the limitation (one line), and still satisfy gates A, B, D below. Do not fabricate fixtures.

## 4. Write the guardrail file

Write `rust/src/bin/generated_guardrails.rs` containing:

- `pub const BEHAVIORAL_GUARDRAILS: &[&str] = &[...]` — human-readable behavioral rules.
- `pub const SYNTACTIC_GUARDRAILS: &[&str] = &[...]` — human-readable syntactic rules.
- Authoritative fixtures baked in as `const` byte arrays / literals.
- One `pub fn validate_semantics_<name>()` per covered interface fn; each probe calls the crate's public API with a fixture input and `assert_eq!`s the result against the fixture output.
- `pub fn header_required_functions() -> &'static [&'static str]` — returns the list of every `pub fn` declared in `rust/src/interfaces/`.
- `pub fn semantics_covered_public_functions() -> &'static [&'static str]` — returns the list of fns that have a `validate_semantics_*` probe.
- `pub const UNREACHABLE_FROM_PUBLIC: &[&str]` and/or `pub const DISPATCH_ONLY: &[&str]` — explicit exclusion lists. Every excluded fn must have a one-line justification comment on the line above.
- `pub fn validate_guardrail_completeness()` — asserts `INTERFACE_FNS ⊆ covered ∪ DISPATCH_ONLY ∪ UNREACHABLE_FROM_PUBLIC`.
- `#[cfg(test)]` tests that call the `validate_*` functions so `cargo test --bin generated_guardrails` exercises them.
- `fn main() {}` — harmless, keeps the bin compilable standalone.

## 5. Self-check (hard gates — do not terminate until all pass)

**Gate A — cargo check compiles.**
Run from `rust/`:
```
cargo check --bin generated_guardrails
```
If this fails, fix the guardrail file until it compiles. Do not terminate with a non-compiling guardrail.

**Gate B — coherence.**
Verify: `INTERFACE_FNS ⊆ semantics_covered ∪ DISPATCH_ONLY ∪ UNREACHABLE_FROM_PUBLIC`. The `validate_guardrail_completeness()` function encodes this, but you should check it by eye too: if `header_required_functions()` lists `ft_printf`, then either `semantics_covered_public_functions()` contains `ft_printf`, or `DISPATCH_ONLY` / `UNREACHABLE_FROM_PUBLIC` contains it. No silent drops.

**Gate C — harness not at fault.**
Run from `rust/`:
```
cargo test --bin generated_guardrails -- --nocapture
```
Interpret the results:
- All tests pass → continue.
- A test panics with the frame pointing at `src/bin/generated_guardrails.rs` AND both sides of the failing `assert_eq!` can be reduced without calling into the crate (e.g., `ref_foo(input)` compared against a literal constant, both of which live in the guardrail file) → the harness itself is buggy. Fix the guardrail and re-run. Do NOT terminate with a buggy harness.
- A test panics with the panic originating from `unimplemented!()` or `todo!()` inside the crate under test (i.e., the unimplemented interface stubs) → expected at this stage. Log which probes hit unimplemented stubs and continue.
- Any other panic → inspect. If the guardrail is asking the crate for something the crate's stubs can't yet provide, that's OK. If the guardrail is asking for something the C code does NOT produce, the guardrail is wrong — fix it.

**Gate D — no synthesised-reference traps.**
If any `validate_semantics_*` probe compares the output of a Rust helper `ref_*` against a literal fixture, and that Rust helper was written by you rather than produced from C-oracle output, delete the helper. The C oracle output IS the reference; a hand-written Rust reference is a second point of failure.

## 6. Terminate

Once gates A–D pass, terminate. The guardrail file is ready for the translation agent.

If you have exhausted reasonable effort and cannot satisfy one of the gates (e.g., C cannot be built and gate C still fails for non-stub reasons), leave a leading comment in the guardrail file documenting which gates could not be satisfied and why, and terminate anyway. A human will review.

# Completion criteria

- `rust/src/bin/generated_guardrails.rs` exists and is non-empty.
- Gate A: `cargo check --bin generated_guardrails` passes.
- Gate B: coherence holds (`INTERFACE_FNS ⊆ covered ∪ excluded`).
- Gate C: no harness-internal panics from `cargo test --bin generated_guardrails`.
- Gate D: no synthesised `ref_*` comparisons against literal fixtures.
- `c/`, `rust/src/interfaces/`, and `rust/Cargo.toml` are unmodified.
- All intermediate build artifacts and scratch files live under `stage_a_workspace/`.
```

- [ ] **Step 2: Verify file is present and non-empty**

Run:
```bash
test -s /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/prompts/guardrail_agent.AGENTS.md && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add guardrail_guided_inference/prompts/guardrail_agent.AGENTS.md
git commit -m "prompts: add guardrail_agent.AGENTS.md for stage A

Encodes the improvements.md checks (coherence, harness-not-at-fault,
C-oracle fixtures, interface anchoring, API-boundary assertions)
as hard gates a codex agent must satisfy before terminating. Replaces
the multi-stage pipeline with a single agent-driven stage, without
touching the existing generate_guardrails.py flow.

Agent scratch work lives under stage_a_workspace/ at benchmark root;
the downstream clean_stage_artifacts step deletes that dir wholesale."
```

---

## Task 4: Orchestrator shell script

**Files:**
- Create: `guardrail_guided_inference/run_agent_pipeline.sh`

- [ ] **Step 1: Write the orchestrator**

Create `guardrail_guided_inference/run_agent_pipeline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Agent-driven guardrail + translation pipeline.
#
# This script runs ALONGSIDE the existing generate_guardrails.py pipeline;
# it does not modify any existing file. Outputs land in a timestamped
# directory so old and new pipeline outputs coexist.
#
# Usage: run_agent_pipeline.sh [MODEL] [SRC_DATASET] [BUDGET_USD]
#   MODEL         codex model name                 (default: gpt-5.4)
#   SRC_DATASET   source dataset dir to copy from  (default: guardrail_guided_inference/dataset)
#   BUDGET_USD    per-benchmark budget per stage   (default: 10)
#
# Produces four directories under
#   guardrail_guided_inference/results/generated_datasets/<MODEL>_<TS>_agent/
#     initial/        stage-A input (guardrail AGENTS.md installed)
#     stageA/         stage-A output (codex writes guardrails)
#     initial_s_B/    stage-B input (translation AGENTS.md installed, guardrail preserved)
#     stageB/         stage-B output (codex translates)

REPO_ROOT="/nas/CRUST-bench-COLM/CRUST-bench"
GGI_ROOT="$REPO_ROOT/guardrail_guided_inference"
CODEX_EXPTS="$REPO_ROOT/src/codex_expts"

MODEL="${1:-gpt-5.4}"
SRC_DATASET="${2:-$GGI_ROOT/dataset}"
BUDGET_USD="${3:-10}"
NUM_WORKERS="${NUM_WORKERS:-16}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="$GGI_ROOT/results/generated_datasets/${MODEL}_${RUN_TS}_agent"

GUARDRAIL_TEMPLATE="$GGI_ROOT/prompts/guardrail_agent.AGENTS.md"
TRANSLATION_TEMPLATE="$GGI_ROOT/prompts/translation_agent.AGENTS.md"

echo "== agent-driven guardrail pipeline =="
echo "MODEL=$MODEL"
echo "SRC_DATASET=$SRC_DATASET"
echo "BUDGET_USD=$BUDGET_USD"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "RUN_ROOT=$RUN_ROOT"
echo

for f in "$GUARDRAIL_TEMPLATE" "$TRANSLATION_TEMPLATE"; do
  if [ ! -f "$f" ]; then
    echo "missing required template: $f" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT"

# 1. Prepare initial input: existing prepare_for_agent.py copies dataset and
#    installs guardrail AGENTS.md per benchmark.
echo "[1/10] prepare initial input"
python "$GGI_ROOT/prepare_for_agent.py" \
  --input_dir   "$SRC_DATASET" \
  --output_dir  "$RUN_ROOT/initial" \
  --agents_template "$GUARDRAIL_TEMPLATE"

# 2. Snapshot -> stageA (codex mutates stageA in place; initial/ is preserved
#    as an audit reference).
echo "[2/10] snapshot initial -> stageA"
cp -r "$RUN_ROOT/initial" "$RUN_ROOT/stageA"

# 3. Run guardrail-generation agent (existing run_codex.py; unchanged).
echo "[3/10] run guardrail-generation agent"
python "$CODEX_EXPTS/run_codex.py" \
  --input_path "$RUN_ROOT/stageA" \
  --model "$MODEL" \
  --budget_usd "$BUDGET_USD" \
  --num_workers "$NUM_WORKERS"

# 4. cargo clean each stageA benchmark before it feeds stage B.
echo "[4/10] cargo clean stageA benchmarks"
for bench in "$RUN_ROOT"/stageA/*/; do
  if [ -f "${bench}rust/Cargo.toml" ]; then
    (cd "${bench}rust" && cargo clean) || true
  fi
done

# 5. Existing prepare_for_agent.py copies stageA -> initial_s_B and installs
#    translation AGENTS.md. It does NOT know about codex logs / rust/target /
#    stage_a_workspace — those are handled in step 6.
echo "[5/10] prepare_for_agent: stageA -> initial_s_B (install translation AGENTS.md)"
python "$GGI_ROOT/prepare_for_agent.py" \
  --input_dir   "$RUN_ROOT/stageA" \
  --output_dir  "$RUN_ROOT/initial_s_B" \
  --agents_template "$TRANSLATION_TEMPLATE"

# 6. New helper strips stage-A artifacts (codex logs, rust/target/,
#    stage_a_workspace/) from initial_s_B so run_codex.py re-runs on
#    every benchmark.
echo "[6/10] clean_stage_artifacts on initial_s_B"
python "$GGI_ROOT/clean_stage_artifacts.py" --dataset_dir "$RUN_ROOT/initial_s_B"

# 7. Snapshot -> stageB (codex mutates stageB in place; initial_s_B is
#    preserved as an audit reference).
echo "[7/10] snapshot initial_s_B -> stageB"
cp -r "$RUN_ROOT/initial_s_B" "$RUN_ROOT/stageB"

# 8. Run translation agent (existing run_codex.py; unchanged).
#    As a side effect, run_codex.py also flattens each benchmark's rust/ crate
#    into $RUN_ROOT/stageB_codex_output/<bench>/ for downstream evaluation.
echo "[8/10] run translation agent"
python "$CODEX_EXPTS/run_codex.py" \
  --input_path "$RUN_ROOT/stageB" \
  --model "$MODEL" \
  --budget_usd "$BUDGET_USD" \
  --num_workers "$NUM_WORKERS"

# 9. Clone stageB_codex_output -> stageB_codex_output_gt_tests so the groundtruth
#    overlay in step 10 leaves the stage-B output intact as an audit reference.
echo "[9/10] copy_parallel: stageB_codex_output -> stageB_codex_output_gt_tests"
python "$CODEX_EXPTS/copy_parallel.py" \
  "$RUN_ROOT/stageB_codex_output" \
  "$RUN_ROOT/stageB_codex_output_gt_tests"

# 10. Overlay RBench groundtruth tests into each benchmark's src/bin/ (replaces
#     the agent-generated guardrail bin with the canonical RBench tests.rs).
#     The evaluation step (cargo test) is intentionally left to the operator's
#     existing reporting tooling in src/utils/compile_rust_utils.py.
echo "[10/10] paste_bin_files: install RBench groundtruth tests into gt_tests dir"
python "$CODEX_EXPTS/paste_bin_files.py" "$RUN_ROOT/stageB_codex_output_gt_tests"

echo
echo "== done =="
echo "output dirs under: $RUN_ROOT"
echo "  initial/                       stage-A input snapshot"
echo "  stageA/                        stage-A output (guardrails generated)"
echo "  stageA_codex_output/           flattened stage-A rust/ crates"
echo "  initial_s_B/                   stage-B input snapshot"
echo "  stageB/                        stage-B output (translation)"
echo "  stageB_codex_output/           flattened stage-B rust/ crates"
echo "  stageB_codex_output_gt_tests/  stage-B with RBench groundtruth tests overlaid"
```

- [ ] **Step 2: Make executable**

Run:
```bash
chmod +x /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/run_agent_pipeline.sh
```

- [ ] **Step 3: Syntax check**

Run:
```bash
bash -n /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/run_agent_pipeline.sh && echo OK
```

Expected: `OK`.

- [ ] **Step 4: Verify template existence check fires**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference
mv prompts/guardrail_agent.AGENTS.md prompts/guardrail_agent.AGENTS.md.bak
bash run_agent_pipeline.sh gpt-5.4 dataset 10; rc=$?
mv prompts/guardrail_agent.AGENTS.md.bak prompts/guardrail_agent.AGENTS.md
echo "exit=$rc"
```

Expected: stderr `missing required template: ...guardrail_agent.AGENTS.md`; `exit=1`.

- [ ] **Step 5: Verify existing pipeline is still runnable (regression guard)**

Confirm the existing entry points still parse:
```bash
python /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/generate_guardrails.py --help | head -5
python /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/prepare_for_agent.py --help | head -5
bash -n /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/run_guardrails_gpt54.sh && echo old-pipeline: OK
```

Expected: help text for both Python scripts, `old-pipeline: OK`. No traceback, no diff.

- [ ] **Step 6: Commit**

```bash
git add guardrail_guided_inference/run_agent_pipeline.sh
git commit -m "run_agent_pipeline.sh: orchestrate stage A + stage B end-to-end

Chains existing prepare_for_agent.py -> existing run_codex.py (stage A)
-> cargo clean -> existing prepare_for_agent.py -> new clean_stage_artifacts
-> existing run_codex.py (stage B) across four audit directories
(initial, stageA, initial_s_B, stageB).

No existing file is modified. The old generate_guardrails.py pipeline
remains runnable unchanged via run_guardrails_gpt54.sh."
```

---

## Task 5: End-to-end smoke test (operator-run, not automated)

**Files:** none modified — this documents how to validate the pipeline manually.

- [ ] **Step 1: Stage a single-benchmark dataset**

```bash
mkdir -p /tmp/ggi_smoke_src
cp -r /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/dataset/bitset /tmp/ggi_smoke_src/
```

- [ ] **Step 2: Run the orchestrator with a small budget**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference
bash run_agent_pipeline.sh gpt-5.4 /tmp/ggi_smoke_src 2
```

Expected: ten `[N/10]` step lines, seven output dirs under `results/generated_datasets/gpt-5.4_<ts>_agent/` (four audit snapshots plus `stageA_codex_output/`, `stageB_codex_output/`, and `stageB_codex_output_gt_tests/`), no `set -e` termination.

- [ ] **Step 3: Verify outputs and isolation**

```bash
ts_dir=$(ls -1dt /nas/CRUST-bench-COLM/CRUST-bench/guardrail_guided_inference/results/generated_datasets/gpt-5.4_*_agent | head -1)
echo "ts_dir=$ts_dir"
ls "$ts_dir"

test -d "$ts_dir/initial/bitset/c"        && echo "initial: OK"
test -d "$ts_dir/stageA/bitset/c"         && echo "stageA: OK"
test -d "$ts_dir/initial_s_B/bitset/c"    && echo "initial_s_B: OK"
test -d "$ts_dir/stageB/bitset/c"         && echo "stageB: OK"
test -f "$ts_dir/stageA/bitset/AGENTS.md" && echo "stageA AGENTS.md: OK"
test -f "$ts_dir/stageB/bitset/AGENTS.md" && echo "stageB AGENTS.md: OK"

# Guardrail file must be preserved into stage B.
test -f "$ts_dir/initial_s_B/bitset/rust/src/bin/generated_guardrails.rs" \
  && echo "initial_s_B guardrail: OK" \
  || echo "initial_s_B guardrail: MISSING (stage A may have budget-exited)"
test -f "$ts_dir/stageB/bitset/rust/src/bin/generated_guardrails.rs" \
  && echo "stageB guardrail: OK" \
  || echo "stageB guardrail: MISSING"

# Stage-A artifacts must NOT appear in initial_s_B/stageB.
test ! -f "$ts_dir/initial_s_B/bitset/codex_stdout.log"   && echo "initial_s_B codex log stripped: OK"
test ! -d "$ts_dir/initial_s_B/bitset/rust/target"        && echo "initial_s_B rust/target stripped: OK"
test ! -d "$ts_dir/initial_s_B/bitset/stage_a_workspace"  && echo "initial_s_B stage_a_workspace stripped: OK"
```

Expected: all `OK` lines. Missing guardrail is informational — a $2 budget may not be enough for the stage-A agent to finish its checks.

- [ ] **Step 4: No commit needed (test only).**

---

## Self-Review

**Spec coverage:**
- Component 1 (guardrail AGENTS.md) → Task 3. ✓
- Component 2 (translation AGENTS.md) → Task 2. ✓
- Component 3 (stage-A-specific cleanup) → Task 1, realised as a NEW helper `clean_stage_artifacts.py` (not a modification to `prepare_for_agent.py`). ✓
- Component 4 (orchestrator) → Task 4. ✓
- Four-directory audit layout → Task 4 Step 1. ✓
- `cargo clean` between stages → Task 4 Step 1, `[4/10]` block. ✓
- Ground-truth evaluation tail (`copy_parallel.py` + `paste_bin_files.py`) → Task 4 Step 1, `[9/10]` and `[10/10]` blocks. ✓
- Budget threading via direct `run_codex.py` call → Task 4 Step 1. ✓

**"Existing pipeline remains as-is" constraint:**
- No modification to `prepare_for_agent.py`. ✓
- No modification to `generate_guardrails.py`. ✓
- No modification to `src/codex_expts/run_codex.py` or `src/codex_expts/run.sh`. ✓
- No modification to any existing AGENTS.md under `dataset/`. ✓
- `run_guardrails_gpt54.sh` — untouched; regression-guarded in Task 4 Step 5. ✓

**Placeholder scan:**
- No `TBD` / `TODO` / `fill in` / `similar to Task N`. Every code step is complete.

**Type/name consistency:**
- `clean_stage_artifacts` function name and CLI `--dataset_dir` flag match between Task 1 implementation and Task 4 orchestrator invocation. ✓
- Return keys of `clean_stage_artifacts` (`codex_stdout_log`, `codex_stderr_log`, `codex_run_summary_json`, `rust_target`, `stage_a_workspace`) are consistent between the implementation and the tests. ✓
- Template filenames (`guardrail_agent.AGENTS.md`, `translation_agent.AGENTS.md`) match between Tasks 2/3 and Task 4's orchestrator. ✓
- `stage_a_workspace/` referenced consistently in Tasks 1 (cleanup), 3 (agent scratch), 4 (cleaned post-copy). ✓
- `rust/src/bin/generated_guardrails.rs` referenced consistently across Tasks 1 (preserved), 2 (do-not-modify), 3 (write target), 4 (smoke-verify). ✓
