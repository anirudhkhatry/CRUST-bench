# Unit Test Interface Repair — Design

**Status:** Drafted 2026-04-17. Awaiting user approval.

## Goal

Extend `src/codex_expts/generate_unit_tests.py` with an opt-in capability to iteratively repair generated Rust unit tests so that they compile against the provided Rust interface definitions. "Repair" here means fixing build errors only — runtime `unimplemented!()` panics from interface stubs are expected and explicitly not considered failures.

## Non-Goals

- Fixing interface files themselves.
- Making tests pass at runtime (interfaces are unimplemented; runtime panics are expected).
- Reordering or deleting bin tests the LLM does not return.
- Modifying generation behavior when `--repair` is not set.

## High-Level Flow

```
generate_unit_tests.py --repair N ...

  (1) existing generation step
        - builds prompts, calls LLM, writes rust/src/bin/*.rs per benchmark

  (2) repair_loop(benchmarks, N, endpoint, config)
        for b in benchmarks: migrate_layout(b.proj_dir)     # one-time, idempotent
        baseline: cargo build + cargo test --no-run for every project (iter 0)
        remaining = projects with nonempty filtered errors
        for i in 1..=N:
            if not remaining: break
            UnitTestRepairer(remaining, prompter, iteration=i, endpoint, config).run()
            write returned bin files (rejecting interface-path writes)
            re-run cargo build + cargo test --no-run, filter errors, update remaining
        write summary.json per project
```

## CLI Surface

New flag on `generate_unit_tests.py`:

```
--repair N    # integer (default: 0). 0 means no repair (current behavior).
```

Semantics:

- `--repair` is processed **after** generation completes and all generated bin files are written to disk.
- When `--repair N` is combined with `--skip_existing`, generation is skipped for projects that already have bins but those projects still participate in the repair loop. This enables running repair as a standalone step.
- `--single_benchmark`, `--parallel`, `--include_c_source` are unchanged.

## One-Time Migration

`migrate_layout(proj_dir)` runs once per benchmark at the start of `repair_loop`. Idempotent.

Per project:

1. If `rust/src/interfaces/` exists and contains `*.rs` files, move every `*.rs` file from `rust/src/interfaces/` to `rust/src/` (overwrite if a same-named file already exists at `src/`). Remove `rust/src/interfaces/` afterwards if it becomes empty.
2. If `rust/src/interfaces/` does not exist, skip the move step.
3. Run `cargo build` once. If it returns any `E0583` (file not found for module) errors, regenerate `rust/src/lib.rs` from scratch: glob `rust/src/*.rs` (excluding `lib.rs` and `main.rs`), and write `pub mod <stem>;` lines in sorted order. Otherwise leave `lib.rs` alone.
4. Files under `rust/src/bin/*.rs` are never touched by migration.

Migration log is written to `rust/metadata/unit_test_repair/migration.log` describing: which files were moved, whether `lib.rs` was regenerated, and the list of modules declared.

## Repair Loop Architecture

New module `src/codex_expts/unit_test_repairer.py` defining class:

```
UnitTestRepairer(benchmarks, prompter, iteration, endpoint, config)
    .run() -> (results, updated_benchmarks)
```

Modeled after the existing `TestRepairer` in `src/test_repairer.py`: same endpoint branching (batched for Claude Opus 4.x / Gemini 3, `Pool` for others, sequential with 20s sleep for older Claudes), same shared-lock pattern.

Driver function `repair_loop(benchmarks, n, endpoint, config)` lives inside `generate_unit_tests.py`:

```
for b in benchmarks: migrate_layout(b.proj_dir)

remaining = []
for b in benchmarks:
    errors = build_and_get_errors(b, iter=0)
    if errors: remaining.append(b)

for i in 1..=n:
    if not remaining: break
    prompter = UnitTestInterfaceRepairPrompter(repair_prompt_path)
    repairer = UnitTestRepairer(remaining, prompter, iteration=i, endpoint, config)
    _, updated = repairer.run()
    for b in updated:
        write_returned_bin_files(b)
    new_remaining = []
    for b in updated:
        errors = build_and_get_errors(b, iter=i)
        if errors: new_remaining.append(b)
    remaining = new_remaining

write per-project summary.json
```

Projects iterate in lockstep. A project drops out of `remaining` as soon as its filtered-error list is empty after build + test compile. Build/test runs are sequential per project (to avoid cargo thrashing); the LLM calls within a single iteration are batched/parallel per existing infrastructure.

### `build_and_get_errors(benchmark, iter)`

1. `cargo build` in `benchmark.proj_dir / 'rust'` — capture stderr.
2. `cargo test --no-run` in same directory, 60s timeout — capture stderr (on timeout, synthesize a `TIMEOUT` error entry).
3. Parse combined stderr via `create_error_report()` from `src/utils/compile_rust_utils.py`.
4. Filter: keep only entries where `file_path` contains `src/bin/`.
5. Write artifacts for this iteration (see Artifact Storage).
6. Return filtered error list.

## Prompter

New module `src/codex_expts/unit_test_repair_prompter.py`:

```
UnitTestInterfaceRepairPrompter(system_prompt_path)
    .build_prompt(benchmark, iteration) -> [[{role, content}, ...]]
    .parse_response(response) -> list[{file_name, content}]   # delegates to find_best_parser
```

Prompt content (per project per iteration):

- **System prompt**: loaded from a new file `src/prompts/unit_test_prompts/unit_test_repair.prompt`. Content states: repair Rust unit tests so they compile against the fixed Rust interfaces; may modify only files under `src/bin/`; do NOT rewrite or propose changes to interface files; return one `{{filename.rs}}` + rust fence per bin test being rewritten; omit bins that do not need changes.
- **User content** (in order):
  1. `## Rust Interface Files (read-only, do not modify):` — every `rust/src/*.rs` file except `lib.rs`, `main.rs`, and anything under `src/bin/`.
  2. `## Current Test Files under src/bin/:` — every `rust/src/bin/*.rs` file.
  3. `## Build / Test Errors (filtered to src/bin/*.rs):` — structured list from `create_error_report()`, one line per error with `error_code`, `error_message`, `file_path:row:col`, and `error_snippet`.
  4. Closing instruction to rewrite only affected bin files, preserving test logic, and to return fenced blocks.

`parse_response` calls `find_best_parser` from `src/prompters/prompter_utils.py`.

## Write-Back Rules

`write_returned_bin_files(benchmark, parsed_files)`:

For each parsed file:

- Strip any directory component; use the basename only.
- If the basename matches an existing interface file at `rust/src/<name>.rs` (excluding `lib.rs`), **reject**: log warning, do not write.
- Otherwise, write to `rust/src/bin/<basename>`. Overwrite if present; create if new.
- Never create subdirectories.

Files present on disk but absent from the LLM response are preserved as-is.

## Artifact Storage

Per-project, per-iteration artifacts under `rust/metadata/unit_test_repair/iter_{i}/`:

```
prompt.json          # messages array sent to the LLM (absent at iter 0)
response.txt         # raw LLM text (absent at iter 0)
errors.json          # filtered parsed errors (the input to the next prompt)
build.stderr         # raw cargo build stderr
test.stderr          # raw cargo test --no-run stderr
written_files.json   # list of bin files actually written after this iteration's response (absent at iter 0)
```

Iteration 0 is the baseline: only `errors.json`, `build.stderr`, `test.stderr`.

Per-project summary at `rust/metadata/unit_test_repair/summary.json`:

```
{
  "iterations_used": int,          # number of repair iterations (LLM calls) actually executed, 0..N
  "converged": bool,
  "final_error_count": int,        # size of the filtered error list at the final iteration
  "per_iter_error_counts": [int, ...]   # one entry per iteration 0..iterations_used inclusive
}
```

Project-level migration log at `rust/metadata/unit_test_repair/migration.log`.

## Convergence and Failure Handling

**Converged** (per project) ⇔ the filtered error list (parsed errors whose `file_path` contains `src/bin/`) is empty after running both `cargo build` and `cargo test --no-run`. Note: convergence is defined on the filtered list, not on raw cargo exit code — if an error originates in an interface file (outside `src/bin/`), the LLM cannot fix it (per the read-only interface rule), so such errors are not counted toward non-convergence. Non-bin errors are still written to `build.stderr` / `test.stderr` for debugging but do not keep the project in `remaining`.

**Hard stops (driver-level):**

- `remaining` is empty → break early.
- Iteration counter reaches `N` → stop; non-converged projects are recorded in their `summary.json`.

**Edge cases:**

| Situation | Handling |
|---|---|
| Empty or unparseable LLM response | Log warning (`written_files.json = []`), leave bin files untouched, proceed to next iteration. |
| LLM returns a bin file not previously present | Accept and write it to `src/bin/<name>`. |
| LLM returns a file matching an interface name at `src/<name>.rs` | Reject, log warning, do not write. |
| `cargo test --no-run` timeout (60s) | Synthesize a `TIMEOUT` error; project stays in `remaining`. |
| Project with no `src/interfaces/` and no interfaces at `src/` | Skip migration, log; project will fail to build and will not converge (reported in `summary.json`). |
| Rate limit / API error | Bubble up from the endpoint layer; that iteration's bin files stay unchanged; next iteration retries. |

## Components Summary

New files:

- `src/codex_expts/unit_test_repairer.py` — `UnitTestRepairer` class.
- `src/codex_expts/unit_test_repair_prompter.py` — `UnitTestInterfaceRepairPrompter` class.
- `src/prompts/unit_test_prompts/unit_test_repair.prompt` — system prompt.

Modified files:

- `src/codex_expts/generate_unit_tests.py` — add `--repair N` CLI flag, `migrate_layout`, `build_and_get_errors`, `write_returned_bin_files`, `repair_loop` driver, invocation after the existing generation step.

Reused (unmodified):

- `src/utils/compile_rust_utils.py::create_error_report` — parse cargo stderr into structured errors.
- `src/prompters/prompter_utils.py::find_best_parser` — parse LLM response into file blocks.
- `src/endpoints/call_endpoint.py::get_result`, `get_results_batched` — endpoint invocation.

## Testing Plan

- Pick a single benchmark (e.g., `aes128_SIMD`) where current layout has `src/interfaces/*.rs`.
- Run `generate_unit_tests.py --single_benchmark aes128_SIMD --skip_existing --repair 3 --endpoint <model> --config <config>`.
- Verify:
  - `src/interfaces/` is gone, interface files now at `src/*.rs`.
  - `src/lib.rs` is consistent with files at `src/` (regenerated only if it was missing declarations).
  - `src/bin/*.rs` files exist and are modified across iterations when repair makes progress.
  - `rust/metadata/unit_test_repair/iter_0/errors.json` reflects the baseline error count.
  - Subsequent `iter_{i}/errors.json` monotonically trend toward zero (or the run exits early when clean).
  - `summary.json` records convergence status correctly.
  - `cargo build && cargo test --no-run` exits cleanly on a converged project.
  - Running `cargo test` on a converged project produces `unimplemented!()` panics at runtime, not compile errors.
