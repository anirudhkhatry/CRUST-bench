# Agent-driven guardrail generation and translation pipeline

**Date:** 2026-04-23
**Scope:** `guardrail_guided_inference/`
**Supersedes:** the elaborate multi-stage pipeline proposed in
`guardrail_guided_inference/improvements.md` §§3–4 (static coherence check,
run-against-stubs, oracle fixtures, FFI oracle, etc.).

## Motivation

`improvements.md` identifies two failure classes in the current
`generate_guardrails.py` output:

1. `proj_42_Kocaeli_Printf` — harness is internally contradictory
   (required set vs. covered set).
2. `aes128_SIMD` — harness's synthesised reference disagrees with its
   own fixture, and asserts on over-specified internal byte layouts.

The doc's proposed fix is a five-stage pipeline: static set-theory check,
`cargo test` against stubs, C-oracle-derived fixtures, interface-anchored
contract, API-boundary prompt rule. Each stage is bespoke code.

This design replaces that pipeline with a single **agent-driven** stage.
Codex already has the capabilities the pipeline would build
(shell, cargo, file I/O, iteration). Instead of templating each check,
the guardrail-generation `AGENTS.md` prescribes the checks as a hard
checklist the agent must satisfy before terminating. The observed
failure classes become the agent's hard gates.

## Non-goals

- Replacing `generate_guardrails.py` in-tree. It stays for comparison.
- FFI oracle (§4.3b). Still tier-2; out of scope here.
- Changing `src/codex_expts/run_codex.py`. The orchestrator calls it directly.

## End-to-end flow

Four working directories per run, under
`guardrail_guided_inference/results/generated_datasets/<model>_<ts>_agent/`:

| Dir            | Role                                                                 |
| -------------- | -------------------------------------------------------------------- |
| `initial/`     | Stage-A input: `dataset/` copied + guardrail `AGENTS.md` installed.  |
| `stageA/`      | Copy of `initial/`. Codex mutates this in-place (guardrail gen).     |
| `initial_s_B/` | `stageA/` cleaned + translation `AGENTS.md` installed. Guardrail kept. |
| `stageB/`      | Copy of `initial_s_B/`. Codex mutates this in-place (translation).   |

`initial/` and `initial_s_B/` are audit snapshots — never mutated by codex.
`initial/` ↔ `stageA/` diff is exactly what the guardrail agent did;
`initial_s_B/` ↔ `stageB/` diff is exactly what the translation agent did.

## Deliverables

1. `guardrail_guided_inference/prompts/guardrail_agent.AGENTS.md` — stage A agent spec.
2. `guardrail_guided_inference/prompts/translation_agent.AGENTS.md` — stage C agent spec.
3. `guardrail_guided_inference/prepare_for_agent.py` — extended cleanup list.
4. `guardrail_guided_inference/run_agent_pipeline.sh` — orchestrator.

## Component 1 — guardrail_agent.AGENTS.md

Spec the codex agent reads at the root of each benchmark in stage A. Must produce
`rust/src/bin/generated_guardrails.rs` that captures the C project's public
contract and is self-consistent against an oracle built from the C code.

**Required agent workflow (numbered checklist in the file):**

1. **Interface anchoring.** Read `rust/src/interfaces/*.rs`. List every
   `pub fn`, `pub struct`, `pub enum`, `pub const` — this is the
   non-negotiable `INTERFACE_FNS` / `INTERFACE_TYPES` set. (§4.4)
2. **C contract reading.** Read C source and headers. For each interface
   fn, find the corresponding C implementation and note its observable
   contract: return values, output buffers crossing the public boundary,
   caller-visible mutations. Do not plan to assert on internal byte
   layouts of buffers not exposed by the C header. (§4.5)
3. **Build the C.** Try `make`, then `cc c/*.c`, then a one-off build
   script. Write a small C driver that exercises the interface functions
   over a deterministic seed list and prints `input → output` pairs as
   hex. Capture these as **authoritative fixtures**. (§4.3a)
   If the C cannot be built after reasonable attempts (missing system
   deps, non-portable build), skip oracle-derived fixtures, document
   the limitation in a leading comment inside the generated file, and
   still satisfy gates §4.1, §4.2, §4.4, §4.5. Do not fabricate fixtures.
4. **Write the guardrail file.** `rust/src/bin/generated_guardrails.rs`
   with `BEHAVIORAL_GUARDRAILS`, `SYNTACTIC_GUARDRAILS`, fixtures baked
   in as literals, `validate_semantics_*` probes, `header_required_functions()`,
   `semantics_covered_public_functions()`, `UNREACHABLE_FROM_PUBLIC` /
   `DISPATCH_ONLY` exclusions, and `validate_guardrail_completeness()`.
5. **Self-check (hard gates — agent must not terminate until all pass):**
   - `cargo check --bin generated_guardrails` compiles.
   - **Coherence.** `INTERFACE_FNS ⊆ semantics_covered ∪ DISPATCH_ONLY ∪ UNREACHABLE_FROM_PUBLIC`. (§4.1)
   - **Harness-not-at-fault.** `cargo test --bin generated_guardrails`.
     If a test panics with the panic frame pointing at
     `src/bin/generated_guardrails.rs` AND both sides of the failing
     `assert_eq!` reduce without calling into the crate, the harness
     itself is buggy — fix it and re-run. Panics from `unimplemented!()`
     inside the crate under test are expected at this stage. (§4.2)
   - No synthesised reference helpers (`ref_*`) compared against other
     synthesised values; references only cross-checked with C oracle output.
6. **Do not modify** `c/`, `rust/src/interfaces/`, or `rust/Cargo.toml`.
7. **Scratch workspace.** Keep all intermediate build artifacts under
   `stage_a_workspace/` at benchmark root. `prepare_for_agent.py` removes
   this directory wholesale when producing stage B input.

**Budget boundary.** The per-benchmark budget is enforced by
`run_codex.py`. If the agent exits without passing gates, the benchmark
is flagged failed for stage A; stage B still proceeds against whatever
`generated_guardrails.rs` file exists. If no guardrail file exists,
translation runs without a guardrail target (degraded, documented).

## Component 2 — translation_agent.AGENTS.md

Derived from the existing `guardrail_guided_inference/dataset/*/AGENTS.md`.
Same objective: translate `c/` → safe Rust in `rust/src/`, preserve
interfaces, `cargo build` succeeds. Additions:

- **Guardrail target (required).** `rust/src/bin/generated_guardrails.rs`
  is the fixed contract for this benchmark. Completion requires both
  `cargo build` AND `cargo test --bin generated_guardrails` to pass.
- **Do-not-modify list** now includes `rust/src/bin/generated_guardrails.rs`
  (alongside `c/`, `rust/src/interfaces/`, `Cargo.toml`). Satisfy it
  as-written; do not edit it.
- **Dispute release valve.** If a guardrail assertion appears to
  contradict the C semantics, the agent leaves a `.guardrail_dispute.md`
  at benchmark root naming the specific assertion and why. It does NOT
  modify the guardrail file to work around. This prevents the agent
  from silently gaming the contract.
- **Completion criteria.** Rust matches C semantics, interfaces
  untouched, guardrail untouched, `cargo build` succeeds,
  `cargo test --bin generated_guardrails` succeeds, no disallowed files
  changed.

## Component 3 — prepare_for_agent.py extension

Current behavior: removes `AGENTS.md`, `rust/metadata/`, `rust/guardrails.txt`
per benchmark; optionally installs a template.

**Problems for two-stage flow:**

- After stage A, each benchmark root has `codex_stdout.log`,
  `codex_stderr.log`, `codex_run_summary.json`. `run_codex.py` skips any
  benchmark whose `codex_stdout.log` exists — the second `run.sh` would
  no-op on every benchmark.
- Stage A's `cargo` runs produce `rust/target/` (hundreds of MB).
  Copying through to stage B is slow and wasteful.
- Stage-A scratch files (C driver, oracle binary, fixtures.txt) shouldn't
  bleed into stage B. Handled by the `stage_a_workspace/` convention
  (see Component 1, item 7); the prepare script just removes that dir.

**Change.** Keep the existing public API (`prepare_dataset` and CLI
args unchanged). Inside the per-benchmark loop, extend the removal
list to also drop:

- `codex_stdout.log`, `codex_stderr.log`, `codex_run_summary.json` (at benchmark root)
- `rust/target/` (cargo build output)
- `stage_a_workspace/` (at benchmark root)

Count removals and include them in the return tuple and in `main()`'s
printout. Behaviour remains idempotent — step 1 of the pipeline (copy
from pristine `dataset/`) finds none of these to remove, which is fine.

## Component 4 — run_agent_pipeline.sh

Lives at `guardrail_guided_inference/run_agent_pipeline.sh`. Bash, `set -euo pipefail`.
Positional args with defaults:

- `$1` model (default `gpt-5.4`)
- `$2` source dataset dir (default `guardrail_guided_inference/dataset`)
- `$3` budget USD per benchmark per stage (default `10`)

```bash
RUN_ROOT=results/generated_datasets/${MODEL}_${RUN_TS}_agent

# 1. Prepare initial input: dataset + guardrail AGENTS.md
python prepare_for_agent.py \
  --input_dir   $SRC_DATASET \
  --output_dir  $RUN_ROOT/initial \
  --agents_template prompts/guardrail_agent.AGENTS.md

# 2. Snapshot -> stageA (codex mutates stageA in place)
cp -r $RUN_ROOT/initial $RUN_ROOT/stageA

# 3. Run guardrail-generation agent
python ../src/codex_expts/run_codex.py \
  --input_path $RUN_ROOT/stageA --model $MODEL --budget_usd $BUDGET --num_workers 16

# 4. cargo clean each stageA benchmark before it feeds stage B
for bench in $RUN_ROOT/stageA/*/; do
  if [ -f "${bench}rust/Cargo.toml" ]; then
    (cd "${bench}rust" && cargo clean) || true
  fi
done

# 5. Produce stage B input: strip codex logs + stage_a_workspace,
#    swap AGENTS.md, preserve rust/src/bin/generated_guardrails.rs
python prepare_for_agent.py \
  --input_dir   $RUN_ROOT/stageA \
  --output_dir  $RUN_ROOT/initial_s_B \
  --agents_template prompts/translation_agent.AGENTS.md

# 6. Snapshot -> stageB
cp -r $RUN_ROOT/initial_s_B $RUN_ROOT/stageB

# 7. Run translation agent
python ../src/codex_expts/run_codex.py \
  --input_path $RUN_ROOT/stageB --model $MODEL --budget_usd $BUDGET --num_workers 16
```

Calling `run_codex.py` directly (rather than `src/codex_expts/run.sh`)
lets the orchestrator pass a per-run `--budget_usd` without editing the
shim script, which hardcodes `--budget_usd 10`.

**Failure behavior.** `run_codex.py` exits 0 even when individual
benchmarks fail (per-benchmark failures are captured in
`codex_run_summary.json`, not in the process exit code). That's
acceptable — stage B proceeds with whatever stage A produced; failures
are observable in the summary files. If the orchestrator itself dies
mid-pipeline (e.g., network or disk error), user re-runs — a new
timestamp prevents collision.

## What this design catches vs. the pipeline in improvements.md

| Failure class                                    | Caught by                                             |
| ------------------------------------------------ | ----------------------------------------------------- |
| `proj_42` required/covered contradiction         | Agent gate §4.1 (coherence check in self-check)      |
| `aes128_SIMD` synthesised-ref vs. fixture bug    | Agent gate §4.2 + §4.3a (C-oracle fixtures)           |
| AES row-major layout over-specification          | Agent rule §4.5 (API-boundary assertions)             |
| Model drift on "which symbols are required"      | Agent rule §4.4 (INTERFACE_FNS anchored to interfaces) |

FFI oracle (§4.3b) remains out of scope. The per-benchmark `bindgen`/`cc`
variance that makes it a week of work in the pipeline version also
makes it flaky under agent control.

## Operational notes

- **Cost.** Each benchmark now runs codex twice instead of once.
  Default `$10` budget per stage → worst case `$20` per benchmark,
  matching the order of magnitude of existing `gpt-5.4_*_agent` runs
  in `results/generated_datasets/`.
- **Cache locality.** `cargo clean` on `stageA/` before producing
  `initial_s_B/` means the translation agent in `stageB/` recompiles
  from scratch. Acceptable for small benchmarks; worth revisiting if
  large crates (e.g., libvcd-scale) dominate wall-clock.
- **Audit.** `initial/` and `initial_s_B/` are preserved so any
  reviewer can diff against `stageA/` / `stageB/` to see exactly what
  each agent changed.

## Rollout

Land as one PR. No migration step — existing `generate_guardrails.py`
output in `results/generated_datasets/` is unaffected. New runs go
under `*_agent/` directories and coexist with old runs.
