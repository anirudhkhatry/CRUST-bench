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
# Produces these directories under
#   guardrail_guided_inference/results/generated_datasets/<MODEL>_<TS>_agent/
#     initial/                         stage-A input (guardrail AGENTS.md installed)
#     stageA/                          stage-A output (codex writes guardrails)
#     stageA_codex_output/             auto-flattened rust/ crates from stageA
#     initial_s_B/                     stage-B input (translation AGENTS.md installed, guardrail preserved)
#     stageB/                          stage-B output (codex translates)
#     stageB_codex_output/             auto-flattened rust/ crates from stageB
#     stageB_codex_output_gt_tests/    clone of stageB_codex_output with RBench
#                                      groundtruth tests overlaid in src/bin/

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
