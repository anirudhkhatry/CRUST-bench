# `mutation_testing/`

Mutation testing for C→Rust translation on CRUST-bench. The study question is which
**test-generation modality** produces the strongest test suite: every setting mutates the
*same* frozen Rust translation of a project, so a score difference is a property of the
tests, not of the translation.

Two mutant populations are used, over the same frozen source:

- **classical** — cargo-mutants' language-generic edits (operator substitution, return-value
  replacement);
- **transpilation-aware** — eleven operators written for this study that simulate faults a
  C→Rust translator actually commits (integer discipline, representation, ownership).

The analysis of both, the four settings, and the results so far are in
[`docs/README.md`](docs/README.md); start there for *what the numbers mean*. This file is
about *where the code is and how to run it*.

## Layout

```
mutation_testing/
├── harness/       importable library: the mutation engine
├── experiments/   campaign drivers (one per run); write into results/
├── analysis/      scorers and reports over a finished run
├── probes/        one-off measurements kept for provenance
├── data/          small committed artefacts: corpus manifests, cross-model CSVs
├── docs/          the write-ups (start at docs/README.md)
├── tests/         unit tests for the operators and the mutant lister
└── results/       run outputs, git-ignored, multi-gigabyte
```

Module names are unique across the tree, and the write-ups in `docs/` cite them bare
(`mutate_shared.py`, `run_transpile_full.py`). The tables below are the name → directory
map; nothing in `docs/` was rewritten for the move.

### `harness/` — the engine

| module | role |
|---|---|
| `run_mutation.py` | the end-to-end pipeline *and* a CLI: transpile → generate tests → repair → mutate. Also the source of `SETTING_DIRS`, the setting → directory map every driver imports. |
| `mutate_shared.py` | applies one mutant and runs **all four** suites against that single build. Makes viability — hence the denominator — shared by construction. Also owns `list_mutants`, `baseline`, `splice`. |
| `mutate_fast.py` | the parallel driver over `mutate_shared`: N worker clones of the workspace, per-mutant timeouts, `outcomes.json` in cargo-mutants' own shape. |
| `split_crates.py` | rebuilds a project as a workspace: `core/` (the frozen translation, the only crate mutated) + `tests-<setting>/` per suite. |
| `split_manifests.py` | manifest surgery for that split — de-workspacing, dependency version resolution. |
| `transpilation_operators.py` | the eleven transpilation-aware operators (tree-sitter based), in three fault families. |
| `list_transpile_mutants.py` | those operators packaged as a drop-in replacement for `mutate_shared.list_mutants`, which is what lets one harness run either mutant population. |
| `paths.py` | `ROOT` / `RESULTS` / `DATA`. The one definition of the three directories. |

### `experiments/` — drivers

| script | what it does |
|---|---|
| `audit_full_set.py` | cheap precondition check over a results tree: frozen source present, per-setting tests present, prior baseline green. Commits no compute. |
| `build_full_set_workspaces.py` | rebuilds the audited projects as split-crate workspaces and probes every baseline. Writes `data/full_set_readiness.json` and `data/full_set_corpus.json`. |
| `run_transpile_full.py` | the transpilation-operator run over the whole four-green corpus (`results_opus5_4settings`). |
| `run_transpile_sample.py` | the same, over the 5-project sample (`opus5_sample`). |
| `run_opus5_sample.py` | the *classical* (cargo-mutants) run over that sample. |
| `ground_truth_mutation.py` | the score ceiling: RBench ground-truth tests instead of generated ones. **Caution:** it `unlink()`s every ground-truth test that fails to compile or pass. |

### `analysis/` — scoring a finished run

| script | what it does |
|---|---|
| `analyse_opus5_sample.py` | the main scorer. Per-setting caught/missed/timeout/unviable, the common-mutant set, and the per-mutation-type breakdown. Works on any run root. |
| `intersection_scores.py` | rescores an existing (non-split) results tree over the mutants viable in all four settings — the retrofit for runs that predate `--split_crates`. |
| `timing_breakdown.py` | where the wall-clock goes (build vs link vs test). |
| `llvmcov_check.py` | line-accurate coverage/oracle decomposition of the `src+test` → `c+rust` differential. |

### `probes/`

`timeout_probe.py` (how the timeout floor changes verdicts), `validate_fast.py`
(`mutate_fast` vs cargo-mutants, mutant for mutant), `rebuild_experiment.py` and
`link_experiment.py` (why a mutant costs ~16–19 s). Kept because the numbers in the docs
came from them.

## Environment

Everything must run under the `repair_rust` conda env — it is the only interpreter on this
box with tree-sitter 0.25, which `transpilation_operators.py` needs:

```bash
PY=/nas/miniconda3/envs/repair_rust/bin/python
```

The drivers import `run_mutation`, which pulls in the CRUST-bench translation pipeline,
whose OpenAI client is constructed at import time. No mutation run ever calls an endpoint,
so a placeholder key is enough (`run_transpile_full.py` sets one itself):

```bash
export OPENAI_API_KEY=unused-by-mutation-runs
```

Scripts run either way — as files or as modules — and work from any working directory:

```bash
$PY experiments/run_transpile_full.py --resume
$PY -m experiments.run_transpile_full --resume
```

## Commands

**The full pipeline** (translate → generate tests → repair → mutate) for one model, in the
shared-denominator layout. `--shared_mutants` requires `--split_crates`:

```bash
$PY harness/run_mutation.py \
    --benchmark_dir  ../datasets/CBench \
    --rust_dir       ../datasets/RBench \
    --output_dir     results/claude/my_run \
    --endpoint       claude --setting all \
    --split_crates --shared_mutants \
    --mutation_workers 4 --jobs 4 --min_test_timeout 120 \
    --resume
```

Add `--skip_translate --reuse_tests` to re-mutate an existing run without touching its
source or its tests.

**A transpilation-operator run over an existing results tree** — three phases, in order:

```bash
$PY experiments/audit_full_set.py --write          # -> data/full_set_projects.json
$PY experiments/build_full_set_workspaces.py --workers 6   # -> data/full_set_{readiness,corpus}.json
$PY experiments/run_transpile_full.py --resume     # -> <run>/_transpile_out, <run>/transpile_report.json
```

`--include-partial` adds the projects the split left with one or two red suites; their
mutants carry per-setting information but cannot enter the four-way comparison.

**Scoring a run** — first positional argument is the output directory inside the run,
second is the run root:

```bash
$PY analysis/analyse_opus5_sample.py                                    # classical, sample
$PY analysis/analyse_opus5_sample.py _transpile_out                     # transpile, sample
$PY analysis/analyse_opus5_sample.py _transpile_out results_opus5_4settings   # transpile, full set
$PY analysis/intersection_scores.py results/gpt/results_gpt55_4settings
```

**Tests** (69 of them; run from this directory):

```bash
$PY -m unittest discover -s tests -t . -q
```

## Conventions worth knowing before you run anything

- **Never `kill` a run.** Use `kill -INT`. SIGTERM skips the harness's `finally` restore and
  leaves a mutant spliced into `core/`, which then rides along on every later mutant of that
  project. The drivers check for this on entry (`verify_clean`) and restore, but only on the
  next start.
- **`--output` for cargo-mutants / `mutate_fast` must be an absolute path outside the
  project tree.** A relative one lands inside the crate being mutated.
- **Do not raise `WORKERS` × `JOBS` past 8 × 4.** Unbounded fan-out has crashed this box.
  The knobs are deliberately identical across runs so scores stay comparable — changing one
  invalidates comparison with everything already reported.
- **`MIN_TIMEOUT = 120`** means each genuinely hanging mutant costs two minutes. Wall-clock
  is driven by hangs, not by mutant count.
- **`results/` is git-ignored; `data/` is not.** Anything a doc cites belongs in `data/`.
- **Never `git add -A` here.** The repo root carries ~120 unrelated modified files and
  multi-gigabyte results trees; it times out. Add paths explicitly.
