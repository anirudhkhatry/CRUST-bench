# Mutation-testing study — index

Comparing **test-generation modalities** for C→Rust translation by **mutation score**
(cargo-mutants), across two models (**gpt-5.5**, **claude-opus-5**). All settings mutate
the *same* frozen Rust source per project, so score differences reflect the
test-generation modality alone.

## The four settings (test-generation inputs)

| setting | inputs to test generation |
|---|---|
| `src` | C source + Rust interfaces |
| `src+test` | C source + **C tests** + Rust source |
| `src/test` | Rust source + Rust interfaces |
| `c+rust` | C source + Rust source |

(All share the same transpiled "frozen" Rust source, which is what gets mutated.)

## Headline results (mutation score, fraction of viable mutants killed)

| setting | gpt-5.5 (n=85) | Opus-5 (n=69) |
|---|---|---|
| `src/test` | 90.3% | 95.3% |
| `c+rust` | 89.7% | 95.7% |
| `src` | 88.5% | 91.9% |
| **`src+test`** | **77.4%** | **84.4%** |

Two robust findings (both models): **(1)** generating tests *against the Rust
implementation* (`c+rust`, `src/test`) beats interfaces-only (`src`); **(2)** handing the
model the **C tests to translate** (`src+test`) is clearly worst.

## Analyses

- [`why_c_tests_hurt_mutation.md`](./why_c_tests_hurt_mutation.md) — why `src+test` (translated
  C tests) underperforms. Differential analysis (547 mutants it missed that `c+rust`
  killed, both models), two mechanisms (weak *relational* oracles vs. narrow *coverage*
  inherited from the C suite), and five worked examples with code excerpts
  (murmurhash_c, proj_2DPartInt, approxidate, cJSON, emlang, clog).

- [`mutant_distribution_and_memory_types.md`](./mutant_distribution_and_memory_types.md) —
  the mutant distribution and memory-type view. **~62% of mutants are unviable** (Rust's
  type/borrow system rejects them); of viable, ~85% killed. Per return-type breakdown:
  **custom ADTs are the most filtered** (80% unviable), **`Option` is the biggest blind
  spot** (68% killed). Proposes Rust-type-aware mutation operators and a
  surviving-mutant → test-generation loop for correctness.

- [`mutation_operator_taxonomy.md`](./mutation_operator_taxonomy.md) — the operator
  taxonomy. Three orthogonal axes (edit shape, fault class, status); cargo-mutants'
  five language-generic genres **and what they showed** (61.7% unviable corpus-wide,
  `Option` the one type-level blind spot, no class-level blind spots at all, and bitwise
  substitutions at 100% in exactly the two settings that see C source without the C
  tests); and the eleven **transpilation-aware** operators written
  for this study, in three fault families (F1 integer discipline, F2 representation,
  F3 ownership), with the design rulings behind each and the arms deliberately left out.
  Also the two results that decide how the numbers read: `cast-width`/`cast-sign` are
  **100% unviable** (Rust deletes that fault class for free), and the bare-operator →
  `wrapping_*` direction of F1 is **equivalent, not missed** (113 mutants at a uniform
  0.0% across four suites). Closes with the state of the full-set run.

- [`plan_shared_mutant_set.md`](./plan_shared_mutant_set.md) — the denominator problem and
  its fix. The scores above are `killed / viable` where `viable` is measured **per
  setting**, and the tests live in the mutated crate, so a setting whose tests break
  compilation loses mutants from its own denominator (`src+test` loses the most, which
  flatters nothing but confounds the comparison). Three landed fixes:
  `analysis/intersection_scores.py` rescores any existing results tree over the mutants viable in
  *all four* settings; `harness/run_mutation.py --split_crates` moves the tests out of the mutated
  crate; and `harness/mutate_shared.py` applies each mutant once and runs all four suites against
  that single build (`harness/run_mutation.py --split_crates --shared_mutants`), which is what
  actually makes the denominator shared *by construction*
  — the split alone leaves each setting rediscovering unviability for itself, and mutants
  that compile as a library but fail a dependent crate's const-eval slip through. Validated
  mutant-for-mutant against cargo-mutants (116 verdicts, all identical), 1.5–2.1× faster.

## Data files (CSVs)

Per-model result tables live in the run output dirs (under `results/<model>/<run>/`);
cross-model files in `mutation_testing/data/`. The code layout and the commands that
produce all of this are in [`../README.md`](../README.md).

| file | contents |
|---|---|
| `results/gpt/results_gpt55_4settings/gpt55_aggregate.csv` | gpt-5.5 per-setting avg score, avg tests kept, coverage, n |
| `results/gpt/results_gpt55_4settings/gpt55_scores_wide.csv` | gpt-5.5 per-project score, one column per setting |
| `results/gpt/results_gpt55_4settings/gpt55_results_long.csv` | gpt-5.5 per project × setting, all fields (killed/viable/mutants_total/sharded/tests/reason) |
| `results/gpt/results_gpt55_4settings/mutation_table.csv` | gpt-5.5 display table (score + killed/viable per cell) |
| `results/claude/results_opus5_4settings/opus5_aggregate.csv` | Opus-5 per-setting aggregate (same schema as gpt55_aggregate) |
| `data/gpt55_vs_opus5.csv` | model head-to-head over the 63 projects comparable in **both** models |
| `data/differential_by_project.csv` | per project × model: src+test killed/missed, and mutants src+test missed but `c+rust` / `src/test` killed |
| `<results_dir>/intersection_table.csv` | per project: intersection size + each setting's score over it (`analysis/intersection_scores.py`) |
| `<results_dir>/intersection_long.csv` | per project × setting: intersection score *and* the setting's own killed/viable/unviable counts |

Column notes:
- **mutation score** = `killed / viable`, where `viable = caught + missed` (unviable and
  timeout mutants excluded from the denominator).
- **comparable** = project produced a valid, mutant-bearing baseline in *all four*
  settings (only these enter the per-setting averages).
- **tests_kept** = number of generated test *files* surviving the green-baseline prune
  (not `#[test]` functions; not a quality signal — see the src+test analysis).

## Reproducing

Raw per-mutant data is under each run at
`<results_dir>/<setting_dir>/<project>/mutants.out/{caught,missed,unviable,timeout}.txt`
and the generated tests at `<results_dir>/<setting_dir>/<project>/src/bin/`.
`<setting_dir>` = `src`, `src+test`, `src_over_test` (`src/test`), `c_and_rust`
(`c+rust`). The full pipeline is `mutation_testing/harness/run_mutation.py`; see
[`../README.md`](../README.md) for the exact invocations.
