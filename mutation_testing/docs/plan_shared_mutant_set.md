# Plan: one mutant set, four test suites

Status: all three phases implemented and wired 2026-08-27/28. Written 2026-08-27.

```bash
python run_mutation.py ... --split_crates --shared_mutants
```

**Caveat when sharding.** `--shared_mutants` shards by stride over `cargo mutants --list`
order; cargo-mutants shards by hashing the mutant name. A *sharded* project therefore runs
a different subset under the two paths, so their numbers are not comparable
mutant-for-mutant and neither are their wall times (measured end-to-end on `aes128_SIMD`
shard 1/12: shared 85 s over 30 viable mutants, split-only 76 s over 19 — shared is ~1.4×
faster per unit of work while looking slower in total). With `--big_threshold 0` both run
the full set and the question does not arise. To compare against an existing run, use
`mutate_shared.py --only-names-from <that run's outcomes.json>`.

Implementation:

- Phase 1 — `intersection_scores.py`, a standalone read-only pass over any results tree
  (`python intersection_scores.py results_gpt55_4settings results_opus5_4settings`).
  Reads the `outcomes.json` files already on disk in either layout (`_mutants_out/` or
  the older in-project `mutants.out/`) and writes `intersection_table.{md,csv}`,
  `intersection_long.csv` and `intersection_report.json` per tree.
- Phase 2 — `split_crates.py` (workspace assembly) plus `run_mutation.py --split_crates`.
  Opt-in: without the flag the pipeline is byte-for-byte the old one.
- Phase 3 — `mutate_shared.py` plus `run_mutation.py --split_crates --shared_mutants`.
  Applies each mutant once and runs all four suites against that one build. Writes
  cargo-mutants-shaped `outcomes.json`, so phase 1 reads its output unchanged, and the
  same per-setting `metadata/<setting_dir>/mutation.json`, so `--resume` keeps working
  (at project granularity: a project with any setting outstanding re-runs those settings).
  Phase 4 then runs one job per *project* instead of one per (project, setting), which
  also retires the per-setting target-dir partitioning phase 2 needed — no two settings
  share a target dir when there is only one run.

What Phase 1 says about the *stored* results (why phase 2 was needed, and why a re-run
is needed on top of it):

| tree | projects | intersection ≥10 | intersection 1–9 | intersection 0 | denominators already agreed |
| --- | --- | --- | --- | --- | --- |
| gpt55 | 93 | 22 | 50 | 14 | 6 |
| opus5 | 87 | 23 | 46 | 16 | 1 |

The mutant *sets* do match (84/93 and 67/87 identical key sets), so the key is stable and
the intersection is well-defined — but unviability differs so widely between settings
that most projects have almost no shared denominator left to score over. Phase 1 makes
the comparison honest; it cannot recover mutants the stored runs threw away. See "Stale
results" under Open questions.

## Problem

The study compares four test-generation modalities (`src`, `src+test`, `src/test`,
`c+rust`) by mutation score. Today each setting is a full copy of the project with
that setting's tests in `src/bin/*.rs`, i.e. **inside the same crate as the mutated
source**. cargo-mutants is invoked once per (project, setting).

Two consequences:

1. **The denominator moves between settings.** cargo-mutants labels a mutant
   `unviable` when the crate fails to compile — and the test code is part of that
   crate. So a setting whose tests break compilation loses mutants from its
   denominator. Measured on the gpt55 tree: only **4 of 87** projects have the same
   viable count across all four settings.

   ```
   CircularBuffer   total=86   viable: src 16  src+test 7   src/test 32  c+rust 37
   Holdem_Odds      total=145  viable: src 68  src+test 12  src/test 48  c+rust 93
   FastHamming      total=117  viable: src 49  src+test 8   src/test 10  c+rust 20
   ```

   `killed/viable` therefore compares scores computed over different mutant
   populations. `src+test` looks worst partly because its tests compile-break the
   most mutants, which is a confound, not a finding.

2. **Every mutant is compiled four times**, once per setting, even though the
   mutated source is byte-identical across settings.

The mutant *set* is already effectively shared: totals match across all four
settings in **85 of 87** projects (the two exceptions are `Linear_Algebra_C`, whose
`src` run produced 0 mutants after a failed baseline, and `Math_Library_in_C`, where
one setting ran a different shard). But that is a coincidence of the pipeline, not a
guarantee.

## Goal

Make the four settings score the *same* mutant set over the *same* denominator, and
stop rebuilding each mutant four times.

## Approach

Three pieces, independently landable, in increasing cost order. Phase 1 is the one
that unblocks the paper; phases 2–3 are the structural fix.

---

### Phase 1 — Intersection scoring (no restructuring)

Report each project's score over the mutants viable in **all four** settings, and
report the intersection size alongside it.

- Identify mutants across settings by the key cargo-mutants already gives us in
  `mutants.out/outcomes.json`: `(scenario.Mutant.file, function.function_name,
  replacement, line/col if present)`. Verify the key is stable across settings on
  one project before relying on it.
- Recompute per-setting `killed / |intersection|` and add an `n_common` column to
  `mutation_table.{md,csv}` and `gpt55_results_long.csv`.
- Keep the existing per-setting `viable` numbers in the long CSV — the divergence is
  itself a finding worth reporting (tests that break compilation).
- Exclude projects whose intersection is below some floor (say 10 mutants) from the
  averaged row rather than silently averaging noise.

Deliverable: the comparison table stops carrying incomparable denominators.

Files: the report/aggregation code in `run_mutation.py` that writes
`mutation_table.md` / `mutation_report.json`.

---

### Phase 2 — Split source and tests into separate crates

Per project, replace the four full copies with one small workspace:

```
<project>/
  Cargo.toml            [workspace] members = ["core", "tests-src", "tests-src_and_test",
                                               "tests-src_over_test", "tests-c_and_rust"]
  core/                 the frozen translation — the ONLY crate that gets mutated
  tests-src/            setting's tests, depends on core
  tests-src_and_test/
  tests-src_over_test/
  tests-c_and_rust/
```

Then invoke, from the project root, once per setting:

```bash
cargo mutants -d <project> --package core --test-package tests-src \
  --timeout-multiplier 2.0 --output <OUTSIDE_THE_TREE>/out_src
```

- `--package core` scopes mutant *generation* to the source crate → the mutant list
  is identical across the four invocations by construction.
- `--test-package <X>` selects which suite runs. Both flags exist in cargo-mutants
  27.1.0 (verified via `--help`).
- Viability becomes a property of `core` alone, so the denominator is shared.

**This is safe for these projects**: the generated tests only use the public API
(`use libutf::libutf_utf::*;` etc.), never crate internals, so moving them out of the
crate changes nothing semantically. Re-verify this holds for all projects before
migrating — grep the test bins for anything reaching past the public surface.

**The trap in this design, and the fix.** `--test-package` selects which suite runs for
each *mutant*, but cargo-mutants 27.1.0 baselines the package it **mutates**. Read off a
split run's `outcomes.json`:

```
Baseline   Build/Test  cargo test [--no-run] --verbose --package=aes128_SIMD@0.1.0   0.20s / 0.05s
mutant N   Build/Test  cargo test [--no-run] --verbose --package=tests-src@0.1.0
```

`core` has no tests of its own, so that baseline is vacuous, and splitting the crates
silently removed the guard the legacy layout got for free (the suite lived in the mutated
crate, so cargo-mutants' baseline covered it). Two consequences, both measured on
`aes128_SIMD` shard 1/12 by appending one `assert_eq!(1, 2)` to a `tests-src` test:

```
before:  ok Unmutated baseline in 0s build + 0s test
         34 mutants tested: 19 caught, 15 unviable       <- 19/19 = 100%
after:   aes128_SIMD/tests-src: unmutated suite is red, skipping mutation
         baseline_valid=false, reason=test_package_baseline_failed
```

A red suite fails on every mutant, so every mutant reads as caught: the score goes to
**100%** instead of the true 89% (17/19), and it fails in the flattering direction. The
second consequence is the timeout: cargo-mutants derives the per-mutant cap from that
0.05s no-op, so `--timeout-multiplier 2.0` collapses to the 20s `minimum-test-timeout`
floor regardless of how slow the suite actually is — and a `Timeout` sits in the
denominator without counting as a kill.

Fix (`_test_package_baseline` in `run_mutation.py`, split path only):

1. Before cargo-mutants, run the test crate unmutated — `cargo test --package tests-<X>
   --no-run` then the real run — and abort that setting with
   `reason: "test_package_baseline_failed"` rather than writing a bogus score. This also
   catches suites that pass test-by-test but fail together, which
   `ensure_green_baseline` cannot see: it runs one `--test <name>` at a time, whereas
   cargo-mutants runs the whole crate.
2. Pass `--baseline skip`, since cargo-mutants' own baseline is now redundant. That in
   turn makes an absolute `--timeout` safe (it otherwise caps the baseline run too), so
   the per-mutant cap becomes `max(20, ceil(measured_suite_time * multiplier))`. The
   measured time is recorded as `baseline_test_s` in `mutation.json`.

Re-verified after the fix: all four settings still `total 34, viable 19, unviable 15,
killed 17`, with `baseline_test_s` 0.14–0.88 s, and no `Baseline` scenario in
`outcomes.json`.

Notes:
- A small per-project workspace is fine and does **not** reintroduce the problem
  fixed by `split_manifests.py`: cargo-mutants copies the enclosing workspace, which
  here is one project (~3.4 MB), not all 93 (~22 MB). Keep dependencies pruned
  per-crate as `split_manifests.py` does.
- The green-baseline phase (repair + prune) must run per test crate, since a test
  that fails to compile now breaks its own crate rather than marking mutants
  unviable.
- `_strip_dangling_targets` still applies, now to the test crates.

---

### Phase 3 — Build each mutant once

**Implemented** as `mutate_shared.py`. Not "cost only" after all: it closes a real
correctness gap left by phase 2, described below.

```bash
python mutate_shared.py <workspace> --package <core> \
  --output <RESULTS>/_mutants_out --target-dir <TD> --shard 1/12
# or reproduce an existing run mutant-for-mutant:
python mutate_shared.py <workspace> --package <core> --output <OUT> \
  --only-names-from <RESULTS>/_mutants_out/src/<project>/mutants.out/outcomes.json
```

Mutants are applied by **splicing** `replacement` over the byte range in the `span` that
`cargo mutants --list --json` already reports, not by patching — cargo-mutants' emitted
`diff` has a pseudo-header (`+++ replace pad_buffer with ()`) no patch tool accepts.
Verified on all 398 `aes128_SIMD` mutants that the spliced file reproduces every added
line of the corresponding diff.

**Viability is now shared by construction, and phase 2's version of it was wrong.** The
naive loop (`cargo build -p core` decides viability, then test each suite) reports **11**
unviable of 34 on `aes128_SIMD` where cargo-mutants reports 15. The 4-mutant gap is
mutants that compile as a library and only fail once a dependent crate const-evaluates
the mutated expression — `*` becoming `/` inside an array length, for instance. Under
phase 2 each setting rediscovered those failures on its own, which is precisely the
per-setting-denominator disease this plan exists to cure; it only looked fine because all
four settings happened to break identically. So the viability probe is one batched
invocation spanning the mutated crate *and every live test crate*:

```
cargo test --no-run --package core --package tests-src --package tests-... (once per mutant)
cargo test --package tests-<X>                                            (per setting, just runs)
```

One failure retires the mutant for all four settings. That restores agreement with
cargo-mutants exactly (15 unviable / 19 viable) and makes the shared denominator a
property of the harness rather than a coincidence. `test_crate_broke` in the returned
`shared` dict counts any mutant that still breaks a suite after a green batched build; it
should stay 0, and is reported rather than folded into `caught` so a phantom kill can't
hide there.

Editing in place does **not** reintroduce the phase-2 stale-artifact bug that forced
per-setting target dirs. That bug came from cargo-mutants copying the workspace to a new
path per run, so a copy's mtimes could land behind a cached artifact's. Here the path is
fixed and a write moves the mtime forward, so cargo's fingerprinting holds: verified by
mutate → test → restore → test, twice over, every suite failing while mutated and passing
while clean.

The baseline gate from phase 2 is carried over and is now the *only* guard — there is no
cargo-mutants baseline at all — so a red suite is excluded from the run rather than
scoring 100%.

Cost: less than the 3–4× hoped for, because on these projects cargo's per-invocation
freshness check, not the `core` build, is the dominant per-mutant cost, and phase 3 still
runs one `cargo test` per setting.

| project | settings | mutants | phase 2 | phase 3 | speedup |
| --- | --- | --- | --- | --- | --- |
| `aes128_SIMD` (small core) | 4 | 34 | 75 s | 50 s | 1.5× |
| `kairoCompiler` (largest core in the tree) | 2 | 14 | 28 s | 14 s | 2.1× |

---

## Context needed to implement

- `split_manifests.py` — already landed. Makes each project a standalone workspace
  root with per-package pruned dependencies. `make_standalone(dir)` prunes an
  already-translated tree; `make_independent_packages(src, dst)` builds the pristine
  non-workspace copy (used for `datasets/RBench_independent_packages`). Originals
  live beside each manifest as `Cargo.toml.workspace.bak` and are the source of
  truth on re-runs — do not delete them. `--restore` undoes everything.
- `run_mutation.py` — phases: 1 translate, 2 test-gen, 3 green baseline, 4 parallel
  mutation. Calls `make_standalone` per setting dir after test-gen, and
  `rewrite_manifest` per project inside `run_translation`.
- Sharding: `--big_threshold` (default 300) / `--big_shard_target` (default 150). All
  stored results predating 2026-08-27 were sharded to ~150 mutants for any project
  over 300, so their totals are **shard sizes, not real totals**. True counts are much
  larger (opus tree: `Genetic` 1715, `libutf` 1061, `approxidate` 937, `jccc` 905,
  `Linear_Algebra_C` 878). `--big_threshold 0` disables sharding.
- cargo-mutants writes its output directory by clearing it on start, and copies the
  enclosing workspace per run. `--output` must be an **absolute** path **outside** the
  project tree — both mistakes have destroyed results in this project before.

## Open questions

- **Timeouts.** `mutation.json` records `timeout` separately from `killed`, and the
  reported score is `killed / viable`, so a timeout sits in the denominator without
  counting as a kill. Semantically a timeout *is* a detection. Every run so far
  reported `timeout: 0`, but unsharded runs test more loop mutations, so decide
  before the writeup whether to fold timeouts into `killed`.
  Still open, but now a one-flag decision: `intersection_scores.py --fold-timeouts`
  scores them as kills, and the long CSV carries the timeout column either way.
- **Target-dir sharing between settings.** Found while verifying phase 2, and possibly
  relevant to the stale-results question below. Running two settings of the same project
  through one `CARGO_TARGET_DIR` makes cargo reuse the previous setting's artifacts: test
  files fail spuriously in the baseline (and are then *deleted* by the prune step), and
  every mutant survives. On `aes128_SIMD` shard 1/12, a shared target gave 17/19 for the
  first setting and 0/23 for the next three; one target dir per setting gives 17/19 for
  all four. Phase 2 therefore partitions the target-dir pool per setting.

  The legacy layout draws from one shared pool too, and a worker can pick up two settings
  of the same project in a row. It is less exposed — the four settings sit at four
  different paths there, rather than being four `--test-package` views of one workspace —
  but this has not been ruled out as a contributor to the inflated-unviable runs, and it
  is worth checking before concluding the stored numbers are only stale for other reasons.

- **Flaky generated tests.** Found while checking why `kairoCompiler` ran only three
  settings in a phase-3 smoke run: `tests-src_over_test/tests/parser_tests.rs:32`
  (`test_parse_builds_nodes_from_tokens`, "a node on the stack") fails about two runs in
  three, so the baseline gate excluded that setting on one run and passed it minutes
  later. This is a problem for the study independent of which phase runs the mutation:
  `ensure_green_baseline` prunes by running each test file **once**, so a flaky test can
  survive the prune and then fail against mutants, scoring phantom kills that inflate that
  setting. Worth measuring how widespread it is — repeat the green baseline N times and
  prune anything that is not consistently green — before trusting per-setting differences
  of a few points.

- **Stale results.** The A/B on `CircularBuffer` produced 86 viable / 62 caught,
  while the stored dataset says 16 viable / 12 killed / 70 unviable — the same
  inflated-unviable signature already confirmed on the five re-run projects. Assume
  the full stored benchmark needs re-running.
- `docs/mutant_distribution_and_memory_types.md` still carries the inflated "62%
  unviable" headline computed from those buggy runs; rewrite after clean numbers.

## Verification

- Phase 1: on one project, dump the mutant keys per setting and assert the
  intersection equals each setting's set where totals already match (85/87 projects).
  **Done** — `intersection_scores.py` computes this per project and reports it as
  `sets_identical` / `denominators_agree` in `intersection_report.json`; the counts
  are in the table above.
- Phase 2: on one project, run all four invocations and assert the four
  `outcomes.json` files list an identical mutant set and identical viable count.
  Then assert the per-setting killed counts still reproduce the pre-migration
  values where the denominators previously agreed.
  **Done** on `aes128_SIMD` (opus5 tree, full 398-mutant list, shards 1/8 and 3/8):

  ```
  cargo mutants --list --package aes128_SIMD --test-package <X>  # byte-identical for all four X
  shard 1/8   every setting: total 50, viable 34, unviable 16   (identical mutant AND viable sets)
  shard 3/8   every setting: total 50, viable 50, killed 46
  shard 3/8   legacy layout, src setting: total 50, viable 50, killed 46  <- reproduces pre-migration
  shard 1/12  every setting: total 34, viable 19, unviable 15, killed 17  (through run_mutation.py)
  ```

  Re-run after the vacuous-baseline fix ("The trap in this design" above): unchanged, all
  four settings total 34, viable 19, unviable 15, killed 17, and a deliberately red suite
  is now rejected instead of scoring 100%.

  The last line is the wired path (`build_workspace` -> `ensure_green_baseline(suite=...)`
  -> `mutate_with_cargo(package=, test_package=)`), and it is also the case that exposed
  the target-dir bug below: with a shared target dir the same run reported 17/19, then
  0/23, 0/23, 0/23.

  Timing on the same shard: 30 s split vs 30 s legacy for `src` — the per-project
  workspace does not cost anything measurable, because cargo-mutants' per-run copy is
  still one project (core + four crates of test sources), not the 93-project umbrella
  that `split_manifests.py` removed.
- Phase 3: every mutant's verdict must match the phase-2 cargo-mutants run on the
  identical mutant set. **Done** — 116 mutant verdicts across 6 (project, setting) pairs,
  all identical:

  ```
  kairoCompiler  src            n=14  per-mutant identical
  kairoCompiler  src+test       n=14  per-mutant identical
  aes128_SIMD    src            n=34  per-mutant identical
  aes128_SIMD    src+test       n=34  per-mutant identical
  aes128_SIMD    src_over_test  n=34  per-mutant identical
  aes128_SIMD    c_and_rust     n=34  per-mutant identical
  ```

  The written `outcomes.json` is a drop-in for cargo-mutants': running
  `intersection_scores.py` over a phase-3 tree gives `n_common 19`, `19/19/19/19`
  viable, `89% (17/19)` for all four settings.
- Both: `cargo check --all-targets` across the tree must stay green — currently
  87/87 on the opus tree, 92/93 on gpt55 (`Megalania` fails on a pre-existing
  dangling `[[test]]` entry, unrelated, handled by `_strip_dangling_targets`).
