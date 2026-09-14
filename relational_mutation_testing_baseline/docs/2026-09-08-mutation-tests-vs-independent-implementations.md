# Can the mutation tests tell a wrong implementation from a right one?

**Date:** 2026-09-08 · **Setting:** `mutation_tests` T′ suites replayed against the
`entry_points_gen_unit_test_decoupled` implementations · **Projects:** 97

**Short version:** on implementations they were not written for, the mutation tests are
weak *behavioural* detectors and strong *conformance* detectors. Of the 13 implementations
the ground truth rejects behaviourally and where at least one mutation test compiled, T′
catches 3 — recall 0.23. On the cleanest subset, the 36 projects where every mutation test
compiled, T′ catches **0 of 6**. But 305 of the 853 tests (36%) do not compile against an
independent implementation at all, and 99 of the 101 names they cannot import are declared
in the RBench interfaces — so 24 implementations that the ground-truth suite passes are
silently missing contracted public API, and the mutation tests are the only thing in the
pipeline that notices.

## 1. What was measured

The `mutation_tests` scoreboard (853/853 T′ green) measures each suite against the one
implementation it was authored for, and the run's exit condition *is* a clean `cargo test`,
so it is close to tautological. This experiment breaks that coupling. It takes an
independent tree of implementations — `datasets/codex_expts/entry_points_gen_unit_test_decoupled/expt_codex_output`,
which the mutation-test pipeline never saw — and grades each of the 97 shared projects
twice against the *same* sources:

| suite | staging | role |
|---|---|---|
| GT | RBench `src/bin` pasted in, `tests/` removed | the label: a failing suite means the implementation is wrong |
| T′ | `rmt_*.rs` pasted into `tests/`, `src/bin` removed | the prediction under test |

The tree was `cargo clean`ed (4.3 GiB of build artefacts removed) and copied to
`datasets/codex_expts/mutation_tests/entrypoint_impls` before any of this ran; the harness
itself works on throwaway copies, so neither tree is mutated.

Two controls matter:

* **The library is built on its own first.** An implementation that does not compile at all
  would be rejected by both suites for a reason neither suite deserves credit for. All
  97 compile as libraries, so nothing is excluded on this ground.
* **T′ is built and run one integration-test target at a time.** Each `rmt_*.rs` is its own
  binary. Building the crate in one shot lets a single test that fails to compile suppress
  the verdict of every other test in the project — which is exactly what happened on the
  first pass, and it cost 35 projects their entire signal.

Raw status distribution over the 97 projects:

| | green | test_failed | build_error | timeout | no_tests |
|---|---|---|---|---|---|
| GT | 65 | 17 | 12 | 2 | 1 |
| T′ | 36 | 11 | 50 | 0 | 0 |

Per individual mutation test, over all 853: **519 green, 25 failed, 4 timed out,
305 did not compile.** On average 64% of a project's tests compile against an independent
implementation; in 18 projects *none* of them do.

## 2. The confusion matrices

A `reject` from either suite conflates two different events, and the raw matrix is
misleading until they are separated: a **behavioural** rejection (the suite compiled, an
assertion failed) is the detection event the experiment is about; a **structural** rejection
(the suite would not compile) is a different claim entirely. So, three matrices.

**Raw — any rejection counts (n=97).**

| GT \ T′ | reject | accept |
|---|---|---|
| reject | 23 | 8 |
| accept | 38 | 27 |

recall 0.74 · precision 0.38 · accuracy 0.52. This flatters T′ badly: most of the 23 and
almost all of the 38 are compile failures.

**Behavioural — the 36 projects where every mutation test compiled and ran.**

| GT \ T′ | reject | accept |
|---|---|---|
| reject | 0 | 6 |
| accept | 3 | 27 |

**recall 0.00.** The six implementations GT rejects here —
`Genetic_neural_network_for_simple_control`, `dict`, `inversion_list`, `kd3`, `libvcd`,
`libwecan` — each fail exactly one ground-truth assertion, and all 50 mutation tests across
those six projects pass. These are the same projects that appeared as `COVERED` failures in
the [previous write-up](2026-09-08-gt-failures-and-mutant-improvements.md): the tests
exercise the right function and demand too little of it. That analysis said the mutation
tests were satisfiable by a wrong implementation; this is the direct confirmation, on
implementations chosen without reference to them.

**Behavioural with partial credit — 68 projects, counting only the tests that compiled.**

| GT \ T′ | reject | accept |
|---|---|---|
| reject | 3 | 10 |
| accept | 5 | 50 |

recall 0.23 · precision 0.38 (but see §4 — the five "false alarms" mostly are not).

Caught: `aes128_SIMD` (2 of 9 tests fire), `emlang` (1 of 10), `fslib` (2 of 9).
Missed: `Genetic_…`, `clog`, `dict`, `inversion_list`, `kd3`, `lib2bit`, `libutf`,
`libvcd`, `libwecan`, `razz_simulation`.

## 3. The structural result, which is the larger one

305 of 853 tests fail to compile, and the reason is nearly always an unresolved import.
The obvious hypothesis is that the transpiled tests over-reached — that they use helpers
the implementation was never required to provide. Checking each unresolved name against the
project's RBench `src/interfaces/` says the opposite:

**99 of the 101 unimportable names are declared in the interfaces.** Only two are not
(`blt::cbt`, `quadtree::quadtree` — both module paths, not items).

So these implementations omit public API the contract requires, and **24 projects do this
while passing the entire ground-truth suite**: `Holdem_Odds` (`card_equal`,
`CARDS_PER_DECK`), `LTRE` (`dfa_free`, `nfa_free`, `ltre_fixed_string`), `Linear_Algebra_C`
(`add_vectors`, `adjugate_matrix`, `sign_matrix`, …), `Megalania`, `VaultSync`, `bhshell`,
`btree_map`, `c_aces` (six of its types), `carrays`, `chtrie`, `fs_c`, `hydra` (seven
functions), `jccc`, `libm17`, `ljmm`, `merkle_tree_c` (the whole blake2b surface), `morton`,
`proj_2DPartInt`, `recordManager`, `rubiksolver`, `satc`, `simple_lang`, `utf8`, `vec`.

That is a gap in RBench's oracle, not in the mutation tests: `src/bin` never calls those
items, so an implementation can drop them and score green. The mutation tests find it only
as a side effect — they fail to compile rather than reporting it — but they are the only
component of the pipeline that touches that part of the surface at all.

## 4. The five "false alarms" mostly are not

Five GT-green projects have at least one compiling mutation test that fails. Reading the
assertion messages, four of the five look like real defects the ground truth misses:

| project | T′ fails | what the assertion says |
|---|---|---|
| `amp` | 6 of 9 | serialisation emits `AMPSTATE:00000012:0000…68656c6c6f…` where the wire format is the raw bytes `\0\0\0\x05hello…` |
| `avalanche` | 3 of 10 | `matrix_fprintf` output is `"1.02.03.0 4.05.06.0 "`; a counter reads 130 where 34 is expected |
| `blt` | 4 of 7 | leaves carry no data — `ceil("b")->data` is null where 3 is expected |
| `mvptree` | 4 of 7 | nearest-neighbour retrieval returns `"B"` where `"C"` is expected |
| `proj_42_Kocaeli_Printf` | 1 of 5 | 0 bytes reach stdout where 10 must |

These are judgements from the failure text, not from re-deriving each spec, so treat the
precision figure of 0.38 as a floor rather than an estimate. `mvptree` is the one to check
first: its own GT failure in the earlier analysis was a decimal-vs-octal mode confusion, so
it has a history of the oracle and the test disagreeing about the spec rather than about the
code.

## 5. What this changes

1. **The headline claim has to be qualified.** "853/853 mutation tests pass" says nothing
   about detection power. The honest number is recall 0.23 with partial credit, 0.00 on the
   clean subset. Both belong in the paper next to the 853.
2. **Portability is a first-class defect of the transpiled tests.** 36% non-compilation is
   not a footnote — a test that cannot be aimed at a second implementation cannot be used as
   an oracle. The immediate fix is a stage-3.5 gate: compile every candidate test against a
   *second* implementation of the same interfaces, and reject or repair it if it does not
   build. That also converts the conformance finding from an accident into a measurement.
3. **The recall result is the same defect the previous write-up diagnosed**, now measured
   rather than inferred. The `outcome == "KILLED"` filter, the `UNTRANSLATABLE` leak, and the
   witness→family rules in
   [that document's §3](2026-09-08-gt-failures-and-mutant-improvements.md) are the changes
   that would move this number; this experiment is the metric to re-run afterwards.
4. **Report the conformance gap to RBench.** 24 implementations passing the ground truth
   while omitting contracted public API is a result about the benchmark, independent of
   anything in this pipeline.

## 6. Artifacts

| path | what |
|---|---|
| `src/codex_expts/mutant_vs_gt_matrix.py` | the harness: stages both suites over each implementation, per-target T′ runs |
| `src/codex_expts/summarise_mutant_vs_gt.py` | the three matrices, the structural breakdown, per-test totals |
| `datasets/codex_expts/mutation_tests/entrypoint_impls/` | the cleaned copy of the implementation tree (6.7 MB) |
| `datasets/codex_expts/mutation_tests/mutant_vs_gt_matrix.json` | per-project rows, including `per_test` status for all 853 tests |
| `datasets/codex_expts/mutation_tests/mutant_vs_gt_matrix.crate_build.json` | the first pass, built per crate rather than per target — kept only to show how much signal that loses |
| `/tmp/mx_full2.log` | the run log |
