# Phase 0 — feasibility results

**Date:** 2026-09-02, harness §2 revised 2026-09-03 · no LLM calls · scripts:
`phase0/panel.py`, `phase0/charness.py`, `phase0/recheck.py`

Phase 0 asked two go/no-go questions from the design spec: **is the soundness panel big
enough**, and **can we drive the C build harness**. Both pass.

## Headline

**The C harness drives 100/100 projects**, so eligibility is decided by the panel alone:
**69 projects at n ≥ 3**, 88 at n ≥ 2. The pilot needs 10.

| | n=0 | n=1 | n=2 | n≥3 | total |
|---|---|---|---|---|---|
| C harness usable | 5 | 7 | 19 | **69** | 100 |
| C harness unusable | 0 | 0 | 0 | 0 | 0 |

The 87/13 split this table carried until 2026-09-03 was two separate problems, both now
fixed: 11 projects the harness could not build (§2.3), and 2 lost to a name mismatch
between the CBench directory and the RBench project (`proj_2DPartInt`,
`proj_42_Kocaeli_Printf`). `charness.csv` is now keyed on the **RBench** name with the
resolved directory in a `cbench` column, so it joins `eligible.csv` unambiguously — the
earlier "89 here vs 87 there" was that join, not two build failures.

**Panel note.** Phase 1 does not use the agreement oracle this section sizes. The codex
transpilation is taken as the reference implementation wherever it is GT-green (§1.2, 94
projects), and the study asks whether a C-derived test discriminates it from its mutated
counterpart. The panel numbers stand as measured and remain the fallback if that
assumption has to be revisited.

## 1. The panel (`panel.json`, `panel_raw.csv`)

459 candidate implementations tested against the RBench ground-truth suite. The GT tests
are already present and **byte-identical** in the candidate trees, so the gate is just
`cargo test` in each.

### 1.1 Open question #2 from the spec is answered: `initial` ≡ `*_output`

The `initial` directories are run scaffolding (`<p>/c/` + `<p>/rust/`), not a separate
transpilation. Comparing implementation-file hashes:

| pair | identical |
|---|---|
| `codex_out` vs `codex_run` | **98/100** |
| `claude_out` vs `claude_run` | **100/100** |
| every cross-model pair | **0/100** |

So they collapse to one member per model, and the panel is built from **model families**,
not directories. Cross-model implementations are fully distinct — the independence the
agreement oracle needs is real.

### 1.2 GT-green rate per source

| source | tested | green | test_failed | build_error | no_tests | timeout |
|---|---|---|---|---|---|---|
| `codex` | 100 | **94** | 2 | 0 | 1 | 3 |
| `claude` | 100 | **81** | 4 | 11 | 1 | 3 |
| `gpt55` (4-settings frozen) | 99 | **70** | 16 | 12 | 1 | 0 |
| `opus5` (4-settings frozen) | 99 | **66** | 19 | 13 | 1 | 0 |
| `o4mini` | 5 | 1 | 0 | 4 | 0 | 0 |
| `gpt55_full` | 56 | 0 | 0 | 0 | **56** | 0 |

`gpt55_full` ships frozen *source* with no test bins, hence 0 tests run. It is the same
model family as `gpt55`, so recovering it would add no independence — **dropped**, not
repaired.

The 4-settings frozen sources are notably less GT-green (66–70%) than the dataset
candidates (81–94%). Expected: those trees were frozen for mutation, not selected for
correctness.

### 1.3 Panel size (GT-green, deduplicated by implementation hash)

```
n=0:  5 projects      n=3: 10 projects
n=1:  7 projects      n=4: 58 projects
n=2: 19 projects      n=5:  1 project
```

Distinct-model-family counts are **identical** to raw panel counts — no project's panel is
padded by two members of the same family. 69 projects reach n ≥ 3.

This is well above the spec's `n ≥ 3` target and comfortably resolves the risk flagged as
"panel too thin". The agreement oracle is viable as designed.

## 2. The C harness (`charness.csv`, `charness.json`)

**100/100** projects compile all library sources, link them against a generated test main,
and run it — which is exactly what Stages 1/2 need.

| harness | count | | project Makefile | count |
|---|---|---|---|---|
| ok | **100** | | ok | 71 |
| | | | failed | 16 |
| | | | no_makefile | 11 |
| | | | timeout | 2 |

`charness.py` drives the live harness (`rmt.cbench.Project.run`). It used to re-implement
the build, and that copy drifted: it reported 89 where the harness itself managed 90, by
compiling `main()`-bearing sources the harness excludes. The Makefile column is recorded
for information only (§2.1).

### 2.1 Design change: gcc-uniform, not Makefile-first

The spec said Makefile-first with a gcc fallback. Phase 0 shows that is the wrong order.
The Makefile path builds the *project's own* artifacts and gives no way to link a
generated test main against the library objects — a `make` success does not imply the
project is drivable. Every project marked usable above is usable **because of the gcc
path**, and the 71 Makefile successes add no project the gcc path does not already cover.

**Revised:** the uniform gcc harness is the primary and only build path; the Makefile is
retained only as an optional cross-check that our object set matches the project's own
expectations (e.g. missing `-D` flags). Recorded per project either way.

### 2.2 Two findings that change Stage 1

**Header libraries (11 projects).** `bigint`, `c_string`, `clog`, `cset`, `csyncmers`,
`libtinyfseq`, `rect_pack_h`, `merkle_tree_c`, `satc`, `Remimu` and `skp` have their
implementation in `*.h`, not `*.c`. The first sweep reported the first seven as "no
sources"; the last three hide it behind a `.c` that is only a driver `main()` (`Remimu`)
or a `#define IMPLEMENTATION` shim (`skp`, 11 lines of `.c` in front of 2153 lines of
header). Stage 1 must therefore be able to **mutate headers**, so `mutable_files` picks
whichever of the sources or headers actually carries code, rather than trusting the file
extension; `charness.csv` records the choice in `mode`. All 11 build, and most reach
n ≥ 4 panel members, so they are not a corner case we can drop.

**Library code only.** The first sweep failed 22 projects by greedily compiling demo,
benchmark, CLI and vendored-test-framework sources (`bin/`, `app/`, `benchmark/`,
`unity/`, …). These are not what the Rust interfaces correspond to. Restricting to
`src/` (or the project root) minus those directories, widening include paths to header
*ancestors*, and linking `-lm -lpthread [-lcrypto -lssl]` took usable from 78 → **89**.

### 2.3 The 11 stragglers — all recovered

| project | failure | cause |
|---|---|---|
| `XOpt` | compile | K&R `va_dcl` in vendored `snprintf.c` |
| `gfc`, `tisp` | compile | headers expected at an install path (`gfc/gfc.h`, `tibs.tsp.h`) |
| `libfor` | compile | generated source with undeclared symbols |
| `roaring-bitmap` | compile | SIMD `always_inline` needs `-mpopcnt`/`-march` |
| `clhash`, `matrix_multiplication` | compile | (SIMD/arch flags, same shape) |
| `jccc`, `impcheck` | link | library code calls its own test framework |
| `lambda-calculus-eval` | link | undefined `search` — helper package outside `src/` |
| `ljmm` | link | needs `--wrap=mmap64` linker flag |

Every one fell to a **generic** harness rule — amalgamation `#include "*.c"` sources are
not translation units, a compile-flag fallback chain, `-fcommon`, forwarding include
shims, `--wrap` autodetection, and a second-chance link that pulls in only sources
defining a reported missing symbol. The per-project override table exists but is empty,
and the regression probe (`phase0/recheck.py`) shows 0 regressions among the 90 that
already worked. Full write-up: [`CHARNESS_FIXES.md`](CHARNESS_FIXES.md).

**Caveat.** `-march=native` makes the three SIMD projects' objects host-specific; a
mutation run has to stay on one machine class.

## 3. Recommended pilot set

Selection now runs off the panel and project shape alone (`rmt/pilot.py`): codex-GT-green
∧ ≥ 5 interface functions ∧ ≥ 60 mutable lines leaves **86** projects, stratified over
mutable size with one header library and one parser-heavy project forced in — parsers were
where the earlier four-settings study found the largest differentials.

## 4. Spec deltas

1. **§5 harness: Makefile-first → gcc-uniform** (§2.1 above).
2. **§4 Stage 1 must handle header libraries** — the mutation target is chosen by where
   the code is, not by file extension (§2.2).
3. **§11 open question 2 resolved** — `initial` ≡ `*_output`; panel counted by model
   family; effective panel is 4 families, not 2.
4. **§3 candidate-set table corrected** — all sets cover exactly 100 projects (the earlier
   111/109 counts included stray non-project files).
5. **§11 risk "panel too thin" retired** — 69 projects at n ≥ 3.
6. **§5 risk "C harness too narrow" retired** — 100/100 (§2.3).

## Files

| file | contents |
|---|---|
| `panel.json` | per project: panel members (source, family, path) + rejection reasons |
| `panel_raw.csv` | all 459 candidate × GT-test verdicts with pass/fail counts |
| `charness.csv` | per project (RBench name + `cbench` dir): make status, harness status, mutation target, link libs, cflags, usable |
| `eligible.csv` | per project: C usability × panel size — the Phase 1 selection input |
| `recheck_before/after.{csv,json}` | the harness regression probe, before and after §2.3 |
| `CHARNESS_FIXES.md` | how the 11 stragglers were recovered |
