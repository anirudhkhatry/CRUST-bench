# Phase 0 — feasibility results

**Date:** 2026-09-02 · no LLM calls · scripts: `phase0_panel.py`, `phase0_charness.py`

Phase 0 asked two go/no-go questions from the design spec: **is the soundness panel big
enough**, and **can we drive the C build harness**. Both pass.

## Headline

**60 projects are fully eligible** (C harness works *and* panel size n ≥ 3).
78 projects at n ≥ 2. The pilot needs 10.

| | n=0 | n=1 | n=2 | n≥3 | total |
|---|---|---|---|---|---|
| C harness usable | 3 | 6 | 18 | **60** | 87 |
| C harness unusable | 2 | 1 | 1 | 9 | 13 |

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

89/100 CBench projects compile all library sources **and** link them against a generated
test main (the stub link probe — which is what Stages 1/2 actually need).

| gcc fallback | count | | project Makefile | count |
|---|---|---|---|---|
| ok | **89** | | ok | 71 |
| compile_failed | 7 | | failed | 16 |
| link_failed | 4 | | no_makefile | 11 |
| | | | timeout | 2 |

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

**Header-only libraries (7 projects).** `bigint`, `c-string`, `clog`, `cset`, `csyncmers`,
`libtinyfseq`, `rect_pack_h` have their implementation in `src/*.h`, not `src/*.c`. The
first sweep reported them as "no sources". Stage 1 must therefore be able to **mutate
headers**, and the harness synthesizes a translation unit including them. All 7 build; 6
of them reach n ≥ 4 panel members, so they are not a corner case we can drop.

**Library code only.** The first sweep failed 22 projects by greedily compiling demo,
benchmark, CLI and vendored-test-framework sources (`bin/`, `app/`, `benchmark/`,
`unity/`, …). These are not what the Rust interfaces correspond to. Restricting to
`src/` (or the project root) minus those directories, widening include paths to header
*ancestors*, and linking `-lm -lpthread [-lcrypto -lssl]` took usable from 78 → **89**.

### 2.3 The remaining 11

| project | failure | cause |
|---|---|---|
| `XOpt` | compile | K&R `va_dcl` in vendored `snprintf.c` |
| `gfc`, `tisp` | compile | headers expected at an install path (`gfc/gfc.h`, `tibs.tsp.h`) |
| `libfor` | compile | generated source with undeclared symbols |
| `roaring-bitmap` | compile | SIMD `always_inline` needs `-mpopcnt`/`-march` |
| `clhash`, `matrix_multiplication` | compile | (SIMD/arch flags, same shape) |
| `jccc`, `impcheck` | link | library code calls its own test framework |
| `lambda-calculus-eval` | link | undefined `search` — genuinely incomplete |
| `ljmm` | link | needs `--wrap=mmap64` linker flag |

Most are per-project flag work (arch flags, include roots). Not worth doing now: 60
eligible projects far exceeds the pilot's needs. Revisit only if Phase 2 wants the full
set.

## 3. Recommended pilot set

59 projects have n ≥ 4 and a working C harness. Take the first 10 by panel size and
source-size diversity, including at least one header-only project (`bigint`) and one
parser-heavy project — parsers were where the earlier four-settings study found the
largest differentials.

## 4. Spec deltas

1. **§5 harness: Makefile-first → gcc-uniform** (§2.1 above).
2. **§4 Stage 1 must handle header-only projects** — mutation targets include `src/*.h`.
3. **§11 open question 2 resolved** — `initial` ≡ `*_output`; panel counted by model
   family; effective panel is 4 families, not 2.
4. **§3 candidate-set table corrected** — all sets cover exactly 100 projects (the earlier
   111/109 counts included stray non-project files).
5. **§11 risk "panel too thin" retired** — 69 projects at n ≥ 3.

## Files

| file | contents |
|---|---|
| `panel.json` | per project: panel members (source, family, path) + rejection reasons |
| `panel_raw.csv` | all 459 candidate × GT-test verdicts with pass/fail counts |
| `charness.csv` | per project: make status, gcc status, mode, link libs, mains, usable |
| `eligible.csv` | per project: C usability × panel size — the Phase 1 selection input |
