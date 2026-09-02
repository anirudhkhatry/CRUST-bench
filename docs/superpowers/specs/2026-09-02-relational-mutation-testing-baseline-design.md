# Relational mutation testing between C and Rust — baseline design

**Date:** 2026-09-02
**Status:** design approved · Phase 0 complete (2026-09-02) · Phase 1 ready
**Working dir:** `relational_mutation_testing_baseline/`

---

## 1. The question

Building on standard mutation testing, we want tests that *distinguish* an original
program from a mutated one — tests that **kill mutants**. The baseline asks whether an
LLM can produce such tests **in C**, and whether their discriminating power **survives
translation into Rust**.

```
C source P  ──mutate──▶  P′  ──LLM──▶  C test T  (passes P, fails P′)
                                        │
                                   transpile
                                        ▼
                                    Rust test RT  ──▶  Rust implementation under test
```

The claim we are testing: *a test constructed to separate P from P′ in C remains a
meaningful correctness check for the corresponding Rust implementation.*

If it holds, we have a cheap, LLM-driven source of semantically meaningful tests, and a
natural extension to memory-behavior mutants — tests that specifically target the fault
classes Rust is supposed to eliminate. If it does not hold, the **failure modes are the
deliverable**: we learn whether the LLM cannot discriminate, whether transpilation loses
the discriminating semantics, or whether C-level faults simply do not induce useful
distinctions in Rust.

## 2. Why this is not the `src+test` setting we already measured

`mutation_testing/docs/why_c_tests_hurt_mutation.md` established that handing a model the
**existing C test suite** to translate is the *worst* of four test-generation modalities
(77.4% gpt-5.5 / 84.4% Opus-5, roughly 11–13 points below generating tests directly
against the Rust implementation). The mechanism was a ~50/50 mix of **coverage deficit**
(the C suite never drives the internal helpers) and **oracle weakness** (relational
assertions like `assert_ne!`/determinism instead of exact values).

This baseline differs in exactly the way that matters: the C tests here are
**mutation-targeted**, not human scenario tests. Each one is generated to expose a
specific injected fault, and is admitted only after being *executed* and shown to pass P
and fail P′. Both deficits are addressed by construction — coverage, because the mutant
sits wherever we injected it including internal helpers; oracle strength, because a weak
assertion provably fails the kill gate.

So this is a genuine test of whether the earlier negative result was about *C tests* or
about *those particular C tests*. That reframing is the scientific contribution
regardless of which way the numbers land.

## 3. The constraint that shapes the design: there is no correct Rust

CRUST-bench's `datasets/RBench/<project>/` contains **`src/interfaces/*.rs`, `src/lib.rs`
and `src/bin/tests.rs` only** — Rust interfaces and a human-written ground-truth *test
suite*. There is **no ground-truth Rust implementation**.

Consequence: we cannot certify that a transpiled test `RT` is *sound* (that correct Rust
would pass it) by running it against a known-correct program, because no such program
exists. Soundness must be established **empirically**. Section 6 is how.

What we do have: multiple independent LLM transpilations of every project that **pass the
RBench ground-truth tests** —

| set | model family | projects | GT-green |
|---|---|---|---|
| `datasets/codex_expts/all/initial_codex_output` | codex | 100 | 94 |
| `datasets/claude_expts/all/initial_claude_output` | claude | 100 | 81 |
| `mutation_testing/results/gpt/results_gpt55_4settings/_frozen_source` | gpt-5.5 | 99 | 70 |
| `mutation_testing/results/claude/results_opus5_4settings/_frozen_source` | opus-5 | 99 | 66 |

Measured in Phase 0. The `datasets/*/all/initial` directories are **run scaffolding, not
separate transpilations** — byte-identical to the corresponding `*_output` trees (codex
98/100, claude 100/100) — so they are excluded. Cross-model implementations are distinct
in 100/100 cases, so the panel is counted by **model family**, not by directory.

Independently produced, GT-green implementations of the same interface are the closest
available stand-in for a reference, and their **agreement** is the basis of our soundness
oracle.

## 4. Pipeline

### Stage 0 — Project selection

A project is admitted when it has (a) a CBench C source that builds, (b) an RBench
interface + ground-truth test suite, and (c) at least *n* independent Rust
implementations that **pass the RBench GT tests** (the panel — see §6.1).

Phase 0 measured this. **69 projects reach n ≥ 3; 60 of those also have a working C
harness** and form the eligible set. `n = 2` materially weakens the agreement oracle (it
cannot distinguish "unsound" from "one candidate is buggy"), so n=2 projects (18 more) are
held back and reported separately rather than mixed in. See
`relational_mutation_testing_baseline/results/phase0/REPORT.md`.

### Stage 1 — LLM-generated C mutants

For each selected target function in `src/*.c` — **or `src/*.h`**, since Phase 0 found 7
header-only projects (`bigint`, `c-string`, `clog`, `cset`, `csyncmers`, `libtinyfseq`,
`rect_pack_h`) whose implementation lives in headers — prompt the model for *N*
**single-point, semantically meaningful** mutants, returned as structured records:

```json
{"file": "...", "function": "...", "original_snippet": "...",
 "mutated_snippet": "...", "fault_class": "...", "rationale": "..."}
```

Applied by exact-match replacement, located via `tree-sitter-c` (`src/utils/parse_c.py`).

**Validity gate:** the snippet applies uniquely, `P′` compiles, and `P′ ≠ P` textually.
Failures are recorded, not silently dropped.

We deliberately do **not** attempt to prove non-equivalence — equivalence is undecidable
and, more usefully, an unkillable mutant is itself a reportable outcome (§8, F1).

### Stage 2 — Killing C test

Given P, the mutant diff, and the target function, generate a self-contained C test `T`.

**The gate is executed, not trusted:**

1. build `P + T` → must **compile and pass**
2. build `P′ + T` → must **compile and fail**

Up to *k* retries, feeding back the real compiler and runtime output. This repair loop is
legitimate: the oracle is the C program itself, not any Rust artifact, so no information
leaks into the measurement we care about.

**Value grounding (mandatory).** The model must not *predict* expected outputs. `T` is
generated with instrumented output, executed against P, and its assertions are then
**rewritten to pin the concretely observed values as literals**. The oracle is thereby
anchored to C's real behavior rather than a model's guess — this removes the single
largest source of downstream false alarms.

Outcome per mutant: `killed` (we hold a discriminating `T`) or `unkilled`.

**Anti-tautology check.** `T` must reference the target function's observable behavior,
not merely re-derive the mutation. We additionally record, for each `T`, which *other*
mutants of the same project it kills (**collateral kills**) — a cheap, informative measure
of how specific vs. general the generated tests are.

### Stage 3 — CT → RT transpilation

Transpile `T` against the **Rust interfaces only** (`RBench/<p>/src/interfaces/`), never
against a Rust implementation. Showing the model an implementation would let it write a
test that trivially agrees with whatever that implementation does, destroying the
measurement.

**Repair is restricted to compile errors.** The repair loop never sees assertion failures.
Repairing on test failure would launder away precisely the signal being measured.
Max *k* iterations; exhaustion is recorded as F3.

### Stage 4 — Execution

Every implementation used in any role must first **pass the RBench ground-truth tests**.
So every `RT` verdict below is on a program already known to be GT-green — which is what
makes "RT finds what the human suite misses" a meaningful claim.

**4a — Panel run (soundness + candidate bugs).** Run `RT` against all *n* panel members
and classify by the pass/fail spread (§6.1).

**4b — Rust-mutant arm (controlled positive signal).** Take a GT-green candidate, apply
the **Rust-level counterpart** of the C mutant, and check whether `RT` kills it.

This arm exists because §3 removes our controlled source of known-buggy Rust. Without it,
a positive result depends on the candidates *happening* to contain bugs — an uncontrolled
and possibly near-zero quantity on GT-green code. With it, we get a direct answer to the
headline transfer question: *does a test built to kill fault X in C kill fault X in Rust?*

Counterpart construction, in order of preference:

1. **LLM-applied**: give the model the C mutant diff and the corresponding Rust function,
   ask it to inject the analogous fault. Gate: compiles, differs textually.
2. **Fallback** where (1) fails: match by *fault class* against the operator-generated
   Rust mutants from `mutation_testing/harness/transpilation_operators.py` / `mutate_shared.py`.

Mechanics (applying a mutant and running suites against one build) reuse
`mutation_testing/harness/mutate_shared.py`.

### Stage 5 — Comparison arm

Run the **RBench ground-truth Rust test suite** against the same panel members and the
same Rust mutants. Report the partition: faults found by **RT only**, by **GT only**, by
**both**, by **neither**. The RT-only cell is the value proposition.

*(Deferred, not Phase 1: comparing against the existing `src+test` and `c+rust` numbers
from the four-settings study. It is the cheapest strong result available — it isolates
"mutation-targeted C tests" from "ordinary C tests" — and the baseline numbers already
exist. Recommended as the first follow-up.)*

## 5. The C build harness

**Revised by Phase 0: gcc-uniform, not Makefile-first.** The original plan was Makefile
first with a gcc fallback. Measurement showed that is the wrong order — the Makefile path
builds the *project's own* artifacts and provides no way to link a **generated test main**
against the library objects, which is what Stages 1/2 require. A `make` success does not
imply the project is drivable, and the 71 Makefile successes add no project the gcc path
does not already cover.

The single build path is therefore:

1. Collect library sources: `src/` when present, else the project root, **minus**
   `bin/ app/ demo/ example/ benchmark/ fuzz/ tools/ vendor/ unity/ test(s)/`. These hold
   demos, CLI front-ends and vendored test frameworks that the Rust interfaces do not
   correspond to; including them was the single largest cause of spurious failures.
2. Header-only projects: synthesize one translation unit including the library headers.
3. Compile each to an object with include paths covering every header directory **and its
   ancestors** (sources commonly `#include "pkg/pkg.h"`).
4. Link the objects, minus the project's own `main()`, against the generated test main,
   with `-lm -lpthread` and `-lcrypto -lssl` if needed. The library set is recorded.

The project Makefile is retained only as an optional **cross-check** that our object set
matches the project's own expectations (a missing `-D`, a required optimisation level).

Phase 0 result: **89/100 projects usable.** The 11 failures are per-project flag work
(SIMD arch flags, include roots, `--wrap` linking) and are documented in the Phase 0
report; they are not on the Phase 1 path.

## 6. Establishing that correct Rust would pass RT

Three mechanisms, all adopted.

### 6.1 Agreement oracle over the panel

Run `RT` against the *n* independent, GT-green panel members and classify:

| spread | verdict |
|---|---|
| fails on **all** *n* | `RT` is **unsound** — over-constrained oracle, or C-only semantics. Rejected. |
| passes on **all** *n* | `RT` is sound but **non-discriminating** on this panel. |
| passes some, fails some | `RT` is sound **and** discriminating; the failures are **candidate bug reports**. |

Rationale: *n* independently produced implementations agreeing on a behavior is reasonable
evidence the behavior is correct. All of them failing one assertion is far better explained
by the assertion being wrong than by every transpiler making the same mistake.

**Known blind spot:** a fault that *every* transpiler commits is invisible to this oracle —
it lands in the "fails on all" bucket and gets rejected as unsound. This is a limitation
shared by any test-based method without a reference implementation, and it is why the
"fails on all" bucket is **hand-sampled** rather than merely counted (§6.4).

The rejection rate here is not overhead — it is a headline number: it *is* the rate at
which C-derived tests do not transfer.

### 6.2 Value-grounded oracles

Covered in Stage 2. Assertions carry values recorded from executing P, not values the
model predicted. Eliminates the "LLM hallucinated the expected output" false-alarm class
before it reaches Rust.

### 6.3 C-semantics filter

Some tests discriminate P from P′ using behavior correct Rust may legitimately not
reproduce. Detect (heuristic scan + LLM judge) and **bucket** — do not silently drop:

- signed-integer overflow / wraparound reliance
- uninitialized reads
- out-of-bounds or past-the-end pointer arithmetic
- `int`/`long` width and signedness assumptions
- `-1` / `errno` error-signalling conventions vs. `Result`
- struct layout, padding, or `memcpy`-of-struct reliance

This bucket is a **headline failure mode**, not noise: it quantifies how much of C-mutant
discriminating power is anchored to semantics Rust deliberately does not have.

### 6.4 Human triage sample

Hand-inspect a fixed sample (target: 30 verdicts, stratified across the buckets above,
with the "fails on all" bucket over-sampled) to calibrate how often the automatic
classification is right. Reported as a confusion table alongside the main numbers. This is
what makes the automated classification defensible to an external audience.

## 7. Metrics — the funnel

Every stage is a measured attrition point:

```
mutants proposed
  → M1  valid (applies, compiles, differs)
  → M2  killed in C          (T passes P, fails P′)          ← "can the LLM discriminate?"
  → M3  RT compiles          (transpilation yield)
  → M4  RT admitted sound    (not rejected by §6.1/§6.3)     ← "does the semantics survive?"
  → M5a RT kills the Rust-level counterpart mutant           ← THE HEADLINE (transfer rate)
  → M5b RT fails on ≥1 GT-green candidate (bug found)
```

Supporting: collateral-kill distribution; GT-suite comparison partition (§5); per-project
and per-fault-class breakdowns; build-path split (§5); panel size *n* per project.

## 8. Failure-mode taxonomy

The deliverable if the pipeline does not work end-to-end. Every artifact is labeled:

- **F1 — Mutant unusable.** Doesn't apply, doesn't compile, or is equivalent (unkillable
  in C by any test the model produced).
- **F2 — No discriminating C test.** The model could not separate P from P′ within *k*
  attempts. Sub-split: never compiled vs. compiled but passed on P′.
- **F3 — RT will not compile.** C↔Rust API shape mismatch — the dominant expected loss,
  since CRUST-bench Rust interfaces are idiomatic (`Result`, slices, ownership) rather
  than C-shaped (out-params, raw pointers, sentinel returns).
- **F4 — Semantics lost.** RT compiles but is rejected as unsound. Sub-classed by the
  §6.3 categories, plus: numeric precision/formatting, error-signalling translation,
  ownership/aliasing idiom.
- **F5 — Non-transfer.** RT is sound but does **not** kill the Rust-level counterpart
  mutant (M4 → M5a attrition). The most interesting negative: the fault exists in Rust and
  the C-derived test cannot see it.

## 9. Phasing

**Phase 0 — feasibility (no LLM calls). ✅ DONE 2026-09-02.** Measured panel size per
project, verified the C build harness across CBench, confirmed which panel members are
GT-green. **Result: 60 eligible projects** (C harness usable ∧ n ≥ 3); 89/100 C harness
usable; 69/100 at n ≥ 3. Full results and the resulting spec deltas:
`relational_mutation_testing_baseline/results/phase0/REPORT.md`.

**Phase 1 — pilot.** ~10 projects × ~10 mutants. **Every** failure hand-inspected; the
§8 taxonomy is populated and refined against real artifacts rather than guessed. Includes
the human triage sample (§6.4).

**Phase 2 — scale.** Extend to the full comparable set (~60–100 projects) **only if the
pilot shows signal**. This mirrors how the existing mutation study was run
(`run_opus5_sample.py` → `run_transpile_full.py`).

**Phase 3 (future, out of Phase 1 scope) — memory-behavior mutants.** The DARPA-facing
extension: off-by-one bounds, missing/double free, use-after-free, uninitialized read,
aliasing violations. Highest value, hardest stage — many such mutants are *unviable* in
Rust (the type system rejects them), which is itself the point worth demonstrating. The
existing finding that **~62% of Rust mutants are unviable** is the relevant prior.

## 10. Implementation layout

Under `relational_mutation_testing_baseline/`:

```
run_relational.py        orchestrator (stages 0–5, resumable, per-mutant checkpointing)
c_mutate.py              Stage 1 — LLM C mutants + tree-sitter application/validation
c_testgen.py             Stage 2 — killing C test + execution gate + value grounding
c_harness.py             §5 build/run harness (Makefile-first, gcc fallback)
transpile_test.py        Stage 3 — CT→RT against interfaces, compile-only repair
rust_run.py              Stage 4 — panel run, Rust-mutant arm, GT-suite arm
soundness.py             §6 agreement oracle + C-semantics filter
analyse.py               §7 funnel + §8 taxonomy → CSVs
prompts/                 mutate_c, testgen_c, transpile_test, mutate_rust_counterpart
results/<run>/<project>/mutants/<id>/
    mutant.json  patch.diff  test.c  test.rs  verdicts.json
```

**Reuse, do not reimplement:** `src/endpoints/call_endpoint.py` +
`src/endpoint_config.py` (model dispatch and caching), `src/benchmark.py` (project
loading), `src/utils/parse_c.py` (tree-sitter-c), `src/utils/compile_c_utils.py`,
`src/utils/rust_project_builder.py`, `mutation_testing/harness/mutate_shared.py` (mutant
application + suite execution), `mutation_testing/harness/split_manifests.py` (standalone crate
manifests).

Default model: `claude-opus-5` via the existing endpoint config; configurable, and the
pilot should be repeated on a second model before any claim is made about the *modality*
rather than the model — this is what made the four-settings result credible.

## 11. Risks and open questions

| risk | mitigation |
|---|---|
| ~~Panel too thin (*n* < 3) — agreement oracle weak~~ | **Retired by Phase 0**: 69 projects reach n ≥ 3 across 4 distinct model families |
| F3 dominates — idiomatic Rust interfaces make CT→RT mechanical translation infeasible | Expected; it is a *result*, not a blocker. Quantify per API-shape category to say precisely which C idioms fail to cross |
| Candidates are GT-green and genuinely bug-free → M5b ≈ 0 | Exactly why the Rust-mutant arm (4b) is in Phase 1 |
| LLM mutants are not single-point, or are semantically null | Validity gate + structured records; equivalent mutants surface as F1 and are reported |
| Correct-but-different Rust rejected as a candidate bug | Agreement oracle + value grounding + §6.3 filter + human triage sample |
| Cost | Phase 0 gate, then 10×10 pilot before any full-set run |

**Open questions to resolve during Phase 0/1:**

1. Target-function selection — all functions, or sampled by complexity/coverage? Sampling
   policy affects comparability with the four-settings numbers.
2. ~~Are the `initial` and `initial_*_output` sets independent?~~ **Resolved by Phase 0:**
   they are not — `initial` is run scaffolding, byte-identical to `*_output`. The panel is
   counted by model family instead, giving 4 independent families, not 2.
3. Retry budget *k* for Stages 2 and 3 — pick from pilot data, then freeze.

## 12. Out of scope

Fuzzing/differential-execution test generation (a useful future control arm, but it
answers a different question than "are LLMs effective at this"); repairing the Rust
implementations; changing the RBench interfaces; and any comparison against the existing
four-settings numbers beyond the deferred follow-up noted in §5.
