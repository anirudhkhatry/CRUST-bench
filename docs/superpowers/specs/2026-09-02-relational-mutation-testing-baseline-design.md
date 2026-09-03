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
available stand-in for a reference, and their **agreement** was the original basis of the
soundness oracle.

**Revised for Phase 1: one reference, not a panel.** The agreement oracle answers "would
correct Rust pass RT?" indirectly and expensively, and its verdicts are hard to explain to
an external audience ("three of four implementations agree"). Phase 1 instead **assumes the
codex transpilation is correct wherever it is GT-green** (94 projects) and calls it `M`.
That converts the whole study into one executed question per mutant:

> the same fault is injected into `M` to give `M′`; does `RT` **pass `M`** and **fail `M′`**?

Both halves are executed, on the same implementation, under the same interfaces — no
cross-implementation reasoning is involved. `RT` failing `M` is *unsoundness* (F3); `RT`
passing `M′` is *non-transfer* (F4). The panel measurement of §6.1 stands as recorded and
is the fallback if the codex-is-correct assumption has to be revisited; it is not on the
Phase 1 path.

## 4. Pipeline

### Stage 0 — Project selection

A project is admitted when it has (a) a CBench C source our harness can build **and run a
generated test main against**, and (b) a GT-green codex implementation to serve as `M`.
Two soft filters follow from what a mutant needs to be useful: ≥ 5 interface functions (a
test must reach the fault through the Rust API) and ≥ 60 mutable C lines (ten mutants in
distinct functions have to fit).

Phase 0 measured this. The C harness now drives **100/100** projects, so eligibility is
decided by the codex GT-green set: 94 projects, of which **86** clear the two soft filters.
Selection is stratified over mutable size with one header library and one parser-heavy
project forced in (`rmt/pilot.py`). See
`relational_mutation_testing_baseline/results/phase0/REPORT.md`.

### Stage 1 — LLM-generated C mutants

The mutation target is whichever of the library sources or headers actually **carries the
code**, not whichever has a `.c` extension: 11 projects are header libraries, either with
no source at all (`bigint`, `cset`, `csyncmers`, …), a lone driver `main()` (`Remimu`), or
a `#define IMPLEMENTATION` shim in front of the header (`skp`: 11 lines of `.c`, 2153 of
`.h`). Headers are compiled either way, so a patch to one takes effect unchanged.

Prompt the model for *N* **single-point, semantically meaningful** mutants, returned as
structured records:

```json
{"file": "...", "function": "...", "original_snippet": "...", "mutated_snippet": "...",
 "fault_class": "...", "semantic_change": "...", "observable_effect": "...",
 "reachable_via": "..."}
```

Applied by exact-match replacement.

**`semantic_change` is load-bearing, not documentation.** It is the *only* thing that
crosses from C to Rust: Stage 4 constructs `M′` from this prose and never sees the C diff
(§4 Stage 4). It must therefore be language-independent — "the loop stops one element
early, so the last element is never examined", not "changed `<=` to `<`". The gate rejects
records whose `semantic_change` is too thin to port.

**Validity gate:** the snippet applies uniquely, `P′` compiles, `P′ ≠ P` textually, and the
description is substantive. Failures are recorded, not silently dropped.

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

**Value grounding falls out of the gate.** An asserted literal survives only if the
original C library actually produces it — step 1 executes `T` against `P` and a wrong
expected value fails there, with the real output fed back for repair. No separate
instrument-and-rewrite pass is needed: the oracle is anchored to C's behavior by
construction, which removes the single largest source of downstream false alarms.

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
Max *k* iterations; exhaustion is recorded as F2.

**Assertions may not be weakened** to make the code compile. Where a C assertion has no
expressible Rust counterpart, the model must mark it `// UNTRANSLATABLE:` and leave it out
rather than relax it; those markers are counted, so a test that compiled by giving up on
what it was checking is visible in the results instead of scoring as a success.

### Stage 4 — Execution

`M` is the codex implementation, which is GT-green by construction (Stage 0). So every
verdict below is on a program already known to pass the human suite — which is what makes
"RT finds what the human suite misses" a meaningful claim.

**4a — Build `M′`.** A second LLM call is given the Rust implementation, the interfaces,
and the mutant's `semantic_change` / `observable_effect` prose — **never the C diff, and
never the C source**. It injects the described fault into `M`. Gate: compiles, differs
textually. Withholding the diff is what makes the arm meaningful: the Rust is an
independent re-implementation, so a diff would invite a syntactic transliteration that may
not even correspond to the same behavior. Prose is the only honest transfer medium, and
"the fault could not be expressed in this Rust" is itself a reportable outcome (F5) — the
Phase 3 unviability finding in miniature.

**4b — The two runs.** `RT` on `M` must **pass**; `RT` on `M′` must **fail**. That
conjunction is the headline: *does a test built to kill fault X in C kill fault X in Rust?*
Failing `M` is F3 (the transpilation changed the meaning) and short-circuits — a test that
does not hold on correct Rust says nothing about `M′`.

**4c — Control: the GT suite on `M′`.** Run the RBench ground-truth suite against `M′`. If
`RT` does not kill `M′` **and** neither does the GT suite, the likely explanation is that
`M′` barely differs (weak or equivalent mutant), not that `RT` is weak. Without this the
F4 count is uninterpretable.

This design exists because §3 removes our controlled source of known-buggy Rust. Without a
constructed `M′`, a positive result would depend on the reference *happening* to contain
bugs — an uncontrolled and possibly near-zero quantity on GT-green code.

### Stage 5 — Comparison arm

The same GT run as §4c, read as a comparison rather than a control: over all `M′`, report
the 2×2 partition of faults found by **RT only**, by **GT only**, by **both**, by
**neither**. The RT-only cell is the value proposition.

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

Phase 0 result: **100/100 projects usable.** The 11 stragglers were recovered by generic
rules, not per-project flags — amalgamation `#include "*.c"` sources are excluded from the
compile set, a compile-flag fallback chain (plain → `-march=native` → `+ -std=gnu99`)
handles SIMD intrinsics, `-fcommon` handles tentative definitions, unresolvable includes
get forwarding shims in a scratch dir, `--wrap` flags are autodetected, and an undefined
symbol triggers a second-chance link that pulls in only sources *defining* it. A
per-project override table exists and is empty. Details: `results/phase0/CHARNESS_FIXES.md`.

**Caveat.** `-march=native` makes three SIMD projects' objects host-specific, so a run has
to stay on one machine class.

## 6. Establishing that correct Rust would pass RT

Phase 1 answers this **directly**: `RT` is run against `M`, the implementation we take as
correct (§3). Passing is soundness; failing is F3. §6.1 below is the indirect mechanism it
replaced — retained because it is measured, and because it is the fallback if the
codex-is-correct assumption fails. §§6.2–6.4 remain in force.

### 6.1 Agreement oracle over the panel *(not on the Phase 1 path)*

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
  → M1  valid (applies, compiles, differs, described)
  → M2  killed in C          (T passes P, fails P′)      ← "can the LLM discriminate?"
  → M3  RT compiles          (transpilation yield)
  → M4  M′ constructed       (the fault exists in Rust)
  → M5  RT passes M          (sound on correct Rust)     ← "does the semantics survive?"
  → M6  RT fails M′                                      ← THE HEADLINE (transfer rate)
```

Supporting: collateral-kill distribution; the GT-on-`M′` partition (§5); per-project and
per-fault-class breakdowns; `// UNTRANSLATABLE:` counts; repair attempts per stage.

## 8. Failure-mode taxonomy

The deliverable if the pipeline does not work end-to-end. Every artifact is labeled with
the outcome of its one terminal question; a mutant that fails the Stage 1 validity gate
never enters the funnel and is counted there instead.

- **F1 — No discriminating C test.** The model could not separate P from P′ within *k*
  attempts, or declared the mutant equivalent. Sub-split: never compiled vs. compiled but
  passed on P′.
- **F2 — RT will not compile.** C↔Rust API shape mismatch — the dominant expected loss,
  since CRUST-bench Rust interfaces are idiomatic (`Result`, slices, ownership) rather
  than C-shaped (out-params, raw pointers, sentinel returns).
- **F3 — Unsound on correct Rust.** RT compiles but fails `M`: the transpilation changed
  the meaning. Sub-classed by the §6.3 categories, plus numeric precision/formatting,
  error-signalling translation, ownership/aliasing idiom.
- **F4 — Non-transfer.** RT passes `M` and also passes `M′` — discriminating power lost in
  translation. The most interesting negative: the fault exists in Rust and the C-derived
  test cannot see it. Read against the §4c control, which says whether `M′` was killable at
  all.
- **F5 — No Rust counterpart, or infrastructure error.** The described fault could not be
  injected into `M` (Rust's types may forbid it — the Phase 3 finding in miniature), or a
  build/API failure prevented a verdict. The two are recorded separately.

## 9. Phasing

**Phase 0 — feasibility (no LLM calls). ✅ DONE 2026-09-02, harness revised 2026-09-03.**
Measured panel size per project, verified the C build harness across CBench, confirmed
which candidates are GT-green. **Result: 100/100 C harness usable; codex GT-green 94/100;
86 eligible** after the Stage 0 filters (69/100 reach panel n ≥ 3, retained as the
fallback oracle). Full results and spec deltas:
`relational_mutation_testing_baseline/results/phase0/REPORT.md`.

**Phase 1 — pilot.** ~10 projects × ~10 mutants. Pipeline built and smoke-tested
end-to-end (`geofence`, 2/2 mutants KILLED); projects and mutants fan out in parallel and
every stage is resumable from disk. **Every** failure hand-inspected; the §8 taxonomy is
populated and refined against real artifacts rather than guessed. Includes the human triage
sample (§6.4).

*Known confound to address before Phase 2:* the same model (`claude-opus-5`) both writes
the `semantic_change` description and reads it to build `M′`, so transfer is measured under
self-consistency. Stage 4 should be repeated with a different model.

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
run.py                   the entrypoint: select | run | report
rmt/
  llm.py                 Anthropic client — disk cache, concurrency cap, retries
  cbench.py              §5 build/run harness; mutation-target selection
  rbench.py              interfaces, the reference impl, cargo test injection, GT runs
  stage1_mutate.py       Stage 1 — mutants + validity gate
  stage2_testgen.py      Stage 2 — killing C test + executed gate + collateral kills
  stage3_transpile.py    Stage 3 — CT→RT against interfaces, compile-only repair
  stage4_relate.py       Stage 4 — build M′ from prose, run RT on M and M′, GT control
  pilot.py               Stage 0 selection
  driver.py              parallel + resumable orchestration
  report.py              §7 funnel + §8 taxonomy
  prompts/*.md           every prompt, versioned separately from the code
phase0/                  panel.py, charness.py, recheck.py
results/<run>/<project>/
  stage1.json  mutants.json  mutants/<id>/{mutant.json,test.c,test.rs,verdicts.json}
results/.llm_cache/      content-addressed, so a resumed run repeats no API call
```

**Parallel and resumable.** Projects and mutants both fan out, capped at 16 in-flight API
requests. Re-issuing an interrupted command recomputes nothing: `mutants.json` skips a whole
project, `verdicts.json` skips a mutant's gcc and cargo work, `stage1.json` skips
re-proposing, and the LLM cache covers the rest. `--redo` ignores all of it.

**Deliberately standalone.** The pipeline does not reuse `src/endpoints/*` or
`mutation_testing/harness/*`, as originally planned. Those carry a model-dispatch layer and
a mutation harness built for a different question, and threading this study's needs
(streaming, in-flight caps, per-mutant checkpoints, a C harness that links a *generated
test main*) through them was more coupling than code saved.

Default model: `claude-opus-5` (adaptive thinking, `effort=high`), overridable via
`RMT_MODEL`. The pilot should be repeated on a second model before any claim is made about
the *modality* rather than the model — this is what made the four-settings result credible.

## 11. Risks and open questions

| risk | mitigation |
|---|---|
| ~~Panel too thin (*n* < 3) — agreement oracle weak~~ | **Retired by Phase 0**: 69 projects reach n ≥ 3 across 4 distinct model families — and Phase 1 no longer depends on the panel (§3) |
| ~~C harness covers too few projects~~ | **Retired by Phase 0**: 100/100 (§5) |
| The codex reference is not actually correct on some project | It is GT-green by construction, and F3 (RT fails `M`) surfaces disagreement rather than hiding it. The panel remains the fallback |
| One model both writes and reads the fault description (self-consistency) | Repeat Stage 4 on a second model before any transfer claim (§9) |
| F3 dominates — idiomatic Rust interfaces make CT→RT mechanical translation infeasible | Expected; it is a *result*, not a blocker. Quantify per API-shape category to say precisely which C idioms fail to cross |
| The reference is GT-green and genuinely bug-free, so nothing is there to find | Exactly why `M′` is constructed (§4 Stage 4) rather than hoping for latent bugs |
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
