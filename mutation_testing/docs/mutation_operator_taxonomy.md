# Mutation operators: the taxonomy we settled on, and the run in flight

Two mutant populations exist in this study. The first is cargo-mutants' — language-generic
bug shapes that know nothing about translation. The second is a set of eleven operators
written for this study that model the specific places where C and Rust disagree, and
therefore the specific places a model translating between them can go wrong.

This document records what those operators are, why each is defined the way it is, what
was deliberately left out, and the state of the run currently scoring them across the
Claude corpus.

- Spec: [`superpowers/specs/2026-08-31-transpilation-mutation-operators-design.md`](./superpowers/specs/2026-08-31-transpilation-mutation-operators-design.md)
- Plan: [`superpowers/plans/2026-08-31-transpilation-operator-generator.md`](./superpowers/plans/2026-08-31-transpilation-operator-generator.md)
- Implementation: `transpilation_operators.py`, adapted to the harness by
  `list_transpile_mutants.py`

---

## 1. Three axes, not one

"Kind of mutation" is ambiguous between three orthogonal classifications, and all three
are load-bearing here.

| axis | question | where it lives |
|---|---|---|
| **syntactic class** | what edit was made? | cargo-mutants' `genre` field |
| **fault class** | what real bug does the edit simulate? | `Operator.family` (F1/F2/F3) |
| **status** | did it compile, and could it ever be killed? | the run's verdict |

Axes 1 and 2 are not the same partition, which is the whole reason the second population
exists. `+` → `-` and `+` → `wrapping_add` are the same *site* and different syntactic
classes. `+` → `wrapping_add` and `a.wrapping_add(b)` → `a + b` are different syntactic
classes and the *same* fault class. A taxonomy built only on edit shape cannot express
"overflow discipline was translated wrongly," which is the thing we want to measure.

---

## 2. Level A — classical operators (cargo-mutants)

Roughly the standard AOR/ROR/LOR set plus statement deletion. Counts are the Opus-5
5-project sample, one setting.

| genre | n | edit |
|---|---|---|
| `BinaryOperator` | 827 | swap a binary operator within its arity class: `* == + / != -`, comparisons `< > <= >=`, bitwise `& \| ^`, logical `&& \|\|`, and the compound-assign forms `*= -= \|= &=` |
| `FnValue` | 309 | delete the body, return a plausible constant: `0`, `1`, `()`, `Default::default()`, `Ok(…)`, `None` |
| `MatchArm` | 33 | delete a match arm |
| `UnaryOperator` | 24 | delete `!` or `-` |
| `MatchArmGuard` | 2 | delete an arm guard |

Nothing in this set knows the code was translated from C. That is not a defect — it is the
baseline the second population is defined against.

### 2.1 What the classical operators showed

Two analyses of this population are already written up in full —
[`mutant_distribution_and_memory_types.md`](./mutant_distribution_and_memory_types.md) for
the distribution and type view, and
[`why_c_tests_hurt_mutation.md`](./why_c_tests_hurt_mutation.md) for the `src+test`
deficit. The results that bear on operator design are these.

**Rust's type system rejects most of the corpus.** Across both models' full runs (12,208
mutants):

| outcome | count | share |
|---|---|---|
| unviable (does not compile) | 7,536 | **61.7%** |
| caught | 2,939 | 24.1% |
| timeout | 1,196 | 9.8% |
| missed | 537 | 4.4% |

So the *scored* population is a sixth of what is generated, and the classical operators'
unviability is itself type-driven: return-value mutants 68.1% unviable, binary-operator
60.3%, delete-branch 55.4%.

**Unviability concentrates on exactly the types that matter.** Grouping return-value
mutants by return type: custom structs and enums are **80.3% unviable**, versus 63.0% for
primitives. The type system is most aggressive precisely where the abstraction is richest,
so the mutants that survive to be scored are disproportionately about primitive arithmetic.
The one clear blind spot on this axis is `Option<T>` — 435 mutants, and of the viable ones
only **68.0% killed**, the lowest of any type bucket.

**On the 5-project sample the classical set is close to saturated.** Common mutant set
(viable and non-timeout in every setting):

| config | classical (n=1101) | transpilation (n=356) |
|---|---|---|
| `src` | 72.3% | 41.6% |
| `src+test` | 37.3% | 13.8% |
| `src/test` | 74.5% | 26.7% |
| `c+rust` | **86.0%** | 39.6% |

One caveat on that `src+test` cell: `clog` produced no `src+test` result in the classical
run, so its 37.3% is pooled over the four projects that did (n=882), while the other three
columns cover all five (n=1101). The transpilation run has all four settings on all five
projects.

Per genre, on the same set:

| genre | n | src | src+test | src/test | c+rust |
|---|---|---|---|---|---|
| `BinaryOperator` | 796 | 69.0 | 34.1 | 69.1 | 83.3 |
| `FnValue` | 247 | 82.2 | 46.6 | 87.9 | 92.3 |
| `MatchArm` | 33 | 81.8 | 26.9 | 97.0 | 100.0 |
| `UnaryOperator` | 23 | 65.2 | 57.1 | 82.6 | 91.3 |
| `MatchArmGuard` | 2 | 100.0 | 0.0 | 100.0 | 100.0 |

**There are no blind-spot classes at all.** Every substitution with n ≥ 3 is killed by at
least one configuration. Contrast the transpilation set, which has five substitution
classes surviving in *every* configuration. This is the empirical case for the second
operator population: at 86% with no class-level gaps, the classical operators have very
little discriminating power left between the stronger settings, and `src/test` vs `c+rust`
(74.5% vs 86.0%) is carried mostly by volume rather than by any identifiable fault class.

**The one striking class-level pattern is bitwise.** On the common set:

| substitution | n | src | src+test | src/test | c+rust |
|---|---|---|---|---|---|
| `^= -> \|=` | 27 | **100.0** | 3.7 | 40.7 | **100.0** |
| `^ -> &` | 23 | **100.0** | 26.1 | 39.1 | **100.0** |
| `^ -> \|` | 23 | **100.0** | 26.1 | 26.1 | **100.0** |
| `>> -> <<` | 11 | **100.0** | 9.1 | 54.5 | **100.0** |
| `& -> \|` | 10 | **100.0** | 30.0 | 90.0 | **100.0** |
| `<< -> >>` | 9 | **100.0** | 22.2 | 66.7 | **100.0** |

The two configurations that reach 100% are the two that see the C source without also
being handed the C tests. `src/test`, which never sees C at all, collapses to a quarter or
half on the same mutants. The natural reading is that bit-manipulation intent is legible in
the C source and largely unrecoverable from the Rust implementation alone — a translated
`^=` looks like any other compound assignment once the C is gone. `src+test` also has the
C source and still scores 3.7–30%, which is the translated-C-test mechanism documented
separately, not a counterexample.

**The `src+test` deficit is broad, not narrow.** Mutants killed by `src/test` but surviving
`src+test`, led by `body -> Default::default()` (18 of 20, **90%**) and
`body -> Ok(Default::default())` (6 of 6, **100%**) — i.e. the return-value class where a
test has to actually assert on the returned value rather than merely call the function.

**Timeouts are a real interpretive problem here, unlike for the transpilation set.**
Corpus-wide, timeouts are 9.8% of all classical mutants (1,196), and 32–38 per config on
the sample — large enough that the `strict`/`folded` gap changes conclusions for individual
projects (emlang moves 0.477 → 0.893, clog 0.578 → 0.789). The transpilation run's sample
had 4 timeouts in total, so its two readings are within 0.6 points of each other
everywhere.

---

## 3. Level B — transpilation operators

Eleven operators in three fault families. Every one is a tree-sitter query plus a rewrite
function; the module deliberately knows nothing about CRUST-bench layout.

### F1 — integer discipline

C wraps on overflow and defines integer widths by platform. Rust panics in debug, and
requires the discipline to be named. The translator has to make a judgement call at every
arithmetic site, and can make it wrongly.

| operator | edit |
|---|---|
| `overflow-discipline` | **bidirectional** across `a + b` ↔ `a.wrapping_add(b)` ↔ `a.saturating_add(b)`, and the same for `sub` / `mul` |
| `shift-discipline` | `<<` → `wrapping_shl`, `>>` → `wrapping_shr` |
| `cast-width` | move one rung on the integer ladder: `as u8` → `as u16`, `as u32` → `as u16`/`as u64` |
| `cast-sign` | flip signedness: `as i32` → `as u32` |

Four design decisions worth recording:

- **Bidirectionality is the point.** Dropping a `wrapping_add` for a bare `+` models the
  translator forgetting that C wraps; adding one models it over-applying the fix. Only
  generating one direction would measure half the fault class.
- **One rung, not arbitrary jumps.** `as u32` → `as u16` is a plausible translator slip;
  `as u32` → `as u8` is not, so the ladder is a neighbour map rather than a full product.
- **`div` and `rem` are deliberately absent** from the wrapping families. C defines no
  wraparound for `/` or `%`, so swapping discipline there models no divergence at all — and
  `wrapping_div` differs from `/` only at `iN::MIN / -1`, and for unsigned types for no
  input whatsoever. Equivalent mutants, decidable for free at generation time, so they are
  never emitted rather than quarantined later at ~17 s of build time each.
- **`usize`/`isize` are excluded from the sign flip.** They are chosen for indexing, not
  for sign, so a flip there models nothing real.

### F2 — representation

C's byte-level assumptions are implicit; Rust makes them explicit and the translator has
to pick.

| operator | edit |
|---|---|
| `text-unit` | `.chars()` ↔ `.bytes()`, and `.len()` → `.chars().count()` |
| `endianness` | `from_le_bytes` ↔ `from_be_bytes`, `to_le_bytes` ↔ `to_be_bytes` |
| `float-width` | `as f32` ↔ `as f64` |

`text-unit` exists because C's `strlen` counts bytes: a translator reaching for `chars()`
is right on ASCII and wrong on everything else — exactly the bug a weak suite cannot see.

`endianness` treats `*_ne_*` asymmetrically: it is a valid rewrite **source** but never a
**target**. On a little-endian host `from_ne_bytes` → `from_le_bytes` is a guaranteed no-op,
so emitting it would add nine uncatchable mutants; but the reverse direction found a
genuinely live `from_ne_bytes` → `from_be_bytes` at `cset.rs:76`, so deleting the operator
outright would have cost a real finding.

### F3 — ownership and interface

The structural gap between C pointers and Rust ownership, which has no analogue in the
classical operator set at all.

| operator | edit |
|---|---|
| `arg-transposition` | swap adjacent call arguments |
| `range-bound` | `..` → `..=`, `..=` → `..`, and shorten a range end by one |
| `error-path-erasure` | `v?` → `v.unwrap_or_default()` |
| `lost-in-place-update` | `&mut x` → `&mut x.clone()` |

- `arg-transposition` is the classic `memcpy(dst, src)` vs `copy(src, dst)` slip. Identical
  adjacent arguments are skipped: `g(x, x)` would otherwise produce a textually identical
  no-op mutant.
- `lost-in-place-update` models the standard escape when the borrow checker fights a
  translation — clone, and the caller's update silently vanishes. `&mut x.clone()` borrows
  a temporary, which Rust permits in argument position, so the mutant compiles and does
  nothing.
- `error-path-erasure` models C's `-1`/`NULL` sentinel becoming a swallowed error.

### The precedence guard

The most serious bug found in review: `_rewrite_overflow` and `_rewrite_shift` spliced raw
operand text without parenthesising, so Rust's precedence reassociated the result.
`365 * yoe + yoe / 4` became `365 * yoe.wrapping_add(yoe / 4)` — a different edit from the
one the mutant's own name described. 31 of 776 records were affected, and those that
compiled entered the denominator mislabelled.

The fix is an `_ATOMIC_EXPR` allow-list: an expression is atomic if a postfix `.method(..)`
binds to the whole of it. Anything not on the list gets parentheses, so unknown node types
fall on the safe side. Parenthesising is semantically neutral in Rust, so the conservative
direction is free.

### The implemented subset

Each operator implements the arm that fires most often on real transpiled code, not every
arm the spec describes:

- `error-path-erasure` implements only `?` → `unwrap_or_default()`, not
  `unwrap_or(x)` → `unwrap_or(Default)`.
- `arg-transposition` covers call sites only, not struct-literal fields.
- `lost-in-place-update` covers `&mut` *arguments* only, not `&mut` receivers.
- `overflow-discipline` and `shift-discipline` omit the `checked_*().unwrap()` arm.

**This matters downstream:** the witness gate must not read the absence of a witness for
one of these shapes as evidence that the bug class does not occur. It is evidence only
that this generator does not currently emit that arm.

---

## 4. Status — and why it is the interesting axis

| status | meaning |
|---|---|
| `Unviable` | does not compile; dropped from the denominator |
| `CaughtMutant` | some test failed |
| `MissedMutant` | survived |
| `Timeout` | hung the suite. Semantically a kill, but scored as neither, so both `strict` (excluded) and `folded` (counted as kills) readings are reported |
| *equivalent* | compiles, is semantically identical to the original, and therefore **cannot** be killed. Not a status any harness can detect — it has to be reasoned about |

tree-sitter gives syntax, not types, so some generated mutants will not compile. That is
acceptable rather than sloppy: the harness already treats a failed build as `Unviable`, so
imprecision costs build time, not correctness. rustc is a free type-checker that prunes
roughly half the corpus.

Two results make this axis the one that decides how the numbers should be read.

**Rust's type system deletes an entire fault class for free.** On the 5-project sample,
`cast-width` and `cast-sign` generated 182 mutants and **all 182 were unviable** — a 100%
rejection rate. The C integer-conversion bug class is genuine and common, and at these
sites it is statically unreachable in Rust. That is a finding about the language, not about
the test suites.

**Equivalence is concentrated, not diffuse.** The bare-operator → `wrapping_*` direction of
F1 sat at a uniform 0.0% kill rate across four independent suites — 113 mutants, 32% of the
scored denominator. In release mode `a + b` *is* `a.wrapping_add(b)` unless something
actually overflows, so no test can kill these without an overflowing input.

The standing ruling, applied throughout: **statically-decidable equivalence is fixed at
generation time; input-dependent equivalence is deferred to the quarantine.** `div`/`rem`,
identical swapped arguments, and `*_ne_*` targets are all the first kind. The bare-operator
F1 direction is the second, and must be labelled an equivalence artifact rather than
reported as a test-suite blind spot.

---

## 5. Disjointness from cargo-mutants: what is proven and what is not

**Edit-disjointness is proven and tested.** `tests/test_disjointness.py` compares stored
cargo-mutants outcomes against regenerated transpilation mutants on `(file, span,
replacement)` — deliberately excluding `genre`, which would make the property true by
construction.

**Behavioural disjointness is not.** 82 `overflow-discipline` and 10 `shift-discipline`
records sit on binary expressions cargo-mutants already mutates under `BinaryOperator`:
same site, disjoint edit text, plausibly the same underlying fault. The defensible claim is
nearer **"~300 of 722 are novel in fault as well as in edit"** than "722 disjoint from
cargo-mutants."

---

## 6. What we are doing currently

### The 5-project sample — complete

722 mutants over the five hardest Opus-5 projects, 15.4 min, scored by
`analyse_opus5_sample.py _transpile_out`. Common mutant set n=356 (viable and non-timeout
in all four settings):

| config | kill rate | with the F1 equivalence mass removed (n=238) |
|---|---|---|
| `src` | 41.6% | **62.2%** |
| `src+test` | 13.8% | **20.2%** |
| `src/test` | 26.7% | **39.9%** |
| `c+rust` | 39.6% | **59.2%** |

Numerators are unchanged by the exclusion (148 / 49 / 95 / 141), which is itself the
evidence those 118 mutants are equivalent rather than missed.

These operators are markedly harder to kill than cargo-mutants', and the config ranking
**flips**: `src` edges out `c+rust`, and `src+test` collapses. Per family, the settings
where `c+rust` wins clearly are exactly those where the C source carries information the
Rust signature does not — argument order (68.6% vs 22.9% for `src/test`), endianness (88.9%
vs 11.1%), range bounds. The families where it does not win are the F1 rows contaminated by
equivalence.

### The full Claude set — running

Scaling to `results/claude/results_opus5_4settings`. Three steps, all landed in `839bf8e`:

1. **`audit_full_set.py`** — checks the preconditions rather than assuming them. 99
   projects in `_frozen_source`, 87 present in all four setting trees, 85 with test files
   and a green recorded baseline everywhere. Those 85 generate **15,014 mutants**.
2. **`build_full_set_workspaces.py`** — rebuilds them in the split-crate layout
   (`core` + one `tests-<setting>` crate). `results_opus5_4settings` predates the
   split-crates fix, so its tests live inside the mutated crate, which is exactly what made
   viability setting-dependent (see [`plan_shared_mutant_set.md`](./plan_shared_mutant_set.md)).
3. **`run_transpile_full.py`** — the sample runner pointed at that corpus, knobs unchanged
   (`WORKERS=8 JOBS=4 MIN_TIMEOUT=120 TIMEOUT_MULT=2.0`) so the two runs stay comparable.

**The split is where suites die**, and this is the main caveat on the full-set corpus:

| corpus | projects | mutants |
|---|---|---|
| four green baselines | **64** | **10,066** |
| 1–3 green | 20 | ~4,900 |
| all four red | 1 (`rbtree_lab`) | — |

Causes, all recorded per project with the compiler output in `full_set_readiness.json`:
tests using private items (`libqueue`'s `queue::QueueNode`); `#[path = "../foo.rs"]` include
hacks whose relative path breaks on the move (`bitset`, `merkle_tree_c`); unresolved names
(`rbtree_lab`); and five projects whose tests compiled but then failed. `c+rust` is hit
hardest — 11 of the 21 — so the attrition is **not random with respect to setting**, and
nothing cross-setting should be read off the partial projects.

The run covers the 64 four-green projects; `--include-partial` adds the rest as a second
phase. Wall-clock is driven by hangs rather than mutant count — the `utf8` validation spent
8 of its 8.6 minutes on 4 timeouts at the 120 s floor — so 6–12 hours is the honest bracket.
The 120 s floor was kept rather than trimmed: comparability with the sample numbers is worth
more than the hours.

### Open items

- **Witness pipeline** (plan 2): instrument `ground_truth_mutation.py`, whose line 53
  currently `unlink()`s every failing ground-truth test and so destroys the bug corpus,
  then the inverse-repair search.
- **Equivalence quarantine** (plan 3): the residual mass sits almost entirely in F1/F2, and
  the pre-registered hypothesis is specifically that `c+rust`'s advantage widens *on F1*.
  Unfiltered F1 numbers will therefore read as a test-quality finding when they are an
  equivalence artifact. The ground-truth suite needed to quarantine them exists for only
  ~5 of 100 projects.
