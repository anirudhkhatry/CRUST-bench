# Transpilation-aware mutation operators

**Date:** 2026-08-31
**Status:** design approved, not yet implemented

## Problem

Mutation testing in this project uses cargo-mutants, whose operators are language-agnostic
bug shapes: swap a binary operator, swap a unary operator, replace a function body with a
value, delete a match arm. Scored over the five hardest Opus-5 projects, those operators
produce a clean and useful result (`c+rust` 85.7%, `src/test` 74.5%, `src` 71.8%,
`src+test` 37.2%, strict, n=1101).

They say nothing about C-to-Rust translation specifically. A mutant that changes `+` to `-`
is a plausible bug in any language. The bugs that actually arise when a model translates C
into Rust are different in kind: an integer that wraps in C and panics in Rust, a `char`
whose signedness is platform-defined, a byte-order helper picked by name rather than by
semantics, an in-place update lost to a `.clone()` inserted to appease the borrow checker.

**Goal:** a catalog of mutation operators that model real C-to-Rust transpilation bugs,
disjoint from cargo-mutants by construction, with each operator's reality demonstrated
mechanically rather than argued.

## Scope

The artifact is the operator catalog plus an empirical study inside this repository. The
generator is written as a standalone module with a single clean interface so that
extracting a reusable tool later is packaging work rather than a rewrite, but no tool
packaging, CLI surface, or support for arbitrary crates is in scope now.

The catalog starts at the eleven operators in Section 1 and is designed to be extended one
operator at a time.

## Section 1: The operator catalog

Site counts come from the 101 frozen Opus-5 translations under
`results/claude/results_opus5_4settings/_frozen_source` (626 non-test `.rs` files);
`hits` is total occurrences, `proj` is the number of projects containing at least one.

### F1. Integer semantics

| operator | mutation | C/Rust divergence modelled | sites |
|---|---|---|---|
| `cast-width` | `as u32` -> `as u64` / `as u16` | C `int`/`long`/`size_t` mapped to the wrong Rust width | 2228 / 84 proj |
| `cast-sign` | `as i32` <-> `as u32` | C `char` signedness is platform-defined; sign decides whether `>>` is arithmetic or logical | (same sites) |
| `overflow-discipline` | `wrapping_add` <-> `+` <-> `saturating_add` <-> `checked_add().unwrap()` | C unsigned wraparound is defined behaviour; Rust `+` panics in debug | 319 / 51, 102 / 30 |
| `shift-discipline` | `<<` -> `wrapping_shl()` / `checked_shl().unwrap()` | C shift past the operand width is UB; Rust panics | 1083 / 83 |

`cast-width` has sites in 84 of 101 projects, making it the highest-yield operator in the
catalog. `overflow-discipline` reaching 51 projects is the more interesting number: over
half the translations are already reasoning explicitly about overflow, which means over
half of them can reason about it wrongly.

### F2. Data representation

| operator | mutation | C/Rust divergence modelled | sites |
|---|---|---|---|
| `endianness` | `to_be_bytes` <-> `to_le_bytes` <-> `to_ne_bytes`, and the `from_*` forms | C hand-rolls byte packing; Rust names it, so the name can be wrong | 123 / ~19 proj |
| `text-unit` | `chars()` <-> `bytes()`, `len()` <-> `chars().count()` | C `strlen` counts bytes; `chars()` is correct on ASCII and wrong on everything else | 131 / 30, 169 / 42 |
| `float-width` | `as f64` -> `as f32` | C `float`/`double` distinction collapsed in translation | 175 / 27 |

Narrower than F1 but higher precision. `endianness` mutants are close to guaranteed
semantically live, and they target clhash and ulidgen directly.

### F3. Ownership and interface

These have no analogue in any existing mutation tool, because they only exist as bugs when
a borrow checker is in the loop.

| operator | mutation | C/Rust divergence modelled | sites |
|---|---|---|---|
| `lost-in-place-update` | insert `.clone()` on a `&mut` argument or receiver | the standard escape from a borrow-check fight; the caller's update silently vanishes | 4917 `&mut` / 95 proj, 550 `.clone()` / 39 |
| `arg-transposition` | swap two adjacent arguments at a call site, or two fields in a struct literal | C signature order lost in translation | AST-only; sized during implementation |
| `error-path-erasure` | `?` -> `.unwrap_or_default()`; `unwrap_or(x)` -> `unwrap_or(Default)` | C's `-1`/`NULL` sentinels become `Result`/`Option` and the error branch gets swallowed | 447 / 34, 278 / 46 |
| `range-bound` | `a..b` <-> `a..=b`, `a..b` -> `a..b-1` | C pointer arithmetic becomes Rust slicing; off-by-one | 290 / 49, 116 / 36 |

### Dropped before implementation

- **NUL-termination / `CString` round-tripping.** Zero sites: no translation in the corpus
  uses `CStr` or `CString`. This is the operator first principles argued hardest for, and
  the corpus refutes it.
- **`overflowing_*` as a swap target.** Zero sites.
- **Enum discriminant perturbation.** 83 sites but only 10 projects. Held out of the core
  catalog; revisit if the witness gate proves generous.

### Disjointness

cargo-mutants' genres are `BinaryOperator`, `UnaryOperator`, `FnValue`, `MatchArm`,
`MatchArmGuard`. Each rewrites an operator token, a whole function body, or a match arm.
Every operator above instead rewrites a *method identity*, a *cast type*, an *argument
position*, or a *range kind*, or else *inserts* a call. The families cannot collide by
construction, and a test enforces it (Section 5).

## Section 2: Generator interface

New module `transpilation_operators.py`. It imports nothing from `run_mutation`,
`split_crates`, or any CRUST-bench path: it takes source text and returns mutant records.
That isolation keeps the module unit-testable against string literals and keeps a future
standalone tool cheap.

### Public interface

```python
def generate(text, rel_path, package, operators=OPERATORS):
    """Every transpilation-aware mutant for one Rust source file.

    Returns cargo-mutants-shaped records -- name, package, file, function, span,
    replacement, genre -- so everything downstream of list_mutants works unchanged.
    """
```

The record shape is fixed by `_MUTANT_KEYS` at `mutate_shared.py:52` and consumed by the
splice at `mutate_fast.py:270`. Matching it is what makes the generator a drop-in.

### The extension point

Each operator is one frozen dataclass in a registry:

```python
@dataclass(frozen=True)
class Operator:
    name: str                 # "endianness"
    family: str               # "F2-representation"
    query: str                # tree-sitter S-expression over the rust grammar
    rewrite: Callable         # (node, src) -> [(node, replacement, label), ...]
    inverse: Callable | None  # drives the witness gate (Section 3)

OPERATORS = [CAST_WIDTH, CAST_SIGN, OVERFLOW_DISCIPLINE, ...]
```

Adding an operator later is appending one object and one unit test. No change to the
generator, the harness, or the analysis. Removing an operator that fails the witness gate
is equally local.

### Why tree-sitter

`tree_sitter` is already a dependency of `run_mutation.py`, so there is no new toolchain.
It provides syntax but not types, so `arg-transposition` degrades from "swap two same-typed
arguments" to "swap two adjacent arguments" and some results will be type errors.

That is acceptable because the harness already contains a type checker: any mutant that
fails to compile is recorded `Unviable` and drops out of the denominator. The generator can
over-generate syntactically and let rustc prune. Precision therefore costs build time, not
correctness -- and at 16-19 s per mutant that is a real budget, so queries are tuned rather
than sprayed.

Rejected alternatives: a Rust helper using `syn` (exact grammar, but a new build artifact,
fiddly byte-offset spans via `span-locations`, and still no type inference without much
heavier machinery); regex or token scanning (handles all of F1 and F2 since those are
method-name and cast-token rewrites, but cannot express any of F3, and fires inside strings
and comments).

### Genre naming

Every record carries a `Transpile*` genre: `TranspileCastWidth`, `TranspileEndianness`,
`TranspileLostUpdate`, and so on. Three consequences: `analyse_opus5_sample.py` already
groups by genre, so the two families partition with no code change; disjointness from
cargo-mutants is visible in the data rather than asserted; and mixed runs stay attributable.

### Position encoding

tree-sitter reports byte offsets and byte-based `(row, column)`. `_offset` at
`mutate_shared.py:55` computes offsets from 1-based columns over Python `str` lines. These
agree on ASCII and diverge on anything else.

Rather than modify the validated splice path, the generator round-trip-verifies every
record it emits: it computes the intended edit by byte offset, checks that
`splice(text, span, replacement)` produces exactly that string, and drops the record if it
does not. A generator that cannot be silently wrong about where it edits is worth the ten
lines.

### Harness integration

`run_fast_mutation` gains one parameter, `mutant_lister=list_mutants`. The transpile path
passes a lister returning cargo mutants, transpile mutants, or both. The default is
byte-identical to the validated run, so nothing needs re-validating.

## Section 3: The witness pipeline

Every operator in the published catalog must be witnessed: a real translation that fails
its RBench ground-truth suite, and goes green when the operator's inverse is applied at a
single site. Operators that cannot be witnessed are dropped from the catalog.

### Step 0: stop destroying the corpus

`ground_truth_mutation.py:53` calls `f.unlink()` on every ground-truth test that fails
against a translation, and `rm._cargo_test_bin` returns only a boolean, so a test that
fails to compile and a test that compiles and asserts wrong are treated identically and
both discarded. The gpt-5.5 run recorded `n_valid = 5` of 100 projects. The corpus this
gate depends on is large, already produced, and currently thrown away.

Instrument the prune step to record, per (project, model, test binary), one of three
outcomes:

- compiled and passed
- **compiled and failed** -- the witness candidates
- failed to compile

Only the middle case feeds the gate. A ground-truth test that will not compile against a
translation is a real transpilation failure, but an interface mismatch that no value-level
mutation operator models; including it would inflate the corpus with bugs the catalog
cannot claim.

This step also produces the map of which projects have a usable ground-truth suite at all,
which Section 4 needs.

### Steps 1-4: the loop

1. **Detect sites.** Run each operator's tree-sitter query as a detector over each failing
   translation, finding real occurrences of the wrong side.
2. **Inverse edit.** Apply the operator's `inverse` at a single site.
3. **Re-run the ground-truth suite.** Accept only if the previously-failing binaries go
   green *and* no previously-passing binary regresses. Requiring the full suite green stops
   an edit that masks one symptom while breaking another from counting as a repair.
4. **Accept the operator** if it is witnessed in **at least 2 distinct projects**. One
   witness can be an idiosyncrasy of a single translation. The per-operator witness count
   is reported, and doubles as the sampling weight in Section 4.

### Bounding the search

`cast-width` alone has 2228 sites, so exhaustive single-site search is not free. Three cuts,
in order of value:

- restrict to files reachable from the failing test, using the existing `llvmcov_check.py`
  so site selection is coverage-guided rather than guesswork;
- single-site edits only;
- a per-project candidate cap.

Cost per candidate is one build plus one test binary for a single crate, far cheaper than a
mutation run, which pays for `core` plus 11 dependent binaries. `mutate_fast`'s clone and
worker-pool machinery transfers directly.

### Stated limitations

- Single-site search cannot witness a bug requiring two coordinated edits, so the gate
  under-counts.
- A repair is evidence, not proof: an inverse edit could coincidentally satisfy the
  assertions. A sample of accepted witnesses is hand-checked rather than trusting the loop
  wholesale.
- Zero witnesses does not refute an operator. It is reported as *proposed but unwitnessed
  in this corpus*, the same honest negative already obtained for `CString`.

## Section 4: Study protocol

### Scope, in two stages

Pilot on ulidgen, clog, emlang, clhash, and cset. There are already 1101 cargo-mutants
verdicts on those five across four settings, so the new operators land beside a directly
comparable baseline on identical translations and identical test suites. Scale to a wider
sample only if the pilot shows the operators discriminate.

### Sampling

The raw generated set is unbalanced by roughly two orders of magnitude -- `cast-width`
would swamp every other operator and make the headline number a report on casts. Therefore:
cap each operator at **30 mutants per project**, sampled with a fixed seed (`seed=0`), and
record both the cap and the seed in `outcomes.json` so results are reproducible. 30 keeps
the pilot's total near the 1101 of the cargo-mutants baseline, so the two runs cost roughly
the same.

The primary reported figure is the **per-operator kill rate per setting**, which is
cap-independent and is what the catalog actually claims. A witness-count-weighted aggregate
is reported as a secondary figure only, and always alongside the unweighted one.

### Run configuration

Unchanged from the validated run, deliberately: `mutate_fast`, one shared build so all
settings score the identical mutant set, 8 workers x 4 jobs, a 120 s timeout floor, and
both strict and folded scores. Only the mutant source changes.

### Equivalence quarantine

Every candidate transpile mutant is first run against the ground-truth suite, the strongest
oracle available. A mutant that survives ground truth is quarantined as
*candidate-equivalent* and excluded from the denominator when scoring generated tests,
rather than counted as a survival.

This matters more here than for cargo-mutants' operators: `wrapping_add` and `+` agree on
every input that never overflows, so an untriggered one is unkillable and would otherwise
read as a test-quality finding.

Where a project has no valid ground-truth suite -- and Step 0 will say how many that is --
the quarantine is unavailable, and those numbers are reported unfiltered and explicitly
labelled. A partial oracle honestly scoped is better than a uniform one that quietly is not
there.

### Analysis

No new analysis code. `analyse_opus5_sample.py` groups by genre, so the `Transpile*` prefix
partitions the two families automatically, and its common-mutant-set logic already produces
comparable denominators across settings.

**Pre-registered hypothesis**, recorded before the run: `c+rust`'s advantage over the other
three settings widens on family F1 relative to generic operators, because integer width,
signedness, and overflow discipline are facts visible only in the C source. If it does not
hold, that is a finding about the settings rather than a defect in the catalog. Writing it
down in advance is what stops the analysis becoming a search for a flattering slice.

## Section 5: Testing

Four levels, cheapest first:

1. **Per-operator unit tests** over string fixtures. No corpus, no cargo, no build.
2. **Round-trip property test:** every emitted record, spliced via
   `mutate_shared.splice`, reproduces the byte-offset edit the generator intended.
3. **Disjointness test:** load the 1101 stored cargo-mutants records from
   `results/claude/opus5_sample/_mutants_out/*/*/mutants.out/outcomes.json` and assert no
   generated record shares a `(file, span, replacement)` triple with any of them.
4. **End-to-end smoke:** one `ulidgen` shard through build, verdict, and `outcomes.json`,
   confirming the records survive the full harness unchanged.

## Risks

| risk | mitigation |
|---|---|
| Over-generation inflates `unviable` and wastes 16-19 s per mutant | Tune queries; cap per operator per project; `unviable` is excluded from the denominator so correctness is unaffected |
| Equivalent mutants read as test-quality findings | Ground-truth quarantine (Section 4), explicitly scoped to projects where a GT suite exists |
| Witness gate finds nothing for most operators | Acceptable and reportable; the catalog shrinks to what the corpus supports, which is the point of deriving it this way |
| Ground-truth suites too scarce to serve as an oracle | Measured in Step 0 before the study depends on it; unfiltered numbers labelled as such |
| Byte/char position mismatch corrupts edits silently | Round-trip verification in the generator, which drops any record it cannot reproduce |

## Out of scope

- Packaging a standalone tool, a CLI, or support for arbitrary Rust crates.
- Changes to the validated `mutate_fast` / `mutate_shared` execution path beyond the single
  `mutant_lister` parameter.
- Rewriting `docs/mutant_distribution_and_memory_types.md`, which carries stale numbers from
  superseded runs and is tracked separately.
- Multi-site witness search.
