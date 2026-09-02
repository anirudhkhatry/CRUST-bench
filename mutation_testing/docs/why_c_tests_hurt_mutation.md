# Why translating the C tests hurts mutation score

**Question:** across the four test-generation settings, why does `src+test` (C source
+ **C tests** + Rust source) produce Rust tests that kill *fewer* mutants than
`c+rust` (C source + Rust source) and `src/test` (Rust source + interfaces)?

**Short answer:** when we hand the model the C test suite to translate, it reproduces
that suite's **assertion strength *and* its coverage scope**. C test suites are usually
scenario/integration-level with coarse assertions; the translated Rust tests inherit
both limitations. `c+rust` and `src/test`, generating tests *from the Rust
implementation*, instead write per-function unit tests with exact-value assertions and
boundary cases — which kill far more mutants.

---

## 1. The numbers

Mutation score per setting (fraction of viable mutants killed):

| setting | inputs | gpt-5.5 | Opus-5 |
|---|---|---|---|
| `src/test` | Rust source + interfaces | 90.3% | 95.3% |
| `c+rust` | C source + Rust source | 89.7% | 95.7% |
| `src` | C source + interfaces | 88.5% | 91.9% |
| **`src+test`** | **C source + C tests + Rust source** | **77.4%** | **84.4%** |

`src+test` is the worst by a wide margin **for both models**, despite receiving the
*most* information (it is the only setting given the actual C tests) and keeping *more*
test files on average. So it is not an information or effort problem — it is a
**test-quality** problem.

### The differential

Over the gpt-5.5 run, we took, per project, the mutants that `src+test` **missed**
(survived) but `c+rust` **caught** — **299 mutants** that `c+rust` kills and `src+test`
does not. Top projects by differential:

```
 c+rust  src/test  project
     27       29    cJSON
     23       22    proj_2DPartInt
     20       18    emlang
     18       15    worsp
     18       11    approxidate
     17        9    libm17
     16       13    mvptree
```

Category breakdown of those 299 differential kills:

```
  14  delete match arm          <- unexercised branch
  13  ==  -> !=                  <- comparison / boundary
  13  +   -> -                   <- arithmetic
  12  <   -> >                   <- comparison / boundary
  12  -   -> +
  12  -   -> /
  12  +   -> *
  11  <   -> ==                  <- boundary
  11  *   -> +
  10  +=  -> *=
  10  <   -> <=                  <- off-by-one boundary
   9  >   -> <
   9  *   -> /
   9  delete !                   <- unexercised negation/branch
```

The differential is dominated by **return-value replacements** (a function's body
replaced by a default — `Default::default()`, `Ok(())`, `0`, `""`, …), **comparison/
boundary operators** (`==↔!=`, `<↔>`, `<↔<=`, `<↔==`), and **deleted branches**
(`delete match`, `delete !`), plus arithmetic swaps. Killing these requires tests that
(a) exercise both sides of each branch and hit boundary values, (b) assert a function's
concrete return value, and (c) pin exact numeric output. That is exactly what `src+test`
lacks.

### It replicates on Opus-5

Repeating the same differential on the Opus-5 run:

```
gpt-5.5 : 299 differential kills | return-value=40, delete-match=14, ==->!=13, +->-13, -->+12, <->>12 ...
opus-5  : 248 differential kills | return-value=33, +->*19,  ==->!=17, +->-12, /->%12, >=-><=11 ...
```

Same category profile (return-value replacements + operator/comparison/boundary swaps)
for both models. The *top projects* differ (gpt: cJSON, 2DPartInt, emlang, worsp,
approxidate; opus: clog, ulidgen, clhash, Remimu) because each model transpiles a
slightly different frozen source — but the **mechanism and the shape of what `src+test`
misses are identical**. So this is a property of the *modality*, not of one model.

---

## 2. Two mechanisms (with concrete examples)

### Mechanism A — weak oracles (relational vs. exact-value)

**`murmurhash_c`** — `src+test` caught 6/13, `c+rust` caught 7/7.

`c+rust` built an **independent reference oracle**: it re-implemented murmurhash in the
test and asserted exact equality over many inputs:

```rust
// c+rust
fn c_reference_murmurhash(key: &[u8], seed: u32) -> u32 { /* full algorithm */ }
// ... assert murmurhash(input, seed) == c_reference_murmurhash(input, seed)
```

`src+test` (mirroring the C tests) used mostly **relational / smoke assertions**:

```rust
// src+test
assert_ne!(murmurhash(b"0", 0), murmurhash(b"1", 0));   // "different inputs differ"
for &(input, seed) in inputs {                          // determinism only
    assert_eq!(murmurhash(input, seed), murmurhash(input, seed));
}
assert_eq!(murmurhash(b"", 0), 0x0000_0000);            // one golden value
```

**Smoking gun:** the mutant `murmurhash.rs:73 replace ^= with |=` is in `src+test`'s
`missed.txt` **and** `c+rust`'s `caught.txt` — same mutant, same source. A mutated hash
is still deterministic and still input-sensitive, so it passes the `assert_ne!` /
determinism checks but dies instantly against an exact-value oracle. Every mutant
`src+test` misses here is a value-changing-but-property-preserving edit (`* → /`,
`>> → <<`, `^= → &=`, `!= → ==`).

### Mechanism B — narrow coverage (inherited C-test scope)

Here `src+test` *does* use exact `assert_eq!` — but it only tests the functions the C
suite happened to test, missing internal helpers and boundaries.

**`proj_2DPartInt`** — `src+test` caught 35/85, `c+rust` 50/64. All the differential
mutants are in `collisions.rs` (`compute_contacts`, `find_col/row/square`):

```
proj_2DPartInt/src/collisions.rs:100:4:  delete ! in compute_contacts
proj_2DPartInt/src/collisions.rs:103:12: replace > with >= in compute_contacts
proj_2DPartInt/src/collisions.rs:110:25: replace - with + in compute_contacts
```

`src+test` mirrors the C test: it drives the top-level **force computation** with a
fixed set of pre-supplied contacts and asserts exact forces (`assert_close`). It never
unit-tests the collision-detection helpers, so mutations inside them survive.

`c+rust` additionally wrote **boundary unit tests for those helpers**:

```rust
// c+rust
assert_eq!(find_col(-2.1, 4, 1.0), -2);
assert_eq!(find_col(-2.0, 4, 1.0),  0);
assert_eq!(find_col(-1.0001, 4, 1.0), 0);
assert_eq!(find_col( 2.0, 4, 1.0), -1);      // hits the boundary both sides
// + assert_contact(...) exercising compute_contacts directly
```

Those boundary cases pin the `>`/`>=`/`<` comparisons and the arithmetic in
`compute_contacts`, killing the mutants `src+test` leaves alive.

**`approxidate`** — `src+test` caught 43/73, `c+rust` 64/64. The differential mutants
are all in the internal date helper `civil_from_days`:

```
approxidate/src/approxidate.rs:227:16: replace >= with < in civil_from_days
approxidate/src/approxidate.rs:229:22: replace /  with * in civil_from_days
```

`src+test` tests only the public `approxidate(date_str, &mut tv)` on a handful of dates
(the C test's cases); for those particular inputs the mutated `civil_from_days` arithmetic
still yields the same top-level result, so the mutants survive. `c+rust`, seeing the
implementation, tests more of the surface (constants, timezone tables, and a wider range
of values) that constrains `civil_from_days` directly.

### Mechanism B, continued — testing the wrong side of the module

**`cJSON`** — differential 27 (the largest for gpt-5.5). Every differential mutant is in
the **parser** (`parse_number`, `parse_string`, `parse_unicode_escape`):

```
src/cjson.rs:208: replace == with != in parse_number
src/cjson.rs:218: replace match guard number.is_finite() with true in parse_number
src/cjson.rs:248: delete match arm '\\' in parse_string      <- escape handling
src/cjson.rs:250: delete match arm 'b'  in parse_string
src/cjson.rs:416: delete ! in parse_unicode_escape
```

`src+test` (mirroring the C tests) exercises the **builder / renderer** path — it
constructs objects and checks the rendered string:

```rust
// src+test
assert!(root.add_item_to_object("root_node1", node1).is_ok());
assert!(rendered.contains("\"root_node1\""));   // substring, not exact
```

It barely touches the parser, and never the escape/number edge cases — so the parser
mutants survive. `c+rust` unit-tests `parse` exhaustively with exact expected values and
error variants:

```rust
// c+rust
assert_eq!(parse(" null ", true).unwrap(), CJson::Null);
assert_eq!(parse("0", true).unwrap(), CJson::Number(0.0));
assert!(matches!(parse(<bad literal>, true), Err(InvalidLiteral(..))));
```

exercising the escape arms, the `is_finite` guard, and the number branch — killing them.
Two lessons in one: `src+test` used **weaker assertions** (`.is_ok()`, `contains(...)`)
*and* **tested the wrong side of the module** (builder, not parser).

**`emlang`** — differential 20. `src+test` asserts only **coarse structural
invariants**:

```rust
// src+test
assert!(prog.size > 0);                 // "parsed at least one instruction"
assert!(prog.cap >= prog.size);         // "capacity covers size"
assert_eq!(parser.input, source);
```

These hold under most mutations. `c+rust` asserts the **exact instruction stream**:

```rust
// c+rust
fn assert_push_int_inst(inst: &Em, expected: i64) {
    assert_eq!(inst.em_type, EmType::Push);
    assert_data_int(&inst.data, expected);
}
```

so mutations in the parser's cross-reference/advance logic (`!= → ==`, `delete !`,
`== → !=`) and `load_file`'s return value flip a concrete instruction and get caught,
whereas `src+test`'s "size > 0" survives them.

### Replication example on Opus-5 — `clog`

Even for the stronger model the pattern is identical. `clog` (Opus-5, differential 45 —
its largest) — differential mutants are in format-string validation and the format
interpreter:

```
src/clog.rs:182: replace >= with < in clog_set_time_fmt      <- boundary check
src/clog.rs:204: replace clog_set_date_fmt -> Result<(),()> with Ok(())  <- return value
src/clog.rs:340: delete match arm '%' in clog_format         <- format directive branch
src/clog.rs:412: replace - with / in tm_from_epoch
```

`src+test` checks that logging succeeds and matches **one** format:

```rust
// src+test
assert!(clog_init_path(0, path).is_ok());
assert_eq!(contents, format!("{}: DEBUG: Hello, world!\n", this_file()));
```

`c+rust` enumerates **many format directives and levels** with exact expected output:

```rust
// c+rust
assert!(clog_set_fmt(id, "%l\n").is_ok());
assert_eq!(contents(&path), "DEBUG\nINFO\nWARN\nERROR\n");
```

covering the `%`-directive match arms and the format-validation boundaries `src+test`
never reaches. Same modality effect, different model.

---

## 3. Why `c+rust` and `src/test` are better

Both are generated **from the Rust implementation** rather than from the C tests. In
practice that leads the model to:

1. **Write exact-value oracles.** It sees the function's return type and computation and
   asserts specific expected values (often reconstructing a reference), instead of the
   coarse "output differs / is deterministic / round-trips" assertions typical of C
   suites.
2. **Unit-test each function, including internal helpers.** It enumerates the functions
   in the source and writes targeted tests for them (`find_col`, `civil_from_days`, …),
   whereas `src+test` only covers whatever the C suite drove at the top level.
3. **Cover branches and boundaries.** Seeing conditionals/`match` arms, it picks inputs
   on both sides of each comparison and at boundary values — which is precisely what
   kills the comparison/branch mutants that dominate the differential.

`src/test` (Rust source + interfaces, no C at all) and `c+rust` (C source + Rust source)
score nearly identically, which reinforces the point: the win comes from **generating
against the implementation**, not from having the C source. Adding the C *tests*
(`src+test`) actively anchors the model to a weaker template.

---

## 3b. Decomposing the gap: coverage vs. oracle (is it *only* weak assertions?)

No — weak (relational) oracles are the **minority**. Classifying every differential
mutant (src+test missed ∩ c+rust caught) by whether src+test's tests even *reference*
the mutated function:

| | gpt-5.5 (299) | Opus-5 (248) |
|---|---|---|
| **Coverage gap** — mutated fn never referenced by src+test tests | **226 (76%)** | **147 (59%)** |
| **Oracle weakness** — fn is tested, mutant still survives | 60 (20%) | 91 (37%) |
| unclassified | 13 (4%) | 10 (4%) |

So the ~11–12% "with vs. without C tests" gap is **primarily a coverage deficit**:
`src+test` reproduces the C suite's *scope* (the public/top-level behaviors the C tests
drove) and never unit-tests the internal helpers `c+rust` targets. Weak oracles
(Mechanism A, murmurhash) are real but secondary. The coverage-gap bucket is dominated by
**operator mutations** (165/226 gpt; 116/147 opus) — arithmetic/comparison ops inside
functions the C-derived tests never call.

The split shifts by model: Opus has *fewer* coverage gaps (59% vs 76%) and *more* oracle
weakness (37% vs 20%), consistent with it writing 3–4× more tests (more functions touched
→ smaller coverage gap, so its residual misses lean more on assertion strength).

*Method caveat:* "referenced" = the function name appears in the test source — a proxy.
It can under-count coverage gaps (transitively-reached functions) and over-count coverage
(referencing a function ≠ executing the mutated branch). The line-accurate version using
`cargo llvm-cov` is in §3c.

## 3c. Line-accurate decomposition (cargo llvm-cov)

The §3b proxy asked "is the function *named* in the tests." The exact version runs
`cargo llvm-cov` on src+test's green suite and asks "was the mutated **line** actually
*executed*." This corrects the proxy substantially:

| | name-proxy (§3b) | **llvm-cov (exact)** |
|---|---|---|
| **gpt-5.5** (differential 299) | | |
| &nbsp;&nbsp;Coverage gap (line never executed) | 76% | **47% (140)** |
| &nbsp;&nbsp;Oracle weakness (line executed, survived) | 20% | **53% (159)** |
| **Opus-5** (differential 248) | | |
| &nbsp;&nbsp;Coverage gap (line never executed) | 59% | **40% (98)** |
| &nbsp;&nbsp;Oracle weakness (line executed, survived) | 37% | **60% (150)** |

Both models land near **~50/50**, oracle-leaning (gpt 53%, Opus 60%). Opus is *more*
oracle-dominated — consistent with it writing 3–4× more tests (fewer unreached lines, so
its residual misses are relatively more about assertion/input strength than pure reach).

The proxy **over-counted coverage gaps**: many helper functions it flagged as
"unreferenced" are in fact reached **transitively** (a public entry point calls them), so
the line does run — the test simply fails to detect the change. Corrected, the gap is
**~50/50**:

- **~47% pure coverage gap** — src+test never drives that line at all (e.g. cJSON's
  escape-handling match arms: `coverage_gap=16`). Fixing needs tests that reach the code.
- **~53% executed-but-undetected** — the line runs but the mutation isn't caught. This
  bucket is itself a mix:
  - *weak oracle* — the assertion is too loose to see the changed value (murmurhash's
    `assert_ne!`/determinism);
  - *weak input/boundary coverage* — the line runs, but not with inputs that make the
    mutation change the observable output (**approxidate: `coverage_gap=0, oracle_weak=18`**
    — every `civil_from_days` arithmetic mutant's line executes under the tested dates,
    but those dates don't distinguish the mutation);
  - *non-propagation* — the mutated value is overwritten/masked before any assertion.

So "is it only relational tests?" — **no.** Relational/weak assertions are one part of the
53% executed-but-undetected half; the other half is pure unreached code. Both a **coverage
problem** (reach the internal helpers and their branches) and an **oracle problem** (assert
exact values on distinguishing inputs) must be fixed to close the gap. `c+rust`/`src/test`
happen to do both because they generate per-function tests with exact assertions.

*(Per-project splits in `llvmcov_decomposition_<model>.json`.)*

## 4. Takeaway

- **Translating an existing (C) test suite caps the generated tests' fault-detection at
  that suite's oracle strength and coverage scope.** For typical C unit/integration
  suites — coarse assertions, top-level scenarios, few boundary cases — that ceiling is
  low, which is why `src+test` trails by ~13 pts (gpt-5.5) / ~11 pts (Opus-5).
- **Generating tests directly from the implementation** (`c+rust`, `src/test`) yields
  tighter assertions, per-function unit tests, and branch/boundary coverage, killing far
  more mutants.
- **Practical recommendation:** if the goal is high-quality Rust tests, generate them
  from the (translated) Rust source rather than porting the C tests. The C tests can
  still seed *behavioral intent*, but should not be the template the tests are built on.

---

*Data: `results_gpt55_4settings/` and `results_opus5_4settings/` (per-project
`mutants.out/{caught,missed}.txt`, generated tests under `<setting>/<project>/src/bin/`).
Differential computed as `src+test`.missed ∩ `c+rust`.caught per project.*
