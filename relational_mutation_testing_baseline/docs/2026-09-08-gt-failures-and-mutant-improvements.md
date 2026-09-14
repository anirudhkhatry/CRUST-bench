# Where T′ says green and the ground truth says red

**Date:** 2026-09-08 · covers the 97-project `mutation_tests` codex arm

The `mutation_tests` setting hands a transpilation model the C sources, the Rust
interfaces, and the sound relational mutation tests (`rust/tests/rmt_*.rs`) — but not the
RBench ground-truth suite. The model works until `cargo test` on the mutation tests is
clean, and that is the run's exit condition.

It reached it everywhere: **853/853 mutation tests pass, in all 97 projects**. Graded
against the held-back RBench suite the same crates score **72/97 (74%)**.

So there is no project where T′ and GT agree on failure. Every one of the 32 ground-truth
failures across 25 projects is a disconnect — T′ green, GT red — and the useful question is
which kind. This note answers that, and then says what to change in the pipeline.

---

## 1. Method, and a correction to the first pass

`src/codex_expts/analyse_gt_failures.py` classifies each failure:

| class | meaning |
|---|---|
| `COVERED` | GT fails in a function a mutation test also exercised — T′ was satisfiable by a wrong implementation |
| `UNCOVERED` | GT fails in a function no mutation test touched — a coverage gap |
| `NO_ASSERT` | not an assertion at all: build error, timeout, or an empty GT suite |

The first version of the script computed the overlap from the API names appearing anywhere
in the failing *file*, so a binary that tests two independent modules looked "covered"
whenever a mutant touched either one. It now splits cargo's output on
`---- <test> stdout ----`, finds that single `#[test] fn` in the RBench source, and
intersects only the calls in its body. That moved `libutf`, `blt`, `emlang` and `clog` from
COVERED to UNCOVERED — libutf's failure is `test_string_substring` in `libutf_string`,
while all ten of its mutants sit in `libutf_utf`.

`gt_test_report.py` keeps only the last 3000 characters of output, which pushes the panic
off the top of any suite that prints as it runs, so `reprobe_gt_failure.py` reran 21 of the
25 projects under the identical procedure and kept the whole log in `/tmp/gt_full/`.

**32 failures, 25 projects: COVERED 17, UNCOVERED 11, NO_ASSERT 4.**

The COVERED majority is the finding, and it is the unwelcome one. The dominant failure mode
is not "the mutation tests never reached that code" but "they reached it and were too weak
to pin it down".

---

## 2. The 17 COVERED failures

| project / test | what the mutation test demanded | what GT demanded that it did not |
|---|---|---|
| **kd3** `test_kd` | `search(&mut iter, …)` on 3–4 points. `m06`/`m10` were seeded in `search_space`/`point_in_search_space`, but the test says "UNTRANSLATABLE: `search_space` takes no iterator and returns nothing" and queries the same box through `search` | `search_space()` itself, then reads the iterator. The impl builds a **local** `KDTreeIterator` inside `search_space` and drops it (`src/kdtree.rs:153`) |
| **dict** `test2` | `dict_get` on dicts from `dict_new(DictType::I32, …)`; the value payload is sized 8 vs 64 bytes | `dict_create_args` with `DictType::Struct`, `size_of::<StrT>()`, `copy/free/hash/cmpr = None` — string keys. `dict_get` returns `None`. Neither the struct-key hash/compare path nor the `_args` constructor was touched |
| **mvptree** `test` | `write(&fname, 0o600)` for a 2-point tree, then re-read and check only the header (branch factor, path length, leaf cap, datatype) | `write(testfile, 00755)` — decimal 755 = `0o1363`, no owner read bit — then `mvptree_read(…).unwrap()`. The impl passes the mode through literally and cannot reopen its own file → `FileNotFound`. T′ picked a readable mode by luck and never pinned mode semantics, nor the datapoints in the round trip |
| **libpsbt** `read_test_vector` | `m09`: `psbt_decode` of a 9-byte payload into a roomy buffer. `m06`: `psbt_read` with `None` as the element handler | Walks records with a callback and asserts `record.record_type == 0`; got 4. Record-type semantics were never inspected, because T′ always passed `None` |
| **libpsbt** `encode_decode_test`, `empty_input_test` | `m06` gave `psbt_init` a 256-byte scratch buffer for a 94-byte PSBT — 2.7× headroom | `psbt_init(…, psbt_len)` — sized exactly to the payload. `psbt_read` returns `OobWrite`. Exact-fit capacity accounting was never constrained |
| **libvcd** `test_ram_vcd` | Synthetic one-line dumps in a temp file: `$var wire 1 ! clk $end`, single-bit signals, no header sections | A real Icarus dump, `src/bin/assets/ram.vcd`: `$var reg 4 " address [3:0] $end` — multi-bit signals with a trailing bit-range token, plus `$date`/`$version`/`$timescale`/`$scope`. The impl rejects it: "invalid `$var` instruction" |
| **libwecan** `run_tests` | `decode_float` in **INTEL** byte order only, 8-bit fields, integral factors (2.0/4.0/8.0), offset 0/1 | `encode_float` → `decode_float` round trip in **MOTOROLA** order, factor 1e-7, multi-byte fields. Big-endian bit numbering, the encode direction and fractional scaling all unconstrained |
| **roaring_bitmap** `test_contains`, `test_invert`, `test_fill_ascending` | `m05`–`m08` all work on 3–4 element sets — entirely inside the sorted-array container (`length() == 8`) | Inserts all 65 536 values, driving array → bitset → inverted-array conversion at `LOW_CUTOFF`/`HIGH_CUTOFF`. The impl indexes `buffer[..4098]` on a 4097-length buffer. No mutation test ever made a container transition happen |
| **libbase122** `test_encode_all_ones` | Round trip of two hand-picked inputs (8×`0xFF`, `{0x15,0xAB}`), plus "decode must return `Err` when trailing 1-bits exceed capacity" | A length sweep `i in 0..=65` over `vec![0xFF; i]`. At i = 32 decode returns `Ok` with **0** bytes. Two points and one refusal rule are not a round-trip property |
| **Math_Library_in_C** `castom_test_exp` | `castom_exp` at 1.0, −1.0, −2.0, −0.5 to **1e-9 absolute** — ~1e-9 relative at magnitude 1 | `for j in −20.0..=80.0 step 0.1` against `f64::exp` at **1e-6 absolute**. At j = 21.5 that is ~9e-16 relative (≈4 ULP on 2.17e9); the impl is off by 1.9e-6. The series' `EPS_10` stopping rule is relative-error-blind and was never probed at magnitude |
| **Math_Library_in_C** `scastom_test_pow` | `castom_pow` with **integral** exponents only: (−2)³, (−3)², (−2)⁻², 2¹⁰ | 250 pairs with **fractional** exponents (`j = 0.4 … 9.6`), forcing the exp/log path, at 1e-6 absolute up to ~1e6. Off by 1.3e-6 |
| **Simple_Config** `test` | `cfg_parse("alpha: 1\nbeta: 2\ngamma: 3\n")`, and — explicitly weakened — `assert!(cfg.capacity >= cfg.entries.len())`, with a comment that the C capacity-injection contract is "UNTRANSLATABLE" | Full struct equality: `Cfg { entries: [], count: 0, capacity: 10 }` against the impl's `capacity: 0`. GT pins `capacity == TEST_CAPACITY` exactly — the one field T′ declined to constrain |
| **Genetic…control** `testMutx` | `m03` builds populations via `createInputPop` + `createStructure` with explicit 4-wide max/min arrays, then calls `mutx` | `createPopulation(&mut population, &[2, 9])` first. It panics in `src/population.rs:29`, "range end index 9 out of range for slice of length 3" — the constructor GT uses was never called by any mutation test |
| **razz_simulation** `test_card_operations` | `m10` asserts deck **membership** before/after `strip_card_from_deck`; `m04` asserts hand capacity with hand-made cards. All order-independent | `create_shuffled_deck()` + `deal_from_deck()` with an expected sequence (`get_max_rank_of_hand() == R9` after the first deal; got `Ace`). GT pins the shuffle order — the deck as a *sequence*, where T′ only ever treated it as a *set* |

Two caveats on the label. `COVERED` is at test granularity, so for `Genetic…/testMutx` (panic
in `createPopulation`) and the `roaring_bitmap` trio (panic in libcore, reached through the
container conversions) the fault site sits outside the overlapping function: covered by
call, uncovered at the fault.

### The recurring shapes

Five patterns account for all 17, and none of them is "the mutation test missed the code":

1. **Explicit weakening where the C oracle did not translate** (kd3, Simple_Config,
   libpsbt/`m06`, dict, mvptree). The tests carry `UNTRANSLATABLE:`/`NOTE:` comments where a
   C fact — a caller-supplied buffer, a capacity, an allocator hook, a pointer identity —
   had no Rust counterpart, and substitute something weaker or reroute through a different
   entry point. GT then checks exactly the dropped fact.
2. **One witness instead of a property** (libbase122, Math ×2).
3. **Over-provisioned buffers** (libpsbt): T′ always allocated slack, GT sizes exactly.
4. **One configuration of a multi-configuration API** (libwecan INTEL-only, dict I32-only,
   libvcd synthetic single-bit input).
5. **Set-like assertions on order-dependent behaviour** (razz_simulation) and small-N
   assertions on size-dependent representations (roaring_bitmap).

The mutation tests do kill their mutants at the seeded fault. But a single-witness kill
leaves the surrounding contract — capacity, byte order, argument range, container regime,
output plumbing — unconstrained, and where the C oracle did not survive translation, T′ is
weaker than the C test it came from.

### The other 15

**UNCOVERED (11):** `Genetic…control/testCrossov`, `blt`, `clog`, `emlang`, `fslib`,
`lib2bit`, `libpgn`, `libutf`, `roaring_bitmap/test_bitset_to_inverted_array`, `skp`,
`tisp`. These are placement failures — see §3.4.

**NO_ASSERT (4):** `bostree` (timeout), `c_blind_rsa_signatures` (build error: cannot
construct `BRSASecretKey` with a struct literal), `hamta` (build error: `Box<String>:
AsRef<str>` unsatisfied), `libfor` (the GT suite declares no `#[test]` functions — unmeasured
rather than failed).

Three file-touching failures were checked and are genuine implementation bugs, not harness
artifacts: emlang's `resources/tests/comments.eml` is present in the graded copy, libvcd's
`src/bin/assets/ram.vcd` arrives with the pasted `src/bin`, and mvptree's `testfile.mvp` is
written by the test itself.

---

## 3. What to change

### 3.1 Stop shipping tests that never killed anything — free, do this first

`prepare_mutation_tests.py:85` accepts a test when
`stage3.compiles and stage4.on_m == "green"`. It never checks whether the test went **red
on M′**, which stage 4 already computes and `mutation_tests.json` records as `outcome`:

```
853 shipped tests:  KILLED 791 | F4_not_discriminating 22 | F5_no_rust_counterpart 39 | F5_error 1
```

62 tests (7%) demonstrably discriminate nothing on the Rust side, and they land where it
hurts:

- **kd3 `m06` — `F4_not_discriminating`.** The `search_space` reroute. The pipeline observed
  that it passes M *and* M′, and shipped it anyway.
- **Simple_Config `m03` — `F5_no_rust_counterpart`**, as are its other eight. That project's
  entire T′ suite has zero demonstrated power.
- **Genetic…control `m03` — `F4_not_discriminating`.**

The fix:

```python
if not stage3.get("compiles") or stage4.get("outcome") != "KILLED":
    continue
```

791/853 tests survive; 33 projects lose at least one; 4 (`Simple_Config`, `ljmm`,
`rbtree_lab`, `worsp`) drop to zero and leave the setting. No LLM calls, and it makes
"T′ green" mean something for the first time.

### 3.2 Close the `UNTRANSLATABLE` leak (`rmt/prompts/transpile.md`)

Rule 4 lets the transpiler drop what it calls scaffolding. In five of the 17 the dropped
thing *was* the discriminating fact, relabelled. Three additions, all checkable in a new
stage 3.5 with no LLM:

- **Entry-point fidelity.** The Rust test must call the counterpart of `reachable_via`.
  kd3's never calls `search_space`. If a fault is unobservable through its own entry point,
  that is a finding about the API, not a licence to test a different function.
- **Budget the markers.** Count `// UNTRANSLATABLE:` lines; more than one or two, or any
  between the mutant's setup and its assertion → retry, then drop the mutant as
  untransferable.
- **Ban silent loosening.** Add to rule 1: an assertion may not be relaxed from `==` to
  `>=`/`is_ok`/`is_some`. Simple_Config's `assert!(cfg.capacity >= cfg.entries.len())` is
  exactly the failure.

### 3.3 One witness → a small family (`rmt/prompts/testgen.md`)

Rule 2 asks for concrete values at one point, which is the minimum needed to kill a mutant
and therefore the maximum the model produces. The widenings that would have caught the rest
are derivable from the signature, so they can be stated as hard requirements:

| rule to add | catches |
|---|---|
| Every enum/mode parameter exercised in **every variant** | libwecan (INTEL only), dict (`I32` only) |
| Every size/capacity parameter at **0, 1, exact-fit, cutoff−1, cutoff, cutoff+1** | libpsbt (always over-provisioned), roaring_bitmap (never crosses a container cutoff) |
| Where the API has an **inverse pair**, assert the round trip over a family | libbase122, mvptree |
| Numeric APIs: assert **relative** error, tolerance taken from the interface's own EPS constants, sampled across magnitudes | Math_Library_in_C ×2 |
| File modes, paths and permissions exercised in the form the interface declares | mvptree (`0o600` vs GT's decimal `00755`) |
| Order-dependent results asserted as a **sequence**, not membership | razz_simulation |
| Parser inputs include one **realistic** artifact, not only minimal synthetic ones | libvcd |

This is the expensive change — it re-runs stages 2–3 — and it is the one that addresses the
majority.

### 3.4 Fix mutant *placement* (`rmt/prompts/mutate.md`)

The 11 UNCOVERED failures are a different problem. Requirement 1 spreads mutants over "as
many different C functions as the library allows" — function count, with no reference to the
public surface:

- **Genetic**: mutants build populations via `createInputPop`/`createStructure`; GT uses
  `createPopulation`, which panics. No mutant called it.
- **libutf**: all 10 mutants in `libutf_utf`; the GT failure is in `libutf_string`.
- **tisp**: mutants in `read_num`/`eval_proc`; GT drives `tisp_eval`/`tisp_read`.

Replace the spread rule with an **entry-point budget**: enumerate the public functions in
`src/interfaces/`, require ≥1 mutant reachable from each, and have stage 1 emit an
`entry_point → mutants` map. Where an entry point gets none, that must be stated
("no reachable C code"), not left silent. With ~10 mutants against interfaces this wide you
cannot cover everything, so the map is also the honest statement of what T′ does not
certify.

### 3.5 Measure suite strength instead of inferring it

Two instruments, neither needing an LLM:

- **`cargo llvm-cov` on the reference implementation with only the T′ suite.** Any interface
  function at 0% is a declared blind spot — a real version of the name-matching heuristic
  used in §1, and a direct feedback signal into §3.4.
- **`cargo-mutants` on the reference as an external control.** Score T′ against a bank of
  cheap operator mutants. A suite that kills only the mutant it was written for is by
  definition overfit; that kill rate is the number to report next to "T′ green".

### Order

1. `outcome == "KILLED"` filter — one line, removes three of the 17 stories.
2. Stage 3.5 entry-point and `UNTRANSLATABLE` checks — mechanical, no regeneration.
3. `llvm-cov` / `cargo-mutants` measurement — tells you whether the rest works.
4. Entry-point budget in `mutate.md` (re-runs stage 1) and the widening rules in
   `testgen.md` (re-runs stages 2–3) — the real cost, and the real fix.

Steps 1 and 2 alone would have changed the 97-project result: they invalidate the T′ suites
for Simple_Config and kd3 rather than certifying implementations the ground truth rejects.

---

## 4. Artifacts

| path | what |
|---|---|
| `datasets/codex_expts/mutation_tests/gt_test_report.json` | the 97-project GT grading (72 green) |
| `datasets/codex_expts/mutation_tests/initial_codex_output/mutation_test_report.json` | the T′ grading (853/853) |
| `datasets/codex_expts/mutation_tests/gt_failure_analysis.json` | per-failure classification, with `covering_mutants` |
| `/tmp/gt_full/*.log` | full GT reruns for 21 of the 25 failing projects |
| `src/codex_expts/analyse_gt_failures.py` | the classifier |
| `src/codex_expts/reprobe_gt_failure.py` | rerun one project's GT suite, keep the whole log |
| `src/codex_expts/dump_covered_case.py` | print a failing GT test body next to the mutation tests that cover it |
| `src/codex_expts/run_mutation_tests_expt.sh`, `mt_janitor.sh` | the resumable driver and its memory watchdog |
