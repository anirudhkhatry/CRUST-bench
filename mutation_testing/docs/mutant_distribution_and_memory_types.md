# Mutant distribution, memory-type filtering, and directions for Rust-correctness testing

Data: gpt-5.5 run, `c+rust` setting (strongest, consistent), pooled over all projects
(`mutants.out/{caught,missed,unviable,timeout}.txt`). 12,208 mutants total.

## 1. Overall distribution — Rust's type system filters most mutants

| outcome | count | share |
|---|---|---|
| **unviable** (mutant doesn't compile) | 7,536 | **61.7%** |
| caught (killed by tests) | 2,939 | 24.1% |
| timeout | 1,196 | 9.8% |
| missed (survived) | 537 | 4.4% |

**~62% of all mutants are unviable** — the Rust compiler / borrow checker statically
rejects them. Only ~28% are *viable* (caught+missed = 3,476). Of viable mutants, tests
kill **84.6%**.

This is the first-order finding and it's Rust-specific: in C, most operator/return
mutations compile, so mutation testing probes them. In Rust, the type/ownership system
makes a majority of syntactic mutations ill-typed, so they never become testable bugs.
That's simultaneously (a) a quantification of how much correctness Rust's static
guarantees buy you, and (b) a problem for mutation *testing* — the default operator set
wastes ~62% of its budget on mutants that can't exist.

By mutant kind:

| kind | n | viable | kill% (of viable) | unviable% |
|---|---|---|---|---|
| binary-operator | 8,833 | 2,532 | 83.9% | 60.3% |
| return-value | 2,676 | 684 | 87.6% | 68.1% |
| delete-branch | 699 | 260 | 82.7% | 55.4% |

## 2. Memory-type view — which types get filtered vs. tested

Return-value mutants (a function body replaced by a default) reveal the mutated **return
type**. Bucketing by type:

| type bucket | n | viable | kill% | **unviable%** |
|---|---|---|---|---|
| primitive / `Copy` (`i32`,`u32`,`f64`,`bool`,…) | 1,190 | 341 | 91.5% | 63.0% |
| **custom struct/enum (abstract datatypes)** | 517 | 82 | 89.0% | **80.3%** |
| `Option<T>` | 435 | 103 | **68.0%** | 71.5% |
| heap / owned ADT (`Vec`,`String`,`Box`,`Rc`,`Arc`,maps) | 357 | 106 | 90.6% | 65.3% |
| `Result<…>` | 113 | 33 | 93.9% | 61.9% |
| reference / borrow (`&T`,`&mut T`) | 64 | 19 | 89.5% | 67.2% |

### How many mutants pertain to memory types?

Only *return-value* mutants carry a type; operator/branch mutants are type-agnostic. Split
that way (c+rust):

| group | gpt-5.5 total | viable | unviable% | Opus-5 total | viable | unviable% |
|---|---|---|---|---|---|---|
| **Memory-type return** (ref/generic/`Option`/`Result`/heap/custom ADT) | **1,486 (12%)** | 343 | **72%** | **1,019 (11%)** | 281 | **66%** |
| Primitive/unit return | 1,190 (10%) | 341 | 63% | 816 (9%) | 286 | 55% |
| Operator/branch (type-agnostic) | 9,532 (78%) | 2,792 | 60% | 7,293 (80%) | 2,375 | 56% |
| **All** | 12,208 | 3,476 | — | 9,128 | 2,942 | — |

- Memory-typed mutants are only **~12% of all mutants**, and they carry the **highest
  unviable rate** (72% gpt / 66% Opus). Of ~1,486 gpt memory-type mutants, only **343
  (~23%) survive to be testable.**
- When viable they are **well-killed** (83.7% gpt / 97.9% Opus). So the deficit is *not*
  detection — it is that most memory-type mutations never become testable mutants. That is
  the quantified motivation for the type-aware operators in §3.

Two further signals (from the per-type table above):

**(a) Abstract datatypes are the most heavily *filtered*.** Custom struct/enum-returning
functions are **80% unviable** — the default-return mutation (`Default::default()`)
usually doesn't typecheck (no `Default` impl, or the ADT can't be trivially constructed),
so cargo-mutants throws the mutation away. **Mutation testing barely probes
ADT-returning code**: of 517 such mutants only 82 are viable. So "is my `struct
Buffer`/`enum Node` logic tested?" is a question this default operator set can't really
answer.

**(b) `Option`-returning logic is the biggest *blind spot among viable mutants*.**
`fn -> Option<T>` → `None` is usually viable, but only **68% killed** — the lowest of any
bucket. Tests often exercise the `Some` path and never assert the "not found / absent"
path, so replacing the body with `None` survives. This is a concrete correctness gap:
nullable/lookup functions are under-checked.

Heap-owned collections (`Vec`/`String`/`Box`) are, by contrast, well-behaved: ~65%
unviable but ~91% killed when viable (replacing with an empty collection is easy for
tests to notice via content assertions). Stack primitives are similar (91.5% killed).

**So the "memory footprint" answer:** it's not stack-vs-heap per se — it's
**constructibility under the default mutation operator**. Types with an easy default
(primitives, `Vec`/`String`, `Result`) yield viable, well-killed mutants. **Custom ADTs
have no cheap default → they get filtered out (80% unviable) → they're the least
mutation-probed part of the code.** And `Option` yields viable-but-surviving mutants,
marking a real test blind spot.

## 2b. Case study — the projects with the fewest mutants are ADT/generic/reference-heavy

Ranking projects by viable-mutant count (gpt-5.5, `c+rust`) makes the filtering concrete.
First, a caveat: four projects report 0 viable **because nothing ran** — cargo-mutants'
baseline itself failed, so `caught=missed=unviable=timeout=0` (XOpt, fslib, bhshell,
VaultSync). Those are *failed runs*, not low-mutant projects; exclude them.

The genuinely fewest-mutant projects are low for one reason — **the compiler filters
almost everything as unviable:**

| project | viable | unviable | timeout | unviable % | what it is |
|---|---|---|---|---|---|
| geofence | 1 | 71 | 5 | 92% | point-in-polygon over `&[Point]` |
| libqueue | 1 | 6 | 6 | 46% | generic `Queue<T>` |
| morton | 2 | 58 | 2 | 94% | Z-order bit codes (`u64`/`Morton`) |
| leftpad | 2 | 20 | – | 91% | string pad into `&mut [u8]` |
| file2str | 2 | 13 | – | 87% | file → `Option<String>` |

The memory types dominating these projects are exactly the ones the default operators
cannot mutate viably:

- **References / borrows** — `&Point`, `&[Point]`, `&str`, `&mut [u8]`, `Option<&T>`.
  A body returning `&T` can't be replaced by a default — there's nothing to borrow.
- **Generics** — `Queue<T>` with `Option<T>` / `Option<&T>` / `&mut self`. The sharpest
  example, an unviable mutant cargo-mutants actually attempted:
  ```
  replace Queue<T>::front -> Option<&T> with Some(Box::leak(Box::new(Default::default())))
  ```
  It fabricates a `&T` via `Box::leak` + `Default` → needs `T: Default`, which the generic
  queue doesn't bound → **unviable**.
- **Option-wrapped heap/refs** — `Option<String>`, `Option<&T>` (also the biggest
  *blind spot* type in §2, here mostly *filtered*).
- **Custom structs** — `Point`, `Morton`, `Queue<T>`: no cheap `Default`, so
  `-> T with Default::default()` doesn't compile.
- **Bit / const math** — morton's `u64` mask arithmetic: operator swaps like `+ → *`
  overflow during const evaluation → compile-rejected.

**Takeaway:** the least-mutant projects are precisely the reference-, generic-, `Option`-,
and custom-ADT-heavy ones — the memory/ownership code Rust won't let cargo-mutants mutate
with its stock operators. Mutation testing therefore gives **almost no signal on exactly
the code where memory-management correctness matters most**, which is the strongest
motivation for the type-aware operators in §3.

## 3. How to improve it — Rust-aware mutation operators

The 62% unviable rate and the ADT gap both point at the same fix: **cargo-mutants'
default C-style operators don't fit Rust's type system.** Type-directed operators would
recover viable, meaningful mutants where we currently get nothing:

- **ADT-returning fns:** instead of `Default::default()`, mutate to a *different valid
  constructor / enum variant* of the same type (e.g. `Node::Leaf` → `Node::Internal`,
  swap struct fields), or perturb one field. Viable and semantically meaningful.
- **`Option`/`Result`:** `Some(x)` ↔ `None`, `Ok(x)` ↔ `Err(_)` with a valid inner —
  and, given the 68% Option kill rate, pair with **test-gen that asserts the absent
  path**.
- **Collections / heap:** off-by-one on length/capacity, drop first/last element, swap
  push/pop order — richer than "return empty".
- **Ownership / lifetimes:** clone→move, drop-early, `&mut` alias perturbations — memory
  semantics the current operators never touch.

This is a fundable direction on its own: *a Rust-type-aware mutation-operator set*, with
the unviable-rate reduction and per-type kill-rate as the evaluation metric.

## 4. Toward testing correctness of Rust implementations

Mutation score is an **adequacy proxy**: a surviving (missed) viable mutant is a precise
statement "the tests do not constrain this behavior." Two uses:

1. **Surviving-mutant-driven test generation (closed loop).** For each missed mutant,
   feed the diff + the reference behavior (the C source / RBench oracle) to the model and
   ask for a test that distinguishes original from mutant. Re-run mutation; repeat until
   the viable set is killed. This turns mutation from a *measurement* into a *test-
   improvement engine* and directly raises correctness confidence — and it naturally
   targets the `Option`/branch blind spots above.

2. **Static-vs-dynamic correctness accounting.** The 62% unviable is a per-project
   measure of how much of the "bug space" Rust's type/borrow system rules out at compile
   time; the remaining viable-but-missed set is what *tests* must cover. Reporting both
   (e.g. "Rust statically rejects 62% of mutations; tests catch 85% of the rest; the
   surviving 4% cluster in Option/ADT logic") is a clean correctness story for a paper,
   and lets you compare languages (C would have a far lower unviable rate) and settings
   (our `src+test` vs `c+rust`).

The immediate, cheap next experiments:
- Break the distribution down **per project category** (parsers vs numeric vs data-
  structures) and confirm the ADT/`Option` gaps concentrate where expected.
- Re-run with an **Option/ADT-aware operator** and measure viable-rate lift.
- Prototype the **surviving-mutant → test-gen loop** on the ~537 missed mutants and see
  how many the model can kill given the C reference.

---
*Distribution from `results_gpt55_4settings/c_and_rust`; per-type buckets parsed from
cargo-mutants return-value mutant descriptions. Kill% is over viable (caught+missed);
unviable%/timeout% are over all mutants of that group.*
