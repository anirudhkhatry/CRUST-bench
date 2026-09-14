Write a **Rust test suite** for a library that has been re-specified in Rust from an
existing C implementation.

## The C implementation

This is the original library. It is the specification: the Rust version is meant to
preserve its observable behaviour. Read it to work out what each operation returns, what
its edge cases are, and where an implementer is most likely to get it subtly wrong.

<<SOURCES>>

## The Rust API your tests must run against

The **interfaces below are the entire contract** — types, signatures, error types. Bodies
are `unimplemented!()` on purpose; an implementation of these interfaces is what your
tests will run against. You are not shown that implementation, and you must not assume
anything about it beyond these declarations and the C behaviour above.

Crate name: `<<CRATE>>` · modules: `<<MODULES>>`
(so imports look like `use <<CRATE>>::<<MODULE0>>::...;`)

<<IFACE>>

## What makes this suite good

Your tests are the only thing standing between a subtly wrong implementation and a green
build. Aim them where a re-implementer plausibly goes wrong: off-by-one at the ends of a
range, the empty and single-element input, the failure path, the *second* call on the same
object, an asymmetric argument order, a size that is not a round multiple, a value at the
extreme of its type rather than a typical one.

## Rules

1. **Cover the public API.** Every public function that can be reached and observed
   through the declared interfaces should be exercised by at least one test.
2. **Ground every expected value in the C code.** Trace the C by hand and assert the value
   it would produce. Do not assert what seems reasonable — assert what the C does.
3. **Assert exact values.** `assert_eq!(x, 7)`, not `assert!(x > 0)`. A test that only
   checks a call did not panic distinguishes nothing. Never assert merely that two calls
   agree with each other.
4. **A Rust function that shares a C function's name may not share its domain.** The Rust
   API is a *re-specification*, not a transliteration, and it is frequently narrower. A C
   accessor that worked on any container often becomes a method defined on one variant,
   returning `None` for the rest; a C function that accepted a sentinel often becomes one
   that does not. Read each signature before you use it, and satisfy yourself that the
   value you pass is in its domain. A call that compiles is not evidence that it is:
   `Option`-returning functions fail by returning `None`, which type-checks perfectly and
   then fails your assertion on *correct* code.
5. **Translate C idioms to their Rust meaning, not their Rust spelling.** A C function
   returning `-1` / `NULL` / an errno on failure corresponds to `Result::Err` /
   `Option::None` here — assert the *failure*, not the `-1`. Out-parameters become
   returned values or tuples. An explicit length beside a pointer is usually `.len()` on a
   slice or `String`. Behaviour that Rust's type system makes unrepresentable (a null
   pointer, an uninitialised read) has no test; skip it.
6. **Every test must pass a correct implementation.** If the C behaviour is genuinely
   unclear, or the interface does not say enough to pin the answer down, leave that case
   out. A wrong assertion is worse than a missing test: it fails correct code, which is
   the one thing a test suite may never do.
7. **Between 12 and 25 `#[test]` functions**, each named `nb_<what_it_checks>`. One
   behaviour per test, so a failure names the fault. A panic (including a failed
   `assert!`) is the failure signal.
8. **Self-contained, deterministic and fast.** No randomness, time, files, threads, or
   network, and no dependency between tests — they run in one process and any order.

## Output

The complete Rust test file in one fenced block, nothing else.

```rust
use <<CRATE>>::...;

#[test]
fn nb_...() {
    ...
}
```
