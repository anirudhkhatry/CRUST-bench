Write a **Rust test** that detects one specific fault in a Rust library.

You are shown the library's real implementation and the exact edit that introduces the
fault. Your job is to write a test that **passes the correct implementation and fails the
faulty one**.

## The correct implementation

This is the reference. It is assumed correct: whatever it computes is the specification.

<<IMPL>>

## The fault

This diff turns the correct implementation into the faulty one. Only these lines change;
everything else is identical.

```diff
<<DIFF>>
```

Faulty function: `<<FUNCTION>>` · fault class: `<<FAULT_CLASS>>`

## The public API your test must go through

Your test is an integration test (`tests/`), so it may only use the crate's **public**
API — the interfaces below. You may not call private helpers, reach into the module you
just read, or modify the library.

Crate name: `<<CRATE>>` · modules: `<<MODULES>>`
(so imports look like `use <<CRATE>>::<<MODULE0>>::...;`)

<<IFACE>>

## What the test has to do

1. **Find an input that reaches the changed lines** and whose result differs. Reading the
   diff tells you what changes; work out concretely which call, with which argument,
   propagates that change to something the public API returns.
2. **Assert the value the correct implementation produces**, computed by tracing the code
   above by hand. Not the faulty value, not an inequality, not a self-consistency check
   between two calls — the exact value, so the test passes the reference and fails the
   mutant.
3. If the changed lines are genuinely unobservable through the public API — dead code, or
   a difference that every public path discards — say so in one line beginning
   `UNREACHABLE:` and return no code block. Do not invent a test you believe will fail
   the reference.

## Rules

- **1 to 3 `#[test]` functions**, each named `rn_<what_it_checks>`. Prefer one focused
  test; add a second only if a different call is needed to pin the fault down.
- **Assert exact values.** `assert_eq!(x, 7)`, not `assert!(x > 0)`. A test that only
  checks a call did not panic distinguishes nothing.
- **A panic (a failed `assert!`) is the failure signal.** Do not catch it.
- **Every test must pass the correct implementation.** A test that fails the reference is
  worthless here: it would "detect" every fault, including the ones nothing can detect.
- **Self-contained, deterministic and fast.** No randomness, time, threads or network. If
  you need a file, create it under a unique name in the current directory and remove it.
- Read each signature before you use it: an `Option`-returning function fails by returning
  `None`, which type-checks perfectly and then fails your assertion on *correct* code.

## Output

The complete Rust test file in one fenced block, nothing else.

```rust
use <<CRATE>>::...;

#[test]
fn rn_...() {
    ...
}
```
