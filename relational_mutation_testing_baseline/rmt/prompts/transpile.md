Transpile a C test into an equivalent **Rust integration test**.

## The C test

This test passes on a C library and fails on a mutant of it. It exits 0 to pass and
non-zero to fail.

```c
<<TEST_C>>
```

## The Rust API it must be re-expressed against

The library has been re-specified in Rust. The **interfaces below are the entire
contract** — types, signatures, error types. Bodies are `unimplemented!()` on purpose;
an implementation of these interfaces is what your test will run against.

Crate name: `<<CRATE>>` · modules: `<<MODULES>>`
(so imports look like `use <<CRATE>>::<<MODULE0>>::...;`)

<<IFACE>>

## Rules

1. **Preserve the test's discriminating power.** Every concrete value the C test asserts
   must survive as an assertion on the corresponding Rust value. If the C test asserts
   `len == 7`, the Rust test asserts `7` too. Do not weaken an equality into a range, and
   do not drop an assertion because it is awkward to express.
2. **Only the public API.** Use the crate's public items exactly as declared above.
   Invent nothing: no helper functions on types that do not have them, no fields that are
   not declared, no methods that are not in the interface.
3. **Translate C idioms to their Rust meaning, not their Rust spelling.**
   - A C function returning `-1` / `NULL` / an errno on failure corresponds to the Rust
     `Result::Err` / `Option::None` in the interface. Assert the *failure*, not the `-1`.
   - C out-parameters correspond to returned values or tuples.
   - Explicit lengths passed alongside pointers are usually implicit in a Rust slice or
     `String`; assert on `.len()` where the C test asserted on the length variable.
   - Fixed-width C types map to the fixed-width Rust type in the interface signature.
   If a C assertion is about behaviour that Rust's type system makes unrepresentable
   (a null pointer, an uninitialised read), translate the *observable outcome* the C test
   was checking, and say so in a `// NOTE:` comment.
4. **One `#[test]` function**, named `rmt_kill`. A panic (including a failed `assert!`)
   is the failure signal — that is the Rust equivalent of the C test's non-zero exit.
5. **Deterministic and fast.** No randomness, time, files, threads, or network.

## Output

The complete Rust test file in one fenced block, nothing else.

```rust
use <<CRATE>>::...;

#[test]
fn rmt_kill() {
    ...
}
```
