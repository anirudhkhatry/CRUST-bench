Translate a C **characterisation test** into an equivalent Rust test.

## The C test

It runs a fixed workload against a C library and records results with three provided
functions. It exits 0 and asserts nothing; the recorded values are compared later.

```c
<<PROBE_C>>
```

## The Rust API the translation must use

The library has been re-specified in Rust. The **interfaces below are the entire
contract** — types, signatures, error types. Bodies are `unimplemented!()` on purpose.

Crate name: `<<CRATE>>` · modules: `<<MODULES>>`
(so imports look like `use <<CRATE>>::<<MODULE0>>::...;`)

<<IFACE>>

## A working Rust test against this API

Written earlier from a different C test of the same library; it compiles and passes on a
correct port. Use it as the worked example of how results are reached through this API
(which constructors, how searches or lookups hand results back, how errors surface).

```rust
<<NARROW_RS>>
```

## The recording functions

These are provided in the same module as your code; do not define them, just call them:

```rust
fn obs_ll (label: &str, v: i64);
fn obs_dbl(label: &str, v: f64);
fn obs_str(label: &str, s: &str);
fn obs_done();          // call once, as the last statement
```

## Rules

1. **Same records, same labels, same order.** Every `obs_*` call in the C test has
   exactly one counterpart with the identical label string. The value recorded is the
   Rust value corresponding to what the C test recorded: a C count becomes a
   `.len() as i64`, a C status `rc == 0` becomes `if r.is_ok() { 1 } else { 0 }`, a C
   `char *` becomes the `&str` (or `String`) the Rust API returns, a C `NULL` string is
   recorded as `obs_str(label, "<NULL>")`. Booleans are `1`/`0`.
2. **Only the public API**, exactly as declared above. Invent nothing.
3. **A Rust function that shares a C function's name may not share its domain.** The
   Rust API is a re-specification and is frequently narrower or safer. Read the
   signature before you call it. A C sentinel (`-1`, `NULL`) is a Rust `Err`/`None`;
   record the *outcome* the C test recorded, through whatever route the Rust API
   offers for the same fact.
4. **When a record truly cannot be expressed** with the declared API, omit that
   `obs_*` call and leave a comment `// UNTRANSLATABLE: <label> — <one-line reason>` in
   its place. Do not substitute a guessed value, and do not record a different fact
   under the same label. Omitting is honest; both other options corrupt the comparison.
   Omission is the last resort, though: at most <<MAX_OMIT>> of the <<N_OBS>> records may
   be omitted, so look for the route the working test above uses before giving up on one.
5. **Do not re-derive values.** If the C test records `obs_ll("insert_100/len", n)` after
   inserting 100 elements, your translation records whatever the Rust library reports as
   the length — not the literal 100.
6. **Structure.** A complete file body: `use` lines, any helpers you need, then one
   `#[test] fn rmt_wide()` whose last statement is `obs_done();`. No `mod` wrapper, no
   other `#[test]` functions, no `main`.
7. **Deterministic and fast.** Mirror the C test's fixed inputs exactly (same seeds,
   same sequences, same sizes). No randomness, time, files, threads or network.

## Output

The complete Rust file body in one fenced block, nothing else.

```rust
use <<CRATE>>::...;

#[test]
fn rmt_wide() {
    ...
    obs_done();
}
```
