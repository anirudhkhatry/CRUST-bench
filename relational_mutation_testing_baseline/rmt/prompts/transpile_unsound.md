## Your Rust test fails on a correct implementation

The Rust test below was your transpilation of the C test above. It compiles, but it
**fails against an implementation that passes the benchmark's full ground-truth suite**.
The C test passes on the C library, so one of these is true:

1. **Your translation misread the Rust contract.** The commonest causes: a function whose
   Rust domain is narrower than its C namesake (`Option`-returning on some inputs), a
   0-based vs 1-based index, a `Result` where C returned a code, a length in bytes vs
   elements, a C sentinel that has no meaning in the Rust API, string vs byte comparison,
   or scaffolding assertions about C-only facts (capacity, pointer identity, buffer
   contents past the length) that the Rust API does not promise.
2. **The C-derived expected value is genuinely what the Rust interface requires**, and this
   implementation is wrong. This is rare: the implementation passed a full test suite.

Re-read the interfaces above, then return the corrected complete test file. Rules:

- Fix the *translation*, not the discriminating power. The fact the C test used to tell
  the library from its mutant must still be asserted, expressed correctly against the Rust
  contract. You may drop or `// UNTRANSLATABLE:` a scaffolding assertion about a C-only
  fact, but never the discriminating one.
- Do not replace an exact expected value with a range, a `>=`, or a "not None" check just
  to make it pass. A test that accepts anything discriminates nothing and will be rejected
  downstream anyway.
- If, after re-reading the interface, you are confident case 2 holds, return the file
  unchanged with a first-line comment `// REFERENCE_DISAGREES: <one-line reason>`.

```
<<FAILURE>>
```

## Your previous test

```rust
<<PREV>>
```

Output the complete corrected Rust file in one fenced block, nothing else.
