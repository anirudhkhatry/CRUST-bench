Your Rust test does not compile. Fix **only the compile errors** and return the complete
file again.

```
<<ERRORS>>
```

## Your previous test

```rust
<<PREV>>
```

Do not change what the test asserts. Do not weaken, delete, or `#[ignore]` an assertion
to make the file build — the assertions are the entire point of this test, and a test
that builds because it checks less is worse than one that does not build. Fix imports,
paths, types, borrows, and mutability only.

If an assertion cannot be expressed against the declared public API at all, keep it as a
commented-out line prefixed `// UNTRANSLATABLE:` with a one-line reason, rather than
silently dropping it.

Output the complete corrected Rust file in one fenced block, nothing else.
