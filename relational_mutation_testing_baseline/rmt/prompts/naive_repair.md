Your Rust test suite does not compile. Fix **only the compile errors** and return the
complete file again.

```
<<ERRORS>>
```

## Your previous suite

```rust
<<PREV>>
```

Do not weaken what the tests assert to make the file build — the assertions are the entire
point. Fix imports, paths, types, borrows, and mutability.

If a test calls something that does not exist in the declared interfaces at all, delete
that whole `#[test]` function rather than replacing its assertions with weaker ones that
compile. Keep every test you can make build.

Output the complete corrected Rust file in one fenced block, nothing else.
