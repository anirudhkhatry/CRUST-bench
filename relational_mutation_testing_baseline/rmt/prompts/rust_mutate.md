Introduce a **specific described fault** into a Rust implementation.

We have an independent fault description, written against the library's behaviour rather
than any particular language. Your job is to make this Rust implementation exhibit
*exactly that fault* — no more, no less.

## The fault to introduce

Fault class: `<<FAULT_CLASS>>`
Affected operation: `<<FUNCTION>>` (reachable through the public API as `<<REACHABLE_VIA>>`)

**What changes:**
<<SEMANTIC_CHANGE>>

**What a caller observes:**
<<OBSERVABLE_EFFECT>>

## The Rust implementation

<<IMPL>>

## Rules

1. **Same fault, idiomatic location.** Find the code that performs the described
   operation and change it so the described behaviour holds. The Rust code is an
   independent re-implementation, so it will not look like any C original — locate the
   fault by *meaning*.
2. **Minimal.** Edits matching the size of the described fault and no larger. One edit is
   the common case; use several only when the described fault genuinely has several sites
   in this implementation — a misunderstood value written in one place and read in
   others — so that the behaviour a caller sees is the described one. Do not refactor,
   rename, reformat, or fix anything else.
3. **Still compiles.** The result must be valid Rust that builds. If the described fault
   is one Rust's type system makes unrepresentable (e.g. reading uninitialised memory,
   a null pointer dereference) do NOT reach for `unsafe` to force it — instead introduce
   the *closest observable equivalent* through safe code, and say what you did in
   `note`. If there is no observable equivalent at all, return
   `{"applicable": false, "note": "..."}` and no patch.
4. **Do not touch <<FORBIDDEN>>.**
5. **Every `original_snippet` must appear EXACTLY ONCE, VERBATIM, in its named file.** Copy
   it character for character, including indentation; include surrounding lines if one
   line is not unique. They are applied by literal string replacement, all or nothing.

## Output

A JSON object only, no prose outside it:

```json
{
  "applicable": true,
  "edits": [
    {
      "file": "src/foo.rs",
      "original_snippet": "        if idx < self.items.len() {",
      "mutated_snippet": "        if idx <= self.items.len() {"
    }
  ],
  "note": "one line: where you located the operation and why this edit reproduces the described fault"
}
```
