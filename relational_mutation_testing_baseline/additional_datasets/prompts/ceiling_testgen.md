Below is a correct implementation and a diff that introduces a behavioural fault into it.
Write one test that passes on the correct implementation and fails on the faulty one.

## The correct implementation

<<SOURCES>>

## The fault that was introduced

```diff
<<DIFF>>
```

## What to write

Write a single <<LANGUAGE>> test file using <<FRAMEWORK>>.

- Import the code under test with EXACTLY this line, character for character:
  `<<IMPORT_LINE>>`
  Do not alter that path.
- Assert exact values, and reach the fault through the public surface.
- The test is executed against both versions and is kept only if it separates them.
- If the fault is genuinely unreachable through the public API, say so in a comment on the
  first line beginning `// UNREACHABLE:` and write the closest test you can.

Return only the test file in a single fenced code block.
