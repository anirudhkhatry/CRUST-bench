You are given one test, and the declared public surface of the file that test exercises.
Re-express the test so that it depends ONLY on that public surface.

## The test

```
<<TEST>>
```

## The public surface of the file under test

These are the only names the file exports. Nothing else about the file is available to you,
and nothing else may appear in your test: not a helper, not an internal module path, not a
field of an object you have not been shown.

<<SURFACE>>

## The transformation that is about to be applied

<<TRANSFORMATION>>

## What to write

Write a single <<LANGUAGE>> test file using <<FRAMEWORK>> that makes exactly the same
observations the test above makes, reaching them only through the exported names listed
above.

- Import the code under test with EXACTLY this line, character for character:
  `<<IMPORT_LINE>>`
  Do not alter that path. It is where the file actually sits in the test workspace, and a
  path you invent instead will not resolve.
- Keep every assertion and every asserted value exactly as it is. You are changing HOW the
  test reaches the behaviour, never WHAT it claims about it. Do not relax an assertion, do
  not delete a case, do not replace an exact value with a looser matcher. The result is
  re-executed against the original program and against a faulty one, and a test that has
  stopped discriminating is discarded.
- If the test already uses only the exported names, return it unchanged.
- Do not add comments explaining the change.

Return only the test file in a single fenced code block.
