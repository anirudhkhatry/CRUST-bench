Write a test that detects the following fault in the program below.

## The program

<<SOURCES>>

## The fault

<<SEMANTIC_CHANGE>>

## What to write

Write a single <<LANGUAGE>> test file using <<FRAMEWORK>> that **passes** on the program
exactly as shown above and **fails** on a version of it carrying that fault.

- Import the code under test with EXACTLY this line, character for character:
  `<<IMPORT_LINE>>`
  Do not alter that path. It is where the file actually sits in the test workspace, and a
  path you invent instead will not resolve.
- Assert exact values. The test is executed against both versions and kept only if it
  separates them, so a weak assertion will simply be rejected.
- Reach the fault through the program's public surface, using the names it already
  exports. A later transformation may reorganise internals.
- Do not describe the fault in a comment; just detect it.

Return only the test file in a single fenced code block.
