You are given a program and a transformation that will be applied to it. Write a test
suite that will be run against the transformed program to check that the transformation
preserved observable behaviour.

## The program

<<SOURCES>>

## The transformation that will be applied

<<TRANSFORMATION>>

## What to write

Write a single <<LANGUAGE>> test file using <<FRAMEWORK>>.

- Import the code under test with EXACTLY this line, character for character:
  `<<IMPORT_LINE>>`
  Do not alter that path. It is where the file actually sits in the test workspace, and a
  path you invent instead will not resolve.
- Assert exact values. A test that only checks a value is non-null, or that two calls
  agree with each other, cannot detect a behavioural change.
- Test only the behaviour visible through the program's public surface, using the names
  the program already exports. The transformation may reorganise internals; a test that
  reaches into internals will fail a correct transformation.
- Cover edge cases: empty inputs, boundaries, error paths, ordering, and any value the
  code treats specially.
- Never write a test that could fail the program exactly as given to you above.

Return only the test file in a single fenced code block.
