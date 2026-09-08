The following Python code uses library <<SOURCE>>. It was migrated to use library
<<TARGET>> instead, but tests written against the original code's behaviour now fail on
the migrated version. Fix the migration so that the original behaviour is preserved. In
the output, first explain the changes you made. Then, provide the modified code. Do not
make any changes to the code that are not related to migrating between these two
libraries. Do not refactor. Do not reformat. Do not optimize. Do not change coding style.

Original code, using `<<SOURCE>>`:

```python
<<IMPL>>
```

The migration to `<<TARGET>>` that the tests reject:

```python
<<CANDIDATE>>
```

## The failing tests

<<FEEDBACK>>

## How to use them

Each test pins down behaviour the original code had and the migration must keep. A test
fails because your migrated code does something observably different, so treat the
assertion as the specification and change the migration until it holds.

Fix the cause, not the symptom. Do not special-case the inputs the test happens to use,
do not add a branch that exists only to satisfy an assertion, and do not import
`<<SOURCE>>` again to get the old behaviour back — the point is to express that behaviour
in terms of `<<TARGET>>`. If a test asserts something you believe the original code did
*not* do, say so and leave that part of the code alone rather than contorting it.
