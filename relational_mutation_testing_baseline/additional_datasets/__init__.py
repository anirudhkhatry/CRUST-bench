"""Task and candidate records, and the metadata.json the spec's §9 requires.

One benchmark instance is a Task (a trusted P0 plus the migration spec M) and a set of
Candidates (compiling post-migration implementations, each labelled GOOD or BAD by a
hidden oracle the test-generating arms never see).
"""
