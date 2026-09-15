You are constructing a benchmark of *incorrect but plausible* migrations. Below is a
migration that is known to be correct. Introduce exactly one behavioural fault into it.

## The correct migration

<<SOURCES>>

## The transformation it was supposed to perform

<<TRANSFORMATION>>

## Hazards specific to this kind of migration

<<HAZARDS>>

## Requirements

The result must:

- still build and load without error;
- still look like a competent attempt at the migration -- a reviewer skimming the diff
  should not immediately see a bug;
- differ in *observable behaviour* on at least one realistic input.

Do not introduce a syntax error, delete a function, or make the program obviously broken.
The fault must be the kind of mistake a careful engineer makes under time pressure.

## Output

Return a single fenced `json` block with

- `file`: the path exactly as labelled above
- `original_snippet`: a verbatim span from that file, unique within it
- `mutated_snippet`: the replacement
- `fault_category`: one of <<CATEGORIES>>
- `description`: one sentence on the behavioural difference, for the benchmark's records
