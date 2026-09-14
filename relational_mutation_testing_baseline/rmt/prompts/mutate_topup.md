## Already tried on this library

The mutants below were proposed for this library earlier and each one either succeeded or
fell out of the pipeline for the reason given. Your <<N>> mutants must be **new faults**:
different from every entry here in what they change, not merely in where the edit sits.

<<TRIED>>

Learn from the fates above:

- Prefer functions that no entry above touches, so the suite spreads over the API.
- Where an earlier fault could not be observed through the public API, choose faults whose
  effect is a **concrete value a caller can read back** -- a returned length, a lookup
  result, a serialised byte, an error where success was due -- not an internal state
  change with no accessor.
- Where an earlier fault had no Rust counterpart or contradicted correct Rust, avoid faults
  that live in C-specific mechanics (allocation sizes, pointer identity, uninitialised
  memory, capacity fields the caller never sees). Prefer faults in the *logic* of the
  operation -- boundaries, ordering, error paths, parsing, arithmetic -- which every
  re-implementation must also get right.
- All <<N>> should be `semantic`; the operator control group already exists.
