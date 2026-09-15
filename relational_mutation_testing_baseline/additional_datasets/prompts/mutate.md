You are producing mutation-testing fixtures. Propose <<N>> faults to inject into the
program below. Each fault must be a small, plausible mistake a developer could make --
not a random token swap, and not something that breaks the program outright.

## The program

<<SOURCES>>

## Output

Return a single fenced `json` block: a list of <<N>> objects, each with

- `file`: the path exactly as labelled above
- `original_snippet`: a verbatim span copied from that file, unique within it
- `mutated_snippet`: the replacement
- `semantic_change`: one sentence describing the behavioural difference in
  language-agnostic terms, without naming any syntax
- `fault_class`: one of arithmetic, boundary, condition, return_value, state_update,
  control_flow, edge_case, ordering, invariant, error_path

`original_snippet` must appear exactly once in the named file, or the fault is discarded.
