You are generating **mutants** of a C library for a mutation-testing study.

A mutant is a single small edit that changes the library's observable behaviour on at
least some input, while still compiling. Good mutants model plausible faults a
programmer (or a C-to-Rust transpiler) could actually introduce.

## The library

Project: `<<PROJECT>>`

<<SOURCES>>

## The observable surface

This C library has been re-specified as a Rust API. A mutant is only useful to us if a
caller can *detect* it through that API, so every mutant must be reachable from one of
these Rust functions/methods:

<<IFACE>>

## What to produce

Exactly <<N>> mutants. Requirements, all of them hard:

1. **Distinct functions.** Spread the mutants over as many different C functions as the
   library allows. Do not put two mutants in the same function unless you have run out
   of functions.
2. **Reachable.** Each mutant must be observable by calling the library through a public
   entry point that corresponds to one of the Rust functions above. Internal helpers are
   fine to mutate, but only if a public call path reaches them.
3. **Compiles.** The mutated code must still be valid C with no new warnings-as-errors.
4. **Small.** One edit. Change an operator, a constant, a bound, an order, a returned
   value — not a rewrite.
5. **Not equivalent.** Avoid edits that cannot change behaviour (dead code, redundant
   assignments, changes only reachable on inputs the code rejects earlier).
6. **`original_snippet` must appear EXACTLY ONCE, VERBATIM, in the named file.** Copy it
   character for character, including indentation. Include 1-3 surrounding lines if the
   line alone is not unique. This is applied by literal string replacement — if it does
   not match uniquely, the mutant is discarded.

7. **Describe it language-independently.** The same mutation will later be applied to a
   *Rust* re-implementation of this library by a different model that will not see the C
   diff. So `semantic_change` must state the behavioural delta in prose that makes sense
   without any C: name the operation being changed, what it did before, what it does
   instead, and under what condition the difference is visible. Do not write "changed `<`
   to `<=` on line 42"; write "the bounds check on the insert index now admits an index
   equal to the current length, so inserting at exactly `len` writes one slot past the
   end instead of being rejected".

Aim for a spread of fault classes, and include at least two memory-behaviour mutants
(buffer bound, allocation size, index arithmetic, off-by-one on a length) where the
library has any — those are the ones we most want to know about.

`fault_class` must be one of:
`arithmetic`, `boundary`, `condition`, `return_value`, `memory_bound`, `alloc_size`,
`state_update`, `type_width`, `control_flow`.

## Output

A JSON array only, no prose outside it:

```json
[
  {
    "id": "m01",
    "file": "src/foo.c",
    "function": "foo_insert",
    "fault_class": "boundary",
    "original_snippet": "    if (i < n) {",
    "mutated_snippet": "    if (i <= n) {",
    "reachable_via": "insert",
    "semantic_change": "The bounds check on the insert index now admits an index equal to the current element count, so inserting at exactly `len` is accepted and writes one slot past the end of the backing storage instead of being rejected as out of range.",
    "observable_effect": "A caller inserting at index == len sees the call succeed (and the container's contents become corrupted) where it previously failed / returned an error.",
    "rationale": "one line: why this is a plausible fault"
  }
]
```

`semantic_change` and `observable_effect` are the fields a Rust engineer would need to
reproduce this exact fault in a Rust port without seeing the C. Write them accordingly.
