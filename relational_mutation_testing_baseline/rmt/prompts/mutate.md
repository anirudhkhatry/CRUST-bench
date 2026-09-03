You are generating **mutants** of a C library for a mutation-testing study.

A mutant is an edit that changes the library's observable behaviour on at least some
input, while still compiling. What we are looking for is not random damage: it is the
code a competent programmer would have written **if they had misunderstood the problem**
— the kind of fault that survives review, not the kind a compiler or a glance would
catch.

## The library

Project: `<<PROJECT>>`

<<SOURCES>>

## The observable surface

This C library has been re-specified as a Rust API. A mutant is only useful to us if a
caller can *detect* it through that API, so every mutant must be reachable from one of
these Rust functions/methods:

<<IFACE>>

## What a good mutant looks like

**Aim for mistakes of understanding, not mistakes of typing.** A mechanical mutation tool
can already flip every operator in this file; that is not what we are measuring. We want
faults that require knowing what the code is *for*:

- **The spec is misread at a boundary.** An interval that should be half-open is treated
  as closed; a limit that means "capacity" is used as "last valid index"; a count is used
  where a size in bytes is meant.
- **A case is not thought about.** Empty input, a single element, all-equal elements,
  duplicate keys, a full container, the wrap-around at the end of a cycle, the first or
  last iteration of a loop.
- **The right operations in the wrong order.** State is updated before the check that
  should have guarded it; a resource is released before its last use; a value is
  normalised after it has already been consumed.
- **An invariant is not restored.** A partial failure leaves the structure half-updated —
  a length incremented without the write, a link repointed without the reverse link, a
  cached summary not invalidated after the data it summarises changed.
- **The wrong thing is compared or propagated.** Lengths compared instead of contents,
  a pointer instead of the value behind it, an error from an inner call swallowed and a
  success reported, a sentinel treated as valid data.
- **A convention is misapplied.** Wrong rounding or truncation direction, degrees where
  radians are meant, inclusive where exclusive is meant, 0-based where 1-based is meant,
  a signed value read as unsigned.

- **A contract between functions is misremembered.** The producer of a value and its
  consumers disagree: a field that holds a capacity is written as a capacity and read as a
  length, an error code convention is inverted at the call site that checks it, an index
  returned 0-based is consumed 1-based, a buffer handed over as "owned" is also freed by
  the caller.

Such an edit is usually **several lines inside one function** — a rearranged sequence, an
added or removed guard, a differently-computed intermediate. That is expected and wanted.

It may also be **several edits across different functions or files**, as long as they are
the *same* misunderstanding showing up everywhere it is relevant. A programmer who
misremembers what a struct field means gets it wrong in the writer and in every reader, not
in one place; that whole-program shape is exactly the kind of fault we want, and a mutant
that changes only one of the sites would often be caught by a mismatch the real fault does
not produce. Use multiple edits when the fault genuinely has multiple sites — not to bundle
two faults together.

**What not to produce.** A bare `+`→`-`, `<`→`<=`, `&&`→`||`, `true`→`false`, or a deleted
statement is acceptable *only* when it genuinely encodes one of the misunderstandings
above, and the `rationale` must then say which one. An edit whose only justification is
"the operator is now different" is rejected.

## What to produce

Exactly <<N>> mutants. Requirements, all of them hard:

1. **Distinct sites.** Spread the mutants over as many different C functions as the
   library allows. Do not put two mutants in the same function unless you have run out
   of functions.
2. **Reachable.** Each mutant must be observable by calling the library through a public
   entry point that corresponds to one of the Rust functions above. Internal helpers are
   fine to mutate, but only if a public call path reaches them.
3. **Compiles.** The mutated code must still be valid C with no new warnings-as-errors.
4. **One coherent fault.** The edit may span several lines, several functions and several
   files, and may add, remove or reorder statements — but it is *one misunderstanding*,
   consistently applied. What it must not be is two unrelated faults bundled together, a
   refactor, or a change to any function's signature.
5. **Not equivalent.** Avoid edits that cannot change behaviour (dead code, redundant
   assignments, changes only reachable on inputs the code rejects earlier).
6. **Every `original_snippet` must appear EXACTLY ONCE, VERBATIM, in its named file.** Copy
   it character for character, including indentation. Include 1-3 surrounding lines if the
   line alone is not unique. These are applied by literal string replacement — if any one
   of a mutant's snippets does not match uniquely, the whole mutant is discarded.

7. **Describe it language-independently.** The same mutation will later be applied to a
   *Rust* re-implementation of this library by a different model that will not see the C
   diff. So `semantic_change` must state the behavioural delta in prose that makes sense
   without any C: name the operation being changed, what it did before, what it does
   instead, and under what condition the difference is visible. Do not write "changed `<`
   to `<=` on line 42"; write "the bounds check on the insert index now admits an index
   equal to the current length, so inserting at exactly `len` writes one slot past the
   end instead of being rejected".

## The mix

Label every mutant with `mutant_kind`:

- **`semantic`** — a mistake of understanding, per the section above. **These must be the
  large majority: at least <<N_SEMANTIC>> of the <<N>>.**
- **`operator`** — a deliberately simple single-token edit, of the sort a mechanical
  mutation tool would emit. Produce **exactly <<N_OPERATOR>>** of these, as a control: they
  let us tell whether a low transfer rate is about C→Rust translation or merely about
  harder faults.

Within the `semantic` group, aim for a spread of fault classes and include at least two
memory-behaviour mutants (buffer bound, allocation size, index arithmetic, off-by-one on a
length) where the library has any.

`fault_class` must be one of:
`arithmetic`, `boundary`, `condition`, `return_value`, `memory_bound`, `alloc_size`,
`state_update`, `type_width`, `control_flow`, `spec_misread`, `edge_case`, `ordering`,
`invariant`, `error_path`, `stale_state`.

## Output

A JSON array only, no prose outside it. `edits` carries one entry per site — one for a
fault that lives in a single function, several for a fault whose sites are spread out:

```json
[
  {
    "id": "m01",
    "function": "foo_insert",
    "fault_class": "invariant",
    "mutant_kind": "semantic",
    "edits": [
      {
        "file": "src/foo.c",
        "original_snippet": "    if (c->len == c->cap && !foo_grow(c))\n        return -1;\n    c->items[c->len] = value;\n    c->len++;",
        "mutated_snippet": "    c->len++;\n    if (c->len > c->cap && !foo_grow(c))\n        return -1;\n    c->items[c->len - 1] = value;"
      }
    ],
    "reachable_via": "insert",
    "semantic_change": "The element count is incremented before the container is grown rather than after the element is stored, so when a growth attempt fails the container is left reporting one more element than it holds. Subsequent reads of the last index return whatever was already in the backing storage, and a later successful insert overwrites from the inflated count.",
    "observable_effect": "After an insert that fails for lack of capacity, the container's reported length is one too large and reading the final element yields stale or uninitialised data instead of the caller seeing an unchanged container.",
    "rationale": "Updating the count first reads naturally, and the failure path that makes it wrong is the one least likely to be exercised."
  }
]
```

`semantic_change` and `observable_effect` are the fields a Rust engineer would need to
reproduce this exact fault in a Rust port without seeing the C. Write them accordingly.
