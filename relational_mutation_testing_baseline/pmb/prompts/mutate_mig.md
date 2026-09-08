You are designing **anticipated migration defects** for a study of library migration.

This module uses **`<<SOURCE>>`**. In the real commit we are studying, a developer rewrote
it to use **`<<TARGET>>`** instead. Your job is to work out what could plausibly *go
wrong* in that rewrite, and then write, into this original file, the behaviour the module
would have if it had gone wrong that way.

Read that twice, because it is not the usual mutation task. You are not looking for bugs
in this code. This code is correct. You are looking for **the places where `<<SOURCE>>`
and `<<TARGET>>` disagree**, and you are producing, for each one, a version of this file
that behaves the way a botched migration would have made it behave.

The point of the exercise: a test that passes on this file and fails on your version is a
test that pins down exactly the behaviour the migration must preserve. That test will then
be rewritten against `<<TARGET>>` and used to judge somebody's real migration.

## The module

Repository: `<<REPO>>` · file: `<<FILE>>`

```python
<<SOURCE_CODE>>
```

## The migration sites

These are the exact places this file changes when it moves from `<<SOURCE>>` to
`<<TARGET>>`, taken from the recorded developer diff:

<<CHANGES>>

## How to find each defect

For every mutant, work through these three steps in order. The `migration_hazard` field
records step 1, and it is the field we check hardest.

**Step 1 — name a real disagreement between the two libraries.** At one of the sites
above, how does `<<TARGET>>` behave differently from `<<SOURCE>>`? Draw on what you know
about these two libraries specifically. Typical shapes:

- a default changed (`strict=True` where the old call was lenient; a different encoding,
  timeout, sort order, buffering mode, protocol version)
- the return type or shape changed (an object where a tuple was returned; a generator
  where a list was; bytes where str was; a wrapper that must be unwrapped)
- the failure mode moved (raises where the old one returned `None`; a different exception
  class; an error surfaced at call time rather than at use time)
- an argument means something else (a delta vs an absolute; seconds vs milliseconds;
  retries vs total attempts; a flag whose polarity is inverted)
- laziness or ordering changed (evaluated on demand rather than eagerly; iteration order
  no longer stable; a resource closed at a different point)
- a near-synonym exists and is subtly not the same call

**Step 2 — assume a migrator gets it wrong.** They carry the old assumption across
unchanged, or they over-correct. Say concretely what the migrated code would then do.

**Step 3 — write that behaviour into the file above.** Edit this `<<SOURCE>>`-using code
so that it *now behaves the way the botched migration would*. This is the mutant.

### Worked example

Site: `ipaddr.IPv4Network(...)` becomes `ipaddress.IPv4Network(...)`.
Step 1: `ipaddress` constructs in **strict mode** by default and rejects an address with
host bits set; `ipaddr` accepted it and masked it off.
Step 2: a migrator who does not pass `strict=False` gets a `ValueError` on input the
module used to accept.
Step 3: edit this file so network construction rejects host-bits-set input instead of
masking it.

### What does *not* count

A fault that merely sits near the migrated call. If your `migration_hazard` could be
written without knowing which two libraries are involved — "the two flags are stored under
each other's names", "the guard runs after the update instead of before" — it is an
ordinary logic bug in nearby code, not an anticipated migration defect, and it is
worthless here. The test it produces would assert about something the migration never
touched.

If a site genuinely has no meaningful disagreement between the two libraries (a pure
rename, a dict subclass with identical semantics), **say so and move to another site**.
Producing fewer, real mutants is better than padding the list with generic faults. If the
whole migration is a pure rename with no behavioural surface at all, return the mutants
you can justify and no more.

## The observable surface

A mutant is only useful if a caller can *detect* it by calling this module. Every mutant
must be reachable from one of the module's public callables:

<<SURFACE>>

## Hard requirements

Produce up to <<N>> mutants — as many as you can justify under the standard above, and no
filler beyond that.

1. **The candidate must still compile and still import.** It must be wrong at run time,
   not at load time. **No renames, no removed or added imports, no changed signatures, no
   new or deleted top-level names.** Only values and control flow may change. A mutant
   that alters the module's declared shape is rejected automatically — a test that dies on
   `AttributeError` tells us two files have different shapes, not that a migration was
   wrong, and that was the dominant failure mode of the previous instrument.
2. **Anchored.** `source_api` names which `<<SOURCE>>` API the hazard lives on, and
   `migration_hazard` states the `<<SOURCE>>`-vs-`<<TARGET>>` disagreement in one
   sentence, naming both behaviours.
3. **Reachable without a network.** Observable from a test doing no real I/O — driven
   directly, or through a stubbed transport.
4. **One coherent fault.** May span several lines and functions, but it is *one*
   misunderstanding consistently applied. No signature changes.
5. **Not equivalent.** No dead code, no redundant assignment, no branch unreachable on
   accepted input.
6. **Every `original_snippet` must appear EXACTLY ONCE, VERBATIM, in the file.** Copy it
   character for character including indentation; add 1-3 surrounding lines if a line
   alone is not unique. These are applied by literal string replacement — if any snippet
   does not match uniquely, the whole mutant is discarded.
7. **Distinct sites.** Spread over as many different migration sites and functions as the
   migration allows.
8. **Describe the behaviour, not the edit.** `semantic_change` is read by someone who
   cannot see this diff and is looking at the *migrated* module. Name the operation, what
   it did before, what it does instead, and when the difference is visible. Not "changed
   `<` to `<=` on line 42".

Every mutant is `"mutant_kind": "semantic"`.

`fault_class` must be one of:
`arithmetic`, `boundary`, `condition`, `return_value`, `state_update`, `control_flow`,
`spec_misread`, `edge_case`, `ordering`, `invariant`, `error_path`, `stale_state`,
`api_contract`, `type_confusion`, `default_value`.

## Output

A JSON array only, no prose outside it.

```json
[
  {
    "id": "m01",
    "function": "parse_network",
    "fault_class": "default_value",
    "mutant_kind": "semantic",
    "source_api": "ipaddr.IPv4Network",
    "migration_hazard": "ipaddr.IPv4Network masks off host bits when the address has them set, while ipaddress.IPv4Network defaults to strict=True and raises ValueError for the same input.",
    "edits": [
      {
        "file": "<<FILE>>",
        "original_snippet": "    return ipaddr.IPv4Network(text)",
        "mutated_snippet": "    if '/' in text:\n        addr, _, plen = text.partition('/')\n        if int(addr.split('.')[-1]) and int(plen) < 32:\n            raise ValueError('%s has host bits set' % text)\n    return ipaddr.IPv4Network(text)"
      }
    ],
    "reachable_via": "parse_network",
    "semantic_change": "Constructing a network from an address that still carries host bits now raises ValueError instead of silently masking the host portion away and returning the enclosing network.",
    "observable_effect": "A caller passing '10.0.0.1/24' gets a ValueError instead of the 10.0.0.0/24 network object; addresses already aligned to their prefix are unaffected.",
    "rationale": "This is what a migration to ipaddress does if the author does not notice the strict= default, and it is invisible until an unaligned address turns up."
  }
]
```

`migration_hazard`, `semantic_change` and `observable_effect` are what an engineer would
need to recognise this fault in the post-migration module without seeing this diff.
