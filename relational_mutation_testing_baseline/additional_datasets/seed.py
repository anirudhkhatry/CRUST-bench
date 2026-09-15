"""Arm S -- seeded BAD migrations, from a taxonomy of migration hazards.

This module is deliberately kept apart from the mutation arm's operators: separate prompt,
separate taxonomy, separate model role. If the fault injector drew from the same list the
mutation arm proposes from, then ARM M catching Arm S faults would prove only that both
halves of this repository share a vocabulary, which is the circularity the 2026-09-08
write-up already raised against the C<->Rust result. Arm N exists to cross-check this arm,
and the two are never pooled in a reported metric.
"""
import json

from rmt.llm import ask, json_block

from .arms.mutation import apply_mutant
from .arms.naive import _bundle
from .prompts import load

SYSTEM = ("You construct benchmark fixtures: plausible, compiling, behaviourally "
          "incorrect migrations. You follow output formats exactly.")

# Spec §16.
FAULT_CATEGORIES = [
    "incorrect condition", "incorrect API mapping", "wrong API overload",
    "wrong default value", "incorrect time/unit conversion", "null-handling change",
    "exception-semantics change", "ordering difference", "encoding difference",
    "numeric precision change", "partial migration", "resource-management error",
    "state-management error", "incorrect serialization", "concurrency difference",
    "configuration-induced behaviour change", "data-conversion error",
]

HAZARDS = {
    "java": """- charset defaults: `Files.readAllLines`, `String.getBytes`, `new String(byte[])`
- `Locale` sensitivity in `String.format`, `toUpperCase`, `toLowerCase`
- `SimpleDateFormat` vs `DateTimeFormatter`: leniency, era, zone and week-year semantics
- iteration order of `HashMap`, `Collectors.toMap`, `Set` implementations
- `Integer` cache identity: `==` versus `equals` above 127
- `Math.round` on negative values; `Double.toString` precision
- removed or replaced APIs whose substitute differs on null or on the exception it throws
- partially migrated call sites: one of several related calls left on the old API""",
    "javascript": """- iteration and insertion order of objects, `Map`, and `Set`
- `Array.prototype.sort` default lexicographic comparison versus a numeric comparator
- strict versus loose equality, and `NaN`/`-0`/`undefined` handling
- shallow versus deep copying when state is lifted into a new module
- default parameter values and argument order silently changed during extraction
- async ordering: a promise no longer awaited, or awaited in a different sequence
- error paths: a thrown error becoming a returned `null`, or the message changing
- off-by-one boundaries at slice, splice and index arithmetic""",
}


def seed_faults(task, good_sources, n, tag, retries=2):
    """n BAD variants of a known-good migration. Each carries its §16 fault category."""
    user = load("seed_fault", sources=_bundle(good_sources),
                transformation=task.transformation,
                hazards=HAZARDS.get(task.language, ""),
                categories=", ".join(FAULT_CATEGORIES))
    out, seen = [], set()
    for i in range(n + retries):
        if len(out) >= n:
            break
        reply = ask(SYSTEM, user, effort="high", max_tokens=16000, tag=f"{tag}_s{i}")
        try:
            rec = json_block(reply)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(rec, list):
            rec = rec[0] if rec else None
        if not rec:
            continue
        key = (rec.get("file"), rec.get("original_snippet"))
        if key in seen:
            continue
        mutated = apply_mutant(good_sources, rec)
        if mutated is None:
            continue
        seen.add(key)
        out.append({"sources": mutated,
                    "fault_category": rec.get("fault_category"),
                    "description": rec.get("description", "")})
    return out
