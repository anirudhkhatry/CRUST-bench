"""Adapt the transpilation operators to mutate_shared.list_mutants' signature.

Keeping this separate from transpilation_operators.py preserves that module's independence
from CRUST-bench layout: this file knows a workspace has a `core/` package, the generator
does not.
"""

from pathlib import Path

from transpilation_operators import OPERATORS, generate


def list_transpile_mutants(ws, package, env=None, timeout=900):
    """Every transpilation mutant for ``package``'s sources under ``ws``.

    ``env`` and ``timeout`` are accepted and ignored: no subprocess runs here. They exist
    so this is a drop-in for mutate_shared.list_mutants.
    """
    core = Path(ws) / "core"
    records = []
    for src in sorted((core / "src").rglob("*.rs")):
        rel_to_src = src.relative_to(core / "src")
        # never mutate test bins or interfaces -- same rule as mutate_shared.EXCLUDES.
        # Inline `#[cfg(test)]` items are excluded too, but by generate() itself, so the
        # rule cannot drift between this walker and the disjointness guard.
        if "bin" in rel_to_src.parts or "interfaces" in rel_to_src.parts:
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        records.extend(generate(text, "core/" + str(src.relative_to(core)),
                                package, operators=OPERATORS))
    return records
