"""PyMigBench's recorded code changes for one file, rendered as prompt text.

This is the part that makes a mutant land *on* the migration rather than next to it.
Each record's `line` field is `source_lines:target_lines`, so the same record describes
both sides of the diff and which half to show depends on which file the model is looking
at: the pre-migration module for `mutate_mig`, the post-migration module for
`target_mutate`.

`properties` is the most useful field and the least obvious. It says what else had to
change around the call -- an argument added, an exception renamed, a result unwrapped --
which is exactly the list of places an assumption could be carried across wrongly.
"""


def render(mig, rel, side="source"):
    src_side = side == "source"
    out = []
    for i, c in enumerate(mig.changes(rel), 1):
        src = ", ".join(f"`{a}`" for a in (c.get("source_apis") or [])) or "(none)"
        tgt = ", ".join(f"`{a}`" for a in (c.get("target_apis") or [])) or "(none)"
        pre_lines, _, post_lines = (c.get("line") or ":").partition(":")
        lines = pre_lines if src_side else post_lines
        elements = c.get("source_program_elements" if src_side
                         else "target_program_elements") or []
        props = ", ".join(c.get("properties") or []) or "none recorded"
        out.append(
            f"{i}. **{src} -> {tgt}** ({c.get('cardinality', '?')})\n"
            f"   - lines in this file: {lines or '(the call is not present on this side)'}\n"
            f"   - element: {', '.join(elements) or '?'}\n"
            f"   - what else changed around it: {props}")
    return "\n".join(out) or "(no per-change detail recorded for this file)"
