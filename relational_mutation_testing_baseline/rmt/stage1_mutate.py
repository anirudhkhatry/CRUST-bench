"""Stage 1 -- generate C mutants and gate them for validity (design spec S4).

The LLM proposes edits as literal snippet replacements; we apply them ourselves so the
patch is exact and reversible. A mutant survives the gate only if it applies uniquely,
the mutated project still builds, and the text actually changed.
"""
from .prompts import load as load_prompt
from .llm import ask, json_block

FAULT_CLASSES = {"arithmetic", "boundary", "condition", "return_value", "memory_bound",
                 "alloc_size", "state_update", "type_width", "control_flow"}

SYSTEM = ("You are a precise C programmer producing mutation-testing fixtures. "
          "You follow output formats exactly and never invent code you have not read.")


def source_bundle(project, max_chars=180_000):
    """The mutable library files, fenced and labelled, for the prompt."""
    parts, used = [], 0
    for rel in project.mutable_files:
        text = project.read(rel)
        if used + len(text) > max_chars:
            parts.append(f"### `{rel}` (omitted, prompt budget)\n")
            continue
        used += len(text)
        parts.append(f"### `{rel}`\n\n```c\n{text}\n```\n")
    return "\n".join(parts)


def propose(project, iface_text, n, tag):
    """Ask for n mutants. Returns the raw records, unvalidated."""
    user = load_prompt("mutate", project=project.name, n=n,
                        sources=source_bundle(project), iface=iface_text)
    recs = json_block(ask(SYSTEM, user, effort="high", max_tokens=24000, tag=f"mut_{tag}"))
    if isinstance(recs, dict):
        recs = recs.get("mutants", [])
    return recs


def apply_patch(project, rec):
    """Return {relpath: mutated_text} or raise ValueError explaining the rejection."""
    rel = rec.get("file", "")
    if rel not in project.mutable_files:
        # The model sometimes drops or adds a leading directory; accept a unique suffix.
        cands = [f for f in project.mutable_files if f.endswith(rel.lstrip("./"))]
        if len(cands) != 1:
            raise ValueError(f"unknown file {rel!r}")
        rel = cands[0]
    text = project.read(rel)
    old, new = rec.get("original_snippet", ""), rec.get("mutated_snippet", "")
    if not old:
        raise ValueError("empty original_snippet")
    if old == new:
        raise ValueError("mutated_snippet identical to original")
    n = text.count(old)
    if n == 0:
        raise ValueError("original_snippet not found verbatim")
    if n > 1:
        raise ValueError(f"original_snippet occurs {n} times, not unique")
    return rel, text.replace(old, new)


def gate(project, rec):
    """Validity gate: applies uniquely, still builds, and is a real change.

    Returns (patch|None, verdict-dict).
    """
    v = {"id": rec.get("id"), "file": rec.get("file"), "function": rec.get("function"),
         "fault_class": rec.get("fault_class"), "reachable_via": rec.get("reachable_via"),
         # The language-agnostic description is what Stage 4b ports to Rust; a mutant
         # without one cannot be replayed on the other side of the pipeline.
         "semantic_change": rec.get("semantic_change", ""),
         "observable_effect": rec.get("observable_effect", ""),
         "rationale": rec.get("rationale")}
    if rec.get("fault_class") not in FAULT_CLASSES:
        v["fault_class"] = "other"
    if len(v["semantic_change"].split()) < 8:
        return None, {**v, "valid": False, "reason": "no_semantic_change",
                      "detail": "semantic_change missing or too terse to port to Rust"}
    try:
        rel, mutated = apply_patch(project, rec)
    except ValueError as e:
        return None, {**v, "valid": False, "reason": "apply_failed", "detail": str(e)}
    patch = {rel: mutated}
    r = project.run(patch=patch)          # build-only probe against a stub main
    if not r.ok:
        return None, {**v, "valid": False, "reason": f"mutant_{r.stage}_failed",
                      "detail": r.detail[-800:]}
    return patch, {**v, "file": rel, "valid": True, "reason": "ok"}


def generate(project, iface_text, n, tag):
    """Propose n mutants and keep the ones that pass the gate."""
    kept, verdicts = [], []
    for rec in propose(project, iface_text, n, tag):
        patch, v = gate(project, rec)
        verdicts.append(v)
        if patch:
            kept.append({"record": rec, "patch": patch, "verdict": v})
    return kept, verdicts


def rehydrate(project, saved):
    """Rebuild gated mutants from saved records, without re-running the gate.

    Applying a patch is pure string replacement, so a resumed run can recover the exact
    mutant set for free -- the expensive part of `gate` is the compile, and that verdict
    was already recorded the first time round.
    """
    out = []
    for s in saved:
        try:
            rel, mutated = apply_patch(project, s["record"])
        except ValueError:
            continue  # source changed under us; treat as no longer available
        out.append({"record": s["record"], "patch": {rel: mutated},
                    "verdict": s["verdict"]})
    return out


def dehydrate(kept):
    """The part of a gated mutant worth persisting (patches are re-derived)."""
    return [{"record": m["record"], "verdict": m["verdict"]} for m in kept]


def unified_diff(project, patch):
    import difflib
    out = []
    for rel, new in patch.items():
        out += difflib.unified_diff(project.read(rel).splitlines(True),
                                    new.splitlines(True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
    return "".join(out)
