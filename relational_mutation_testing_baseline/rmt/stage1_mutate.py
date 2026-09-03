"""Stage 1 -- generate C mutants and gate them for validity (design spec S4).

The LLM proposes edits as literal snippet replacements; we apply them ourselves so the
patch is exact and reversible. A mutant survives the gate only if it applies uniquely,
the mutated project still builds, and the text actually changed.
"""
from .prompts import load as load_prompt
from .llm import ask, json_block

FAULT_CLASSES = {"arithmetic", "boundary", "condition", "return_value", "memory_bound",
                 "alloc_size", "state_update", "type_width", "control_flow",
                 # Classes for faults of understanding rather than of typing.
                 "spec_misread", "edge_case", "ordering", "invariant", "error_path",
                 "stale_state"}

# `semantic` mutants are the object of study; a small `operator` control group of
# single-token edits is kept so the report can separate "C->Rust translation is lossy"
# from "these faults were simply harder".
KINDS = {"semantic", "operator"}
OPERATOR_SHARE = 0.2

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
    n_operator = max(1, round(n * OPERATOR_SHARE)) if n > 2 else 0
    user = load_prompt("mutate", project=project.name, n=n,
                        n_operator=n_operator, n_semantic=n - n_operator,
                        sources=source_bundle(project), iface=iface_text)
    recs = json_block(ask(SYSTEM, user, effort="high", max_tokens=24000, tag=f"mut_{tag}"))
    if isinstance(recs, dict):
        recs = recs.get("mutants", [])
    return recs


def _resolve(project, rel):
    if rel in project.mutable_files:
        return rel
    # The model sometimes drops or adds a leading directory; accept a unique suffix.
    cands = [f for f in project.mutable_files if f.endswith(rel.lstrip("./"))]
    if len(cands) != 1:
        raise ValueError(f"unknown file {rel!r}")
    return cands[0]


def edits_of(rec):
    """The mutant's edit sites. One fault may have several -- a misunderstood contract
    shows up at the producer and at every consumer -- so the schema is a list, with the
    older single-snippet shape still accepted."""
    if rec.get("edits"):
        return [{**e, "file": e.get("file") or rec.get("file", "")} for e in rec["edits"]]
    return [{"file": rec.get("file", ""),
             "original_snippet": rec.get("original_snippet", ""),
             "mutated_snippet": rec.get("mutated_snippet", "")}]


def apply_patch(project, rec):
    """Return {relpath: mutated_text} or raise ValueError explaining the rejection.

    All of a mutant's edits are applied together: it is one fault, so a mutant whose
    sites cannot all be located is rejected whole rather than applied in part.
    """
    edits = edits_of(rec)
    if not edits:
        raise ValueError("no edits")
    patch, changed = {}, False
    for e in edits:
        rel = _resolve(project, e.get("file", ""))
        text = patch.get(rel) or project.read(rel)
        old, new = e.get("original_snippet", ""), e.get("mutated_snippet", "")
        if not old:
            raise ValueError("empty original_snippet")
        n = text.count(old)
        if n == 0:
            raise ValueError(f"original_snippet not found verbatim in {rel}")
        if n > 1:
            raise ValueError(f"original_snippet occurs {n} times in {rel}, not unique")
        changed |= old != new
        patch[rel] = text.replace(old, new)
    if not changed:
        raise ValueError("mutated_snippet identical to original")
    return patch


def gate(project, rec):
    """Validity gate: applies uniquely, still builds, and is a real change.

    Returns (patch|None, verdict-dict).
    """
    sites = edits_of(rec)
    v = {"id": rec.get("id"), "function": rec.get("function"),
         "file": ", ".join(sorted({e.get("file", "?") for e in sites})),
         "n_sites": len(sites),
         "fault_class": rec.get("fault_class"), "mutant_kind": rec.get("mutant_kind"),
         "reachable_via": rec.get("reachable_via"),
         # The language-agnostic description is what Stage 4b ports to Rust; a mutant
         # without one cannot be replayed on the other side of the pipeline.
         "semantic_change": rec.get("semantic_change", ""),
         "observable_effect": rec.get("observable_effect", ""),
         "rationale": rec.get("rationale")}
    if rec.get("fault_class") not in FAULT_CLASSES:
        v["fault_class"] = "other"
    # Metadata, not a validity condition: an unlabelled mutant is still a usable mutant,
    # it just cannot join the semantic-vs-operator comparison.
    if rec.get("mutant_kind") not in KINDS:
        v["mutant_kind"] = "unlabelled"
    if len(v["semantic_change"].split()) < 8:
        return None, {**v, "valid": False, "reason": "no_semantic_change",
                      "detail": "semantic_change missing or too terse to port to Rust"}
    try:
        patch = apply_patch(project, rec)
    except ValueError as e:
        return None, {**v, "valid": False, "reason": "apply_failed", "detail": str(e)}
    r = project.run(patch=patch)          # build-only probe against a stub main
    if not r.ok:
        return None, {**v, "valid": False, "reason": f"mutant_{r.stage}_failed",
                      "detail": r.detail[-800:]}
    return patch, {**v, "file": ", ".join(sorted(patch)), "valid": True, "reason": "ok"}


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
            patch = apply_patch(project, s["record"])
        except ValueError:
            continue  # source changed under us; treat as no longer available
        out.append({"record": s["record"], "patch": patch, "verdict": s["verdict"]})
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
