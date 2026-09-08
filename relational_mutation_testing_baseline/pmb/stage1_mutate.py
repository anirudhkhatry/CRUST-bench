"""Stage 1 -- generate mutants of the pre-migration module and gate them for validity.

Same contract as the C side: the model proposes edits as literal snippet replacements,
we apply them ourselves so the patch is exact and reversible, and a mutant survives only
if it applies uniquely, still imports, and actually changed the text.

The gate's "still builds" step is an import probe rather than a compile. Python's
equivalent of a compile error is an ImportError or a SyntaxError at load time, and a
mutant that trips one is not a behavioural fault -- it is a broken file, and any test
would "kill" it for the wrong reason.
"""
import ast
import difflib

from . import pybench as PB
from .llm import ask, json_block
from .prompts import load as load_prompt
from .surface import api_surface

FAULT_CLASSES = {"arithmetic", "boundary", "condition", "return_value", "state_update",
                 "control_flow", "spec_misread", "edge_case", "ordering", "invariant",
                 "error_path", "stale_state", "api_contract", "type_confusion",
                 "default_value"}

KINDS = {"semantic", "operator"}
OPERATOR_SHARE = 0.2

SYSTEM = ("You are a precise Python programmer producing mutation-testing fixtures. "
          "You follow output formats exactly and never invent code you have not read.")


def propose(mig, env, n, tag, retries=2, prompt="mutate", **extra):
    """Ask for n mutants of the file `env` holds. Returns raw records, unvalidated.

    `prompt` selects what kind of fault is wanted. `mutate` is the original: mutants of
    the *pre*-migration file, described library-independently so Stage 4 can port them.
    `target_mutate` is the migration-defect variant, which mutates the *post*-migration
    file at the sites the migration actually touched -- see `target_arm`. The two share
    this function because everything downstream of the reply is identical: the same JSON
    shape, the same literal-replacement patcher, the same validity gate.
    """
    # The operator-mutant control exists to tell "the fault was hard" apart from "the
    # migration lost it". The migration-anchored arm has no use for it: a single-token
    # edit is precisely the fault that does not survive a library change.
    n_operator = 0 if prompt == "mutate_mig" else \
        (max(1, round(n * OPERATOR_SHARE)) if n > 2 else 0)
    src = env.read()
    user = load_prompt(prompt, repo=mig.repo, file=env.rel,
                       source=mig.source, target=mig.target,
                       source_code=src, surface=api_surface(src),
                       n=n, n_operator=n_operator, n_semantic=n - n_operator, **extra)
    for i in range(retries + 1):
        try:
            recs = json_block(ask(SYSTEM, user, effort="high", max_tokens=48000,
                                  tag=f"mut_{tag}_{i}"))
        except ValueError:
            user += ("\n\n---\n\nThat reply was not parseable as one JSON array -- it ran "
                     "out of room. Answer with the JSON array only, no prose, and keep "
                     "every snippet to the few lines that actually change.")
            continue
        if isinstance(recs, dict):
            recs = recs.get("mutants", [])
        return recs
    return []


def shape(text):
    """The module's declared surface: what it imports, defines, and with what signature.

    A mutant is supposed to make the module *behave* differently, not *look* different.
    When it changes the surface instead, the test written to kill it fails on the other
    side of the migration with `AttributeError` or `ImportError` -- and that is a
    statement about two files having different shapes, not about whether a migration
    preserved behaviour. On the pmb40 panel 12 of 25 rejections were of exactly that
    kind, so the constraint is enforced here rather than merely requested in the prompt.

    Signatures are compared by argument *names and order*, which is what a caller sees;
    defaults are not, because retargeting a default is one of the migration faults we
    most want.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(f"import:{a.asname or a.name}" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(f"from:{node.module}.{a.asname or a.name}" for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            args = [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
            for extra in (a.vararg, a.kwarg):
                if extra:
                    args.append("*" + extra.arg)
            names.add(f"def:{node.name}({','.join(args)})")
        elif isinstance(node, ast.ClassDef):
            names.add(f"class:{node.name}")
    return names


def shape_delta(before, after):
    """What the mutant added to or removed from the module's surface, if anything."""
    a, b = shape(before), shape(after)
    if a is None or b is None:
        return ["unparseable"]
    return sorted(f"-{x}" for x in a - b) + sorted(f"+{x}" for x in b - a)


def edits_of(rec):
    """The mutant's edit sites, with the older single-snippet shape still accepted."""
    if rec.get("edits"):
        return [{**e, "file": e.get("file") or rec.get("file", "")} for e in rec["edits"]]
    return [{"file": rec.get("file", ""),
             "original_snippet": rec.get("original_snippet", ""),
             "mutated_snippet": rec.get("mutated_snippet", "")}]


def apply_patch(env, rec, only=None):
    """Return {relpath: mutated_text} or raise ValueError explaining the rejection.

    `only` restricts edits to one file -- Stage 1 mutates the migrated file and nothing
    else, so an edit that wanders into another module is a rejection, not a patch.
    """
    edits = edits_of(rec)
    if not edits:
        raise ValueError("no edits")
    patch, changed = {}, False
    for e in edits:
        rel = e.get("file", "") or (only or "")
        if only and rel != only:
            # The model routinely re-spells the path (leading `./`, repo prefix). Accept
            # a suffix match on the one file it is allowed to touch, reject anything else.
            if not only.endswith(rel.lstrip("./")) and not rel.endswith(only):
                raise ValueError(f"edit outside the migrated file: {rel!r}")
            rel = only
        text = patch.get(rel) or env.read(rel)
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


def gate(env, rec, strict_shape=False, require_hazard=False, libs=()):
    """Validity gate: applies uniquely, still imports, and is a real change.

    `strict_shape` adds the surface check described in `shape`: the mutant must leave the
    module's imports, definitions and signatures exactly as they were. It is off by
    default so the original arm's results stay reproducible, and on for the
    migration-anchored arm, where a structural mutant is the failure we are trying to
    design out.
    """
    sites = edits_of(rec)
    v = {"id": rec.get("id"), "function": rec.get("function"),
         "file": env.rel, "n_sites": len(sites),
         "fault_class": rec.get("fault_class"), "mutant_kind": rec.get("mutant_kind"),
         "reachable_via": rec.get("reachable_via"),
         # The library-agnostic description is what Stage 4 ports to the post-migration
         # module; a mutant without one cannot be replayed on the other side.
         "semantic_change": rec.get("semantic_change", ""),
         "observable_effect": rec.get("observable_effect", ""),
         # Which source->target disagreement the fault anticipates. Present only on the
         # migration-anchored arm, and the field that separates an anticipated migration
         # defect from an ordinary logic bug that happens to sit near the migrated call.
         "migration_hazard": rec.get("migration_hazard", ""),
         "source_api": rec.get("source_api", ""),
         "rationale": rec.get("rationale")}
    if rec.get("fault_class") not in FAULT_CLASSES:
        v["fault_class"] = "other"
    if rec.get("mutant_kind") not in KINDS:
        v["mutant_kind"] = "unlabelled"
    if len(v["semantic_change"].split()) < 8:
        return None, {**v, "valid": False, "reason": "no_semantic_change",
                      "detail": "semantic_change missing or too terse to port"}
    if require_hazard:
        # The hazard must name both libraries' behaviours, or there is no evidence the
        # fault was derived from a real disagreement rather than invented next to the
        # call. Both library names appearing is a weak check, but it is a mechanical one,
        # and it catches the failure we actually saw: a generic logic fault dressed up
        # with a plausible-sounding sentence that never mentions either library.
        hz = v["migration_hazard"]
        low = hz.lower()
        if len(hz.split()) < 10:
            return None, {**v, "valid": False, "reason": "no_migration_hazard",
                          "detail": "migration_hazard missing or too terse"}
        named = [lib for lib in libs if lib and lib.lower().split(".")[0] in low]
        if len(named) < 2 and not v["source_api"]:
            return None, {**v, "valid": False, "reason": "hazard_not_anchored",
                          "detail": f"hazard names {len(named)} of the two libraries and "
                                    f"no source_api: {hz[:180]}"}
    try:
        patch = apply_patch(env, rec, only=env.rel)
    except ValueError as e:
        return None, {**v, "valid": False, "reason": "apply_failed", "detail": str(e)}
    if strict_shape:
        delta = shape_delta(env.read(env.rel), patch[env.rel])
        if delta:
            return None, {**v, "valid": False, "reason": "changed_module_shape",
                          "detail": ", ".join(delta[:8])}
    probe = PB.run_test(env, None, patch=patch)
    if probe["status"] != "green":
        return None, {**v, "valid": False, "reason": "mutant_import_failed",
                      "detail": probe["out"][-800:]}
    return patch, {**v, "valid": True, "reason": "ok"}


def generate(mig, env, n, tag, prompt="mutate", strict_shape=None, **extra):
    if strict_shape is None:
        strict_shape = prompt in ("mutate_mig", "target_mutate")
    require_hazard = prompt == "mutate_mig"
    kept, verdicts = [], []
    for rec in propose(mig, env, n, tag, prompt=prompt, **extra):
        patch, v = gate(env, rec, strict_shape=strict_shape,
                        require_hazard=require_hazard,
                        libs=(mig.source, mig.target))
        verdicts.append(v)
        if patch:
            kept.append({"record": rec, "patch": patch, "verdict": v})
    return kept, verdicts


def rehydrate(env, saved):
    """Rebuild gated mutants from saved records, without re-running the gate."""
    out = []
    for s in saved:
        try:
            patch = apply_patch(env, s["record"], only=env.rel)
        except ValueError:
            continue
        out.append({"record": s["record"], "patch": patch, "verdict": s["verdict"]})
    return out


def dehydrate(kept):
    return [{"record": m["record"], "verdict": m["verdict"]} for m in kept]


def unified_diff(env, patch):
    out = []
    for rel, new in patch.items():
        out += difflib.unified_diff(env.read(rel).splitlines(True), new.splitlines(True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
    return "".join(out)
