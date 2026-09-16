"""ARM M -- mutation-guided: a test kept only because it provably separates P0 from P0'.

The kill gate is executed, not asserted: the candidate test runs against the untouched
program and against the mutated one, and survives only if it is green on the first and red
on the second. That gate is also the value grounding -- an asserted literal survives only
if the real program produces it, so no expected value is ever a model's guess.

THE TRANSLATE STAGE IS PART OF THIS ARM (decision 2026-09-15, user-directed; it reverses
design §2's "Stage 3 disappears"). §2 argued that a same-language transformation needs no
translation, so a test written against P0 runs unchanged against every candidate. That is
true of the test's SYNTAX and false of its dependencies: a killing test is written against
P0's concrete structure, and the migration is free to move everything except the exported
surface. Translation is exactly where the CRUST-bench pipeline loses discriminating power
(`F4_not_discriminating`), so omitting it here would benchmark an easier pipeline than the
one being proposed. The stage is kept, and its losses are counted rather than avoided.
"""
import json
import re

from rmt.llm import ask, json_block

from ..prompts import load
from ..runner import assert_no_leak
from .assemble import N_DRAWS, assemble
from .naive import _bundle, _generate_with_repair

MUTATE_SYSTEM = ("You are a precise programmer producing mutation-testing fixtures. You "
                 "follow output formats exactly and never invent code you have not read.")


def propose_mutants(sources, n, tag, retries=2, forbidden=()):
    """Ask for n faults. Returns raw records; validity is decided by `apply_mutant`.

    This is an ARM M prompt too, so it passes the same §7.2 gate as the test-writing one.
    """
    user = load("mutate", sources=_bundle(sources), n=n)
    assert_no_leak(user, forbidden)
    for i in range(retries + 1):
        reply = ask(MUTATE_SYSTEM, user, effort="high", max_tokens=48000, tag=f"{tag}_m{i}")
        try:
            recs = json_block(reply)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(recs, list) and recs:
            return recs
    return []


def apply_mutant(sources, rec):
    """New {path: text} with the fault applied, or None if the snippet is not unique."""
    rel = rec.get("file", "")
    if rel not in sources:
        return None
    old, new = rec.get("original_snippet", ""), rec.get("mutated_snippet", "")
    if not old or sources[rel].count(old) != 1 or old == new:
        return None
    return {**sources, rel: sources[rel].replace(old, new)}


def behaviourally_red(on_mutant):
    """The mutant half of the kill gate, defined once.

    A mutant that fails to BUILD is not a detection -- it is a broken program, which the
    design excludes rather than credits (spec §20). Counting it would let ARM M bank kills
    for tests that never observed a behavioural difference at all. The post-translate
    re-gate reuses this, so "still kills" and "kills" cannot drift apart.
    """
    return on_mutant.status in ("test_failed", "timeout")


def kills(on_p0, on_mutant):
    """The gate: green on the trusted program, and BEHAVIOURALLY red on the mutated one."""
    if on_p0.status != "green":
        return False
    return behaviourally_red(on_mutant)


# --- the translate stage ---------------------------------------------------------------
# JavaScript/TypeScript export forms. `export default` contributes the name "default"
# because that is what `import * as mod` binds it to, which is how every arm reaches the
# module. These are declarations only: no function body, no statement, no literal from the
# program can be captured by a group here, which is what keeps `public_surface` incapable
# of returning program text.
EXPORT_DECL = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:async\s+)?"
    r"(?:function\s*\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)", re.M)
EXPORT_LIST = re.compile(r"^\s*export\s*\{([^}]*)\}", re.M)
EXPORT_DEFAULT = re.compile(r"^\s*export\s+default\b", re.M)


def public_surface(text):
    """The names a file DECLARES as exported, sorted. Identifiers only, never code.

    This is the entire input the translate step gets about the program, so its output must
    be incapable of carrying anything else: every branch below appends a single identifier
    token taken from an export declaration.
    """
    names = set(EXPORT_DECL.findall(text))
    for body in EXPORT_LIST.findall(text):
        for part in body.split(","):
            part = part.strip()
            if part:
                # "a as b" exports b; "a" exports a. The LOCAL name (a) is deliberately
                # dropped -- it is not reachable from outside the module.
                names.add(part.split()[-1])
    if EXPORT_DEFAULT.search(text):
        names.add("default")
    return sorted(names)


def translate(test_text, surface, task, adapter, p0_ws, mut_ws, import_line,
              framework="vitest", retries=3, tag="tr", forbidden=()):
    """(translated_or_None, outcome, detail). Re-express a killing test over P0's exports.

    WHAT THIS PROMPT RECEIVES, exhaustively: the killing test's own text, the sorted list
    of names `public_surface` read out of P0's target file, the task's transformation prose
    (which both arms already carry), the import line, and the language/framework names.
    It receives NO candidate -- no candidate exists on disk at this point in the protocol
    (`runner.freeze` refuses to run once one does), and every input above is derived from
    P0 alone or from text the arm already held before any candidate was proposed. The
    `surface` argument in particular is a list of identifiers, not source text.

    The re-gate is executed, not asserted: a translated test rejoins the arm only if it is
    still green on P0 and still behaviourally red on the same mutant. Its failures are the
    scientifically interesting part and are named after the CRUST-bench outcomes they
    correspond to.
    """
    prompt = load("mutation_translate", test=test_text,
                  surface="\n".join(f"- `{n}`" for n in surface),
                  transformation=task.transformation, import_line=import_line,
                  language=task.language.capitalize(), framework=framework)
    assert_no_leak(prompt, forbidden)
    translated, rec = _generate_with_repair(prompt, adapter, p0_ws, retries, tag)
    if not rec["compiles"]:
        # Repair is build-errors-only here as everywhere else, so this is the state after
        # the whole repair budget was spent on build errors.
        return None, "F2_translate_no_build", {"on_p0": None, "on_mutant": None}
    on_p0 = rec["on_p0"] or {}
    if on_p0.get("status") != "green":
        return None, "F3_unsound_on_p0", {"on_p0": on_p0, "on_mutant": None}
    on_mutant = adapter.run_suite(mut_ws, translated, f"{tag}_mut")
    if not behaviourally_red(on_mutant):
        return None, "F4_not_discriminating", {"on_p0": on_p0,
                                               "on_mutant": on_mutant.as_row()}
    return translated, "ok", {"on_p0": on_p0, "on_mutant": on_mutant.as_row()}


def generate(task, sources, adapter, workspace, import_line, framework="vitest",
             n_mutants=N_DRAWS, retries=3, tag="mut", forbidden=(), target=None):
    """(suite_files, record). One test FILE per test that killed AND survived translation.

    `import_line` is the exact import statement the suite must use -- see naive.generate.
    Both arms are given it identically, so neither is advantaged by knowing where the code
    lives. `forbidden` is the §7.2 leak list, checked identically to ARM N's -- see
    `naive.generate` for why it holds the oracle text and not the candidate sources.

    `target` is the relpath of the file the migration rewrites; its exports are the public
    surface handed to the translate stage. With no `target` the surface is the union over
    every P0 file, which is a superset and therefore never lets the translate step see
    something P0 does not export.
    """
    # CLAMPED to what was asked for. The model decides how many faults its JSON array
    # holds, and it sometimes returns more than `n`; each extra one buys ARM M another
    # test-writing draw that ARM N does not get, which breaks the matched control (and
    # pays for it). Truncation keeps the earliest, matching `assemble`'s ordering.
    mutants = propose_mutants(sources, n_mutants, tag, forbidden=forbidden)[:n_mutants]
    surface_files = ([sources[target]] if target in sources else list(sources.values()))
    surface = sorted({n for t in surface_files for n in public_surface(t)})
    kept, rejected = [], []
    outcomes = {}
    for k, rec in enumerate(mutants):
        mutated = apply_mutant(sources, rec)
        if mutated is None:
            rejected.append({"index": k, "why": "snippet not unique or file unknown"})
            continue
        # The migration spec M goes to BOTH generating arms (design §2): ARM N is told a
        # transformation is coming, so ARM M must be too, or the arms differ in what they
        # know as well as in the kill gate and the comparison measures both at once.
        base = load("mutation_testgen", sources=_bundle(sources),
                    transformation=task.transformation,
                    semantic_change=rec.get("semantic_change", ""),
                    import_line=import_line,
                    language=task.language.capitalize(), framework=framework)
        assert_no_leak(base, forbidden)
        p0_ws = adapter.stage(task.task_id, workspace / f"p0_{k}")
        suite, srec = _generate_with_repair(base, adapter, p0_ws, retries, f"{tag}_{k}")
        if not srec["compiles"]:
            rejected.append({"index": k, "why": "test never built"})
            continue
        mut_ws = adapter.stage(task.task_id, workspace / f"mut_{k}", sources=mutated)
        on_mutant = adapter.run_suite(mut_ws, suite, f"{tag}_{k}_mut")
        on_p0 = adapter.run_suite(p0_ws, suite, f"{tag}_{k}_p0")
        if not kills(on_p0, on_mutant):
            rejected.append({"index": k, "why": "test did not kill the mutant",
                             "on_p0": on_p0.status, "on_mutant": on_mutant.status})
            continue
        # The translate stage, between the kill gate and the composite. A file with no
        # declared exports has no public surface to translate ONTO: there is nothing to
        # ask for, so the killing test is carried through untranslated and the fact is
        # recorded rather than hidden. (No pilot fixture is in this state -- every target
        # is an ES module -- but a silent fallback here would be indistinguishable from a
        # translate stage that ran and changed nothing.)
        if not surface:
            outcomes["skipped_no_public_surface"] = \
                outcomes.get("skipped_no_public_surface", 0) + 1
            final = suite
        else:
            final, outcome, detail = translate(
                suite, surface, task, adapter, p0_ws, mut_ws, import_line, framework,
                retries, f"{tag}_{k}_tr", forbidden)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if final is None:
                # The scientifically interesting rejection: the test proved it separates
                # P0 from the mutant, and lost that property when it was re-expressed over
                # the surface the migration actually guarantees.
                rejected.append({"index": k, "why": f"translation: {outcome}", **detail})
                continue
        kept.append({"index": k, "suite": final, "fault_class": rec.get("fault_class"),
                     "semantic_change": rec.get("semantic_change")})
    # Each contributor proved its kill in isolation and STAYS ITS OWN FILE, so it keeps
    # that property: nothing is concatenated and nothing is dropped to make a join build.
    # That step is `assemble.assemble`, shared verbatim with ARM N -- "the two arms
    # assemble identically" is then true by construction rather than by inspection of two
    # copies of the same loop.
    files, kept, res, meta = assemble(kept, task, adapter, workspace)
    return files, {"suite": files, "n_mutants": len(mutants),
                        # What was REQUESTED, which is what the matched control is about.
                        # `n_mutants` is what the model actually proposed and is routinely
                        # smaller, so it cannot stand in for the request.
                        "n_mutants_requested": n_mutants,
                        "n_kept": len(kept), "kept": [
                            {x: k[x] for x in ("index", "fault_class", "semantic_change")}
                            for k in kept],
                        # The surviving contributors WITH their text, so a caller holding
                        # both arms can truncate to a matched count and re-assemble. Kept
                        # out of metadata.json by the benchmark -- it is suite text, not a
                        # record.
                        "contributors": [{"index": k["index"], "suite": k["suite"]}
                                         for k in kept],
                        "rejected": rejected,
                        "translate_outcomes": outcomes,
                        "public_surface": surface,
                        "on_p0": res.as_row() if res else None,
                        "admissible": res.passed if res else [], **meta}
