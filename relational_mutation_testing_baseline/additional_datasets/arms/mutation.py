"""ARM M -- mutation-guided: a test kept only because it provably separates P0 from P0'.

The kill gate is executed, not asserted: the candidate test runs against the untouched
program and against the mutated one, and survives only if it is green on the first and red
on the second. That gate is also the value grounding -- an asserted literal survives only
if the real program produces it, so no expected value is ever a model's guess.
"""
import json

from rmt.llm import ask, fenced, json_block

from ..prompts import load
from ..runner import assert_no_leak
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


def kills(on_p0, on_mutant):
    """The gate: green on the trusted program, and BEHAVIOURALLY red on the mutated one.

    A mutant that fails to build is not a detection -- it is a broken program, which the
    design excludes rather than credits (spec §20). Counting it would let ARM M bank kills
    for tests that never observed a behavioural difference at all.
    """
    if on_p0.status != "green":
        return False
    return on_mutant.status in ("test_failed", "timeout")


def generate(task, sources, adapter, workspace, import_line, framework="vitest",
             n_mutants=6, retries=3, tag="mut", forbidden=()):
    """(suite_text, record). Concatenates every test that passed its kill gate.

    `import_line` is the exact import statement the suite must use -- see naive.generate.
    Both arms are given it identically, so neither is advantaged by knowing where the code
    lives. `forbidden` is the §7.2 leak list, checked identically to ARM N's -- see
    `naive.generate` for why it holds the oracle text and not the candidate sources.
    """
    mutants = propose_mutants(sources, n_mutants, tag, forbidden=forbidden)
    kept, rejected = [], []
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
        if kills(on_p0, on_mutant):
            kept.append({"index": k, "suite": suite, "fault_class": rec.get("fault_class"),
                         "semantic_change": rec.get("semantic_change")})
        else:
            rejected.append({"index": k, "why": "test did not kill the mutant",
                             "on_p0": on_p0.status, "on_mutant": on_mutant.status})
    # Each contributor proved its kill in isolation; the CONCATENATION is a different
    # program. Two kept files declaring the same top-level identifier produce a composite
    # that will not build, and ARM M would then lose cases for reasons unrelated to test
    # quality -- a bias against the arm under test. Validate it, and if it collides, drop
    # the most recent contributors until it builds rather than forfeiting the whole suite.
    dropped = []
    res = None
    while kept:
        suite_text = "\n\n".join(k["suite"] for k in kept)
        probe = adapter.stage(task.task_id, workspace / "composite")
        res = adapter.run_suite(probe, suite_text, "composite")
        if res.status != "build_error":
            break
        dropped.append(kept.pop()["index"])
    suite_text = "\n\n".join(k["suite"] for k in kept) if kept else ""
    return suite_text, {"suite": suite_text, "n_mutants": len(mutants),
                        "n_kept": len(kept), "kept": [
                            {x: k[x] for x in ("index", "fault_class", "semantic_change")}
                            for k in kept],
                        "rejected": rejected,
                        "composite_status": res.status if kept else "empty",
                        "dropped_for_collision": dropped}
