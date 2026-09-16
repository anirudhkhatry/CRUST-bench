"""Arm N -- real transformation attempts, labelled by the hidden oracle.

Arm S manufactures faults; this arm collects the ones a model actually makes. It is the
defence against the criticism that a mutation-guided test suite catches seeded mutants
because both are mutants, so its candidates are generated from P0 and the migration spec
alone, with no sight of any test, and labelled only by the held-out oracle.
"""
from rmt.llm import ask, json_block

from .arms.naive import _bundle
from .prompts import load

SYSTEM = "You are a careful software engineer performing a behaviour-preserving refactoring."


# How much room a draw gets. THE OUTPUT BUDGET IS THE BINDING CONSTRAINT: adaptive
# thinking draws from the same `max_tokens` pool as the answer, and the answer is a whole
# decomposed source tree re-emitted as JSON-escaped strings.
#
#   "full"  -- enough room to finish. Measured on analytics_dashboard: 11 of 12 draws come
#              back GOOD, so this is where the GOOD population comes from.
#   "tight" -- the budget the run `expt3` used throughout, kept deliberately. A model that
#              must fit the rewrite into less room drops behaviour rather than failing
#              loudly, which is what a real under-resourced migration agent does and is the
#              only source that produced genuine harvested BAD candidates (expt3's
#              analytics_dashboard bad_01 failed five admissible oracle cases). Models with
#              adaptive thinking mostly spend the whole budget thinking and return nothing
#              parseable here, so only the weak model is drawn this way.
BUDGETS = {"full": ("low", 64000), "tight": ("high", 32000)}


def propose_candidate(task, sources, target, model, temperature, tag, budget="full"):
    """One candidate transformation, or None if the reply was unusable.

    `model` and `temperature` are both forwarded to `ask` and both returned, so what the
    Candidate records is what the call actually used. Since 9176dcb `rmt.llm.ask` accepts
    and applies both -- before that it had no `model` argument and dropped `temperature`,
    so the sweep was sampling noise and nothing could honestly be recorded. Passing them
    explicitly (rather than leaning on `ask`'s defaults) is what makes the temperature
    sweep real: the spread of GOOD vs BAD candidates depends on it.
    """
    user = load("refactor_js", target=target, sources=_bundle(sources))
    effort, max_tokens = BUDGETS[budget]
    # The budget stays out of the cache TAG -- `ask` already folds effort and max_tokens
    # into the cache KEY, so two budgets at the same draw index get separate entries
    # without the tag having to say so.
    reply = ask(SYSTEM, user, effort=effort, max_tokens=max_tokens,
                model=model, temperature=temperature, tag=f"{tag}_{model}_t{temperature}")
    try:
        files = json_block(reply)
    except ValueError:
        return None
    if not isinstance(files, dict) or not files:
        return None
    # A truncated reply parses as nothing at all, so this is mostly belt-and-braces -- but
    # a file whose text stops mid-token is worse than no candidate: it becomes an EXCLUDE
    # that looks like a model that cannot refactor rather than a budget we set too low.
    if any(not isinstance(v, str) for v in files.values()):
        return None
    return {"sources": {**sources, **files}, "model": model, "temperature": temperature,
            "budget": budget}


def label(adapter, task, sources, oracle_text, workspace, admissible=None):
    """(GOOD|BAD|EXCLUDE, why). EXCLUDE means it never built -- spec §20.

    `admissible` is the set of oracle cases that are GREEN ON P0, and the label is decided
    on THAT SUBSET ALONE. Deciding it on whole-suite greenness instead makes GOOD
    unreachable whenever the holdout is not itself green on P0 -- which is the normal case
    here, since a RefactorBench holdout also asserts behaviour the pre-refactor program
    does not yet have. Measured on analytics_dashboard: P0 fails 3 of 127 cases, so every
    candidate, including a perfect behaviour-preserving refactor, was labelled BAD and the
    GOOD population came back empty on every fixture. A case P0 itself fails cannot
    distinguish a faithful transformation from a broken one, so it must not be allowed to
    condemn one.
    """
    ws = adapter.stage(task.task_id, workspace, sources=sources)
    # `run_suites` so the case names match `prepare_oracle`'s admissible set -- see the
    # note there. `oracle_text` may be one file's text or a list of them.
    files = [oracle_text] if isinstance(oracle_text, str) else list(oracle_text)
    res = adapter.run_suites(ws, files, "oracle")
    if res.status in ("build_error", "no_tests", "error"):
        return "EXCLUDE", f"hidden oracle could not run: {res.status}"
    if res.status == "timeout":
        return "BAD", "hidden oracle timed out"
    if admissible is None:
        failed = list(res.failed)
    else:
        adm = set(admissible)
        failed = [c for c in res.failed if c in adm]
        # A case that is admissible but absent from this run's report did not merely fail
        # -- it never ran, because the candidate broke the file that carries it. Silently
        # treating "not reported" as "passed" would label such a candidate GOOD.
        missing = adm - set(res.passed) - set(res.failed)
        failed += sorted(missing)
    if not failed:
        return "GOOD", None
    # `hidden_oracle_failure` is read by people, so the file-index prefix `run_suites` adds
    # to every case name comes off here. It is load-bearing for matching admissible sets,
    # and noise in a sentence explaining why a candidate is BAD.
    return "BAD", f"hidden oracle failed: {', '.join(_unprefixed(c) for c in failed[:6])}"


def _unprefixed(case):
    """`"3::Foo bar"` -> `"Foo bar"`. Leaves an unprefixed name alone."""
    head, sep, rest = case.partition("::")
    return rest if sep and head.isdigit() else case
