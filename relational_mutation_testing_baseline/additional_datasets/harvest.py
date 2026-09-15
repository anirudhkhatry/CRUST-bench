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


def propose_candidate(task, sources, target, model, temperature, tag):
    """One candidate transformation, or None if the reply was unusable.

    `model` and `temperature` are both forwarded to `ask` and both returned, so what the
    Candidate records is what the call actually used. Since 9176dcb `rmt.llm.ask` accepts
    and applies both -- before that it had no `model` argument and dropped `temperature`,
    so the sweep was sampling noise and nothing could honestly be recorded. Passing them
    explicitly (rather than leaning on `ask`'s defaults) is what makes the temperature
    sweep real: the spread of GOOD vs BAD candidates depends on it.
    """
    user = load("refactor_js", target=target, sources=_bundle(sources))
    reply = ask(SYSTEM, user, effort="high", max_tokens=32000,
                model=model, temperature=temperature, tag=f"{tag}_{model}_t{temperature}")
    try:
        files = json_block(reply)
    except ValueError:
        return None
    if not isinstance(files, dict) or not files:
        return None
    return {"sources": {**sources, **files}, "model": model, "temperature": temperature}


def label(adapter, task, sources, oracle_text, workspace):
    """(GOOD|BAD|EXCLUDE, why). EXCLUDE means it never built -- spec §20."""
    ws = adapter.stage(task.task_id, workspace, sources=sources)
    res = adapter.run_suite(ws, oracle_text, "oracle")
    if res.status in ("build_error", "no_tests", "error"):
        return "EXCLUDE", f"hidden oracle could not run: {res.status}"
    if res.status == "green":
        return "GOOD", None
    if res.status == "timeout":
        return "BAD", "hidden oracle timed out"
    return "BAD", f"hidden oracle failed: {', '.join(res.failed[:6])}"
