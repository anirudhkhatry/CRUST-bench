"""Prepare a fixture's hidden oracle and its Baseline A half, and verify the split.

Baseline A and the hidden oracle are the same file until it is split, which would make
Baseline A a tautology. The split is only trustworthy if it is checked: both halves green
on the untouched P0, their case names union to the whole suite's, and their intersection
empty. A fixture that fails any of the three keeps its oracle and loses only Baseline A.
"""
from . import js_vitest as JS
from .js_split import split_suite


def prepare_oracle(name, workspace):
    whole = JS.holdout_text(name)
    oracle_text, existing_text, assignment = split_suite(whole, seed=name)
    ws = JS.stage(name, workspace)

    full = JS.run_suite(ws, whole, "whole")
    if full.status != "green":
        return {"oracle_text": whole, "existing_text": "", "assignment": {},
                "admissible_oracle": full.passed, "admissible_existing": [],
                "arm_a_usable": False,
                "reason": f"holdout suite is not green on P0: {full.status}"}

    if not assignment:
        return {"oracle_text": whole, "existing_text": "", "assignment": {},
                "admissible_oracle": full.passed, "admissible_existing": [],
                "arm_a_usable": False, "reason": "unsplittable: fewer than two leaf cases"}

    a = JS.run_suite(ws, oracle_text, "oracle")
    b = JS.run_suite(ws, existing_text, "existing")
    got = {"oracle_text": oracle_text, "existing_text": existing_text,
           "assignment": assignment, "admissible_oracle": a.passed,
           "admissible_existing": b.passed, "arm_a_usable": True, "reason": ""}

    overlap = set(a.passed) & set(b.passed)
    union = set(a.passed) | set(b.passed)
    if a.status != "green" or b.status != "green":
        got |= {"arm_a_usable": False,
                "reason": f"a half is not green on P0: {a.status}/{b.status}"}
    elif overlap:
        got |= {"arm_a_usable": False,
                "reason": f"halves overlap on {len(overlap)} cases"}
    elif union != set(full.cases):
        missing = set(full.cases) - union
        got |= {"arm_a_usable": False,
                "reason": f"halves lose {len(missing)} cases from the whole suite"}
    if not got["arm_a_usable"]:
        # Keep the whole suite as the oracle; only Baseline A is forfeited.
        got |= {"oracle_text": whole, "existing_text": "",
                "admissible_oracle": full.passed, "admissible_existing": []}
    return got
