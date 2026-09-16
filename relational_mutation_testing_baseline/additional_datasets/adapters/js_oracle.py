"""Prepare a fixture's hidden oracle -- the WHOLE developer suite -- and verify it on P0.

Decision, 2026-09-15 (user-directed, spec §4.4): the developer/holdout suite *is* the
oracle, used whole and unsplit. It labels every candidate GOOD or BAD, and the experiment
is ARM M against ARM N on those labelled implementations. Baseline A is consequently the
oracle itself: its GAR and its BRR are both 1.0 by construction, so it is reported as
definitional and excluded from every comparison rather than engineered around. The full
suite is a strictly stronger labeller than half of it.

What is still checked here, and why it is not optional: a holdout that is not green on the
fixture's own untouched sources cannot label anything -- every candidate would come back
GOOD by default -- so the suite is RUN against P0 and only the cases that pass there are
admissible. That execution is the verification; `reason` records it when it fails.

`js_split.split_suite` is no longer called from this module: its only purpose was to
manufacture an independent Baseline A, which no longer exists. It is deliberately kept in
the tree, correct and covered by tests/test_js_split.py, because the retired design is
recorded in the spec and the splitter is the artefact that design refers to.
"""
from . import js_vitest as JS


def prepare_oracle(name, workspace):
    """{oracle_text, admissible_oracle, reason} for `name`, verified against its own P0."""
    whole = JS.holdout_text(name)
    ws = JS.stage(name, workspace)
    # `run_suites`, not `run_suite`, even though the oracle is a single file: its case
    # names must be built in the same namespace as the names a candidate is scored on, and
    # `run_suites` file-index prefixes them. `harvest.label` compares a candidate's failed
    # cases against `admissible_oracle`, and `score_task` scores ARM A from it; both go
    # through `run_suites`, so all three agree on what a case is called.
    full = JS.run_suites(ws, [whole], "whole")
    # A non-green holdout still yields whatever cases DID pass: those cases are sound
    # evidence, and the caller (which owns the paid-call budget) decides whether an
    # oracle with too few of them is worth running. It must never be silently upgraded
    # to "fine", hence the reason string.
    return {"oracle_text": whole, "admissible_oracle": full.passed,
            "reason": "" if full.status == "green"
                      else f"holdout suite is not green on P0: {full.status}"}
