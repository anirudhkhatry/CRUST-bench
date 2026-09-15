"""ARM C -- the ceiling: one test per BAD candidate, written from the GOOD->BAD diff.

This arm is shown exactly what is wrong, in the language the test is written in, with no
translation step to lose it in. So it is an upper bound, not a technique: whatever it
cannot kill is a fault that is hard to reach through the public surface rather than one
the other arms mislaid, and `1 - its BRR` is the floor under every other arm's miss rate.

It is exempt from the §7 freeze because it is handed the candidate by construction, and it
is scored per candidate rather than once per task, so it does not go through `score_task`.
"""
import difflib

from ..prompts import load
from .naive import _bundle, _generate_with_repair


def diff_of(good_sources, bad_sources):
    """Unified diff over every file the fault touched or added."""
    out = []
    for rel in sorted(set(good_sources) | set(bad_sources)):
        a = good_sources.get(rel, "").splitlines(keepends=True)
        b = bad_sources.get(rel, "").splitlines(keepends=True)
        if a == b:
            continue
        out += list(difflib.unified_diff(a, b, fromfile=f"good/{rel}", tofile=f"bad/{rel}"))
    return "".join(out)


def is_unreachable(suite_text):
    return (suite_text or "").lstrip().startswith("// UNREACHABLE:")


def outcome(on_good, on_bad):
    """KILLED | F_UNSOUND_ON_GOOD | F_BAD_DID_NOT_BUILD | F_NOT_DISCRIMINATING.

    A suite that will not BUILD against the candidate has observed no behaviour at all: it
    reports that the candidate is shaped differently (a renamed export, a moved module),
    which is exactly the structural/behavioural confusion `score.py` separates and
    `mutation.kills` already refuses to bank (spec §20). Crediting it here would be worse
    than elsewhere, because ARM C is the ceiling -- `1 - its BRR` is the floor under every
    other arm's miss rate, so an inflated ARM C makes every other arm look worse by
    construction. It is therefore a MISS for the ceiling, not a kill.
    """
    if on_good.status != "green":
        return "F_UNSOUND_ON_GOOD"
    if on_bad.status == "build_error":
        return "F_BAD_DID_NOT_BUILD"
    if on_bad.status in ("test_failed", "timeout"):
        return "KILLED"
    return "F_NOT_DISCRIMINATING"


def generate(task, good_sources, bad_sources, adapter, workspace, import_line,
             framework="vitest", retries=3, tag="ceil"):
    base = load("ceiling_testgen", sources=_bundle(good_sources),
                diff=diff_of(good_sources, bad_sources), import_line=import_line,
                language=task.language.capitalize(), framework=framework)
    good_ws = adapter.stage(task.task_id, workspace / "good", sources=good_sources)
    suite, rec = _generate_with_repair(base, adapter, good_ws, retries, tag)
    rec["unreachable"] = is_unreachable(suite)
    return suite, rec


def score_ceiling(task, candidates, adapter, workspace, good_sources, import_line):
    """One row per BAD candidate. GOOD candidates are not scored: ARM C has no GAR."""
    from ..score import ACCEPT, BEHAVIOURAL, REJECT, STRUCTURAL

    rows = []
    for cid, (cand, sources) in sorted(candidates.items()):
        if cand.label != "BAD":
            continue
        ws = workspace / f"ceil_{cid}"
        suite, rec = generate(task, good_sources, sources, adapter, ws, import_line,
                              tag=f"ceil_{cid}")
        if not rec["compiles"]:
            rows.append({"task_id": task.task_id, "benchmark": task.benchmark,
                         "candidate_id": cid, "ground_truth": "BAD", "label": "BAD",
                         "bad_arm": cand.bad_arm, "fault_category": cand.fault_category,
                         "arm": "ceiling", "compiles": True,
                         "existing_tests_pass": None, "generated_tests_pass": True,
                         "combined_tests_pass": None,
                         "number_generated_tests": 0, "tests_failed": 0,
                         "verdict": ACCEPT, "rejection_kind": "",
                         "status": "no_suite", "execution_time": 0.0,
                         "ceiling_outcome": "F_NO_SUITE", "unreachable": rec["unreachable"],
                         "model": cand.model, "model_version": cand.model_version,
                         "hidden_oracle_failure": cand.hidden_oracle_failure})
            continue
        good_ws = adapter.stage(task.task_id, ws / "good2", sources=good_sources)
        bad_ws = adapter.stage(task.task_id, ws / "bad", sources=sources)
        on_good = adapter.run_suite(good_ws, suite, f"ceiling@{cid}_good")
        on_bad = adapter.run_suite(bad_ws, suite, f"ceiling@{cid}")
        oc = outcome(on_good, on_bad)
        killed = oc == "KILLED"
        # Only a KILLED row is a rejection now, and it is behavioural by construction: the
        # suite was green on GOOD and red on BAD through an assertion or a timeout. The
        # structurally-driven row is an ACCEPT (a ceiling MISS) and still carries
        # `structural` so the reason is legible in rows.json -- no ARM C row can any longer
        # claim `behavioural` for a difference in shape, which is the mislabel
        # `score.py`'s behavioural/structural split could not have undone afterwards.
        kind = BEHAVIOURAL if killed else (
            STRUCTURAL if oc == "F_BAD_DID_NOT_BUILD" else "")
        rows.append({"task_id": task.task_id, "benchmark": task.benchmark,
                     "candidate_id": cid, "ground_truth": "BAD", "label": "BAD",
                     "bad_arm": cand.bad_arm, "fault_category": cand.fault_category,
                     "arm": "ceiling", "compiles": True, "existing_tests_pass": None,
                     "generated_tests_pass": not killed,
                     "combined_tests_pass": None,
                     "number_generated_tests": len(on_good.cases),
                     "tests_failed": len(on_bad.failed),
                     "verdict": REJECT if killed else ACCEPT,
                     "rejection_kind": kind,
                     "status": on_bad.status, "execution_time": on_bad.seconds,
                     "ceiling_outcome": oc, "unreachable": rec["unreachable"],
                     "model": cand.model, "model_version": cand.model_version,
                     "hidden_oracle_failure": cand.hidden_oracle_failure})
    return rows
