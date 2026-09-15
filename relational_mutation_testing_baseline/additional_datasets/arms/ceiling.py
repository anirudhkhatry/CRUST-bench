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
from .naive import SYSTEM, _bundle, _generate_with_repair


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
    """KILLED | F_UNSOUND_ON_GOOD | F_NOT_DISCRIMINATING."""
    if on_good.status != "green":
        return "F_UNSOUND_ON_GOOD"
    if on_bad.status in ("test_failed", "timeout", "build_error"):
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
    from ..score import ACCEPT, BEHAVIOURAL, REJECT

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
        rows.append({"task_id": task.task_id, "benchmark": task.benchmark,
                     "candidate_id": cid, "ground_truth": "BAD", "label": "BAD",
                     "bad_arm": cand.bad_arm, "fault_category": cand.fault_category,
                     "arm": "ceiling", "compiles": True, "existing_tests_pass": None,
                     "generated_tests_pass": not killed,
                     "number_generated_tests": len(on_good.cases),
                     "tests_failed": len(on_bad.failed),
                     "verdict": REJECT if killed else ACCEPT,
                     "rejection_kind": BEHAVIOURAL if killed else "",
                     "status": on_bad.status, "execution_time": on_bad.seconds,
                     "ceiling_outcome": oc, "unreachable": rec["unreachable"],
                     "model": cand.model, "model_version": cand.model_version,
                     "hidden_oracle_failure": cand.hidden_oracle_failure})
    return rows
