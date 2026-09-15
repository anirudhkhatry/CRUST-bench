import pytest

from additional_datasets.runner import LeakageError, freeze, score_task
from additional_datasets.score import ACCEPT, REJECT
from additional_datasets.spec import Candidate, SuiteResult, Task, write_metadata


def _task(tmp_path):
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="r", source_commit="c",
                transformation="t", results_root=tmp_path)


class ScriptedAdapter:
    """Returns a result keyed by (candidate_id, suite_name)."""

    def __init__(self, table):
        self.table = table

    def stage(self, name, dest, sources=None):
        return dest

    def run_suite(self, workspace, suite_text, suite_name):
        return self.table[suite_name]


GREEN = SuiteResult(status="green", cases={"t1": "ok"}, rc=0, seconds=0.1)
RED = SuiteResult(status="test_failed", cases={"t1": "FAILED"}, rc=1, seconds=0.1)


def test_score_task_emits_a_row_per_arm_per_candidate(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    freeze(t, {"mutation": "M", "naive": "N"})
    good = Candidate("good_01", t.task_id, "GOOD", None, None, "m", "m", {}, "ok", None)
    bad = Candidate("bad_01", t.task_id, "BAD", "S", "ordering difference", "m", "m", {},
                    "ok", "oracle failed")
    cands = {"good_01": (good, {}), "bad_01": (bad, {})}
    adapter = ScriptedAdapter({"mutation@good_01": GREEN, "mutation@bad_01": RED,
                               "naive@good_01": GREEN, "naive@bad_01": GREEN})
    rows = score_task(t, cands, suites={"mutation": "M", "naive": "N"},
                      admissible={"mutation": ["t1"], "naive": ["t1"]},
                      adapter=adapter, workspace=tmp_path / "ws")
    assert len(rows) == 4
    by = {(r["arm"], r["candidate_id"]): r for r in rows}
    assert by[("mutation", "bad_01")]["verdict"] == REJECT
    assert by[("naive", "bad_01")]["verdict"] == ACCEPT
    assert by[("mutation", "good_01")]["verdict"] == ACCEPT


def test_rows_carry_every_spec_19_column(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    freeze(t, {"mutation": "M"})
    bad = Candidate("bad_01", t.task_id, "BAD", "N", "ordering difference", "m", "m", {},
                    "ok", "oracle failed")
    rows = score_task(t, {"bad_01": (bad, {})}, suites={"mutation": "M"},
                      admissible={"mutation": ["t1"]},
                      adapter=ScriptedAdapter({"mutation@bad_01": RED}),
                      workspace=tmp_path / "ws")
    r = rows[0]
    for col in ("task_id", "candidate_id", "ground_truth", "compiles",
                "existing_tests_pass", "generated_tests_pass", "combined_tests_pass",
                "number_generated_tests", "tests_failed", "fault_category",
                "execution_time", "arm", "bad_arm"):
        assert col in r, col
    assert r["ground_truth"] == "BAD"
    assert r["generated_tests_pass"] is False
    assert r["bad_arm"] == "N"
    # None, not False: "this generated suite is not Baseline A" is a different claim from
    # "Baseline A failed". A regression to `else False` would pass a key-presence check.
    assert r["existing_tests_pass"] is None
    # No Baseline A in this run, so there is nothing to combine with.
    assert r["combined_tests_pass"] is None
    # The suite's real size, not the admissible subset.
    assert r["number_generated_tests"] == 1


def test_combined_is_the_and_of_baseline_a_and_the_generated_arm(tmp_path):
    """Design §14's fifth baseline: existing + generated, the RQ2 column.

    A candidate the developer suite already catches is not new evidence for the generated
    arm; a candidate only the generated arm catches is exactly the incremental value the
    experiment exists to measure.
    """
    t = _task(tmp_path)
    write_metadata(t, [])
    bad = Candidate("bad_01", t.task_id, "BAD", "S", "ordering difference", "m", "m", {},
                    "ok", "oracle failed")
    freeze(t, {"mutation": "M", "existing": "E"})

    # Baseline A accepts, the generated arm rejects -> combined rejects: the generated arm
    # found something the developer suite missed.
    rows = score_task(t, {"bad_01": (bad, {})}, suites={"mutation": "M"},
                      admissible={"mutation": ["t1"]},
                      adapter=ScriptedAdapter({"mutation@bad_01": RED,
                                               "existing@bad_01": GREEN}),
                      workspace=tmp_path / "ws",
                      existing_text="E", existing_admissible=["t1"])
    by = {r["arm"]: r for r in rows}
    assert by["mutation"]["combined_tests_pass"] is False
    assert by["existing"]["combined_tests_pass"] is None, "the baseline row IS the baseline"

    # Both accept -> combined accepts.
    rows = score_task(t, {"bad_01": (bad, {})}, suites={"mutation": "M"},
                      admissible={"mutation": ["t1"]},
                      adapter=ScriptedAdapter({"mutation@bad_01": GREEN,
                                               "existing@bad_01": GREEN}),
                      workspace=tmp_path / "ws2",
                      existing_text="E", existing_admissible=["t1"])
    assert {r["arm"]: r for r in rows}["mutation"]["combined_tests_pass"] is True


def test_an_arm_with_no_admissible_set_is_refused(tmp_path):
    """An arm that cannot reject would silently accept everything and look like a result."""
    t = _task(tmp_path)
    write_metadata(t, [])
    bad = Candidate("bad_01", t.task_id, "BAD", "S", None, "m", "m", {}, "ok", "why")
    freeze(t, {"mutation": "M", "naive": "N"})
    with pytest.raises(LeakageError, match="no admissible set"):
        score_task(t, {"bad_01": (bad, {})}, suites={"mutation": "M", "naive": "N"},
                   admissible={"mutation": ["t1"]},          # "naive" omitted
                   adapter=ScriptedAdapter({}), workspace=tmp_path / "ws")


def test_existing_tests_are_scored_as_their_own_arm(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    bad = Candidate("bad_01", t.task_id, "BAD", "S", "ordering difference", "m", "m", {},
                    "ok", "oracle failed")
    freeze(t, {"mutation": "M", "existing": "E"})
    rows = score_task(t, {"bad_01": (bad, {})}, suites={"mutation": "M"},
                      admissible={"mutation": ["t1"]},
                      adapter=ScriptedAdapter({"mutation@bad_01": RED,
                                               "existing@bad_01": GREEN}),
                      workspace=tmp_path / "ws",
                      existing_text="E", existing_admissible=["t1"])
    arms = {r["arm"] for r in rows}
    assert arms == {"mutation", "existing"}
    existing = [r for r in rows if r["arm"] == "existing"][0]
    assert existing["verdict"] == ACCEPT
    assert existing["existing_tests_pass"] is True
