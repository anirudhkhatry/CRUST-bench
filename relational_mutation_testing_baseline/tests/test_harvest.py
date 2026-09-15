from additional_datasets import harvest
from additional_datasets.spec import SuiteResult, Task


def _task(tmp_path):
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="Create-Inc/refactor-bench",
                source_commit="deadbeef", transformation="Decompose it.",
                results_root=tmp_path)


class FakeAdapter:
    def __init__(self, result):
        self.result = result

    def stage(self, name, dest, sources=None):
        return dest

    def run_suite(self, workspace, suite_text, suite_name):
        return self.result


def test_a_candidate_passing_the_hidden_oracle_is_good(tmp_path):
    a = FakeAdapter(SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1))
    assert harvest.label(a, _task(tmp_path), {}, "suite", tmp_path) == ("GOOD", None)


def test_a_candidate_failing_the_hidden_oracle_is_bad_and_records_why(tmp_path):
    a = FakeAdapter(SuiteResult(status="test_failed", cases={"a": "ok", "b": "FAILED"},
                                rc=1, seconds=0.1))
    label, why = harvest.label(a, _task(tmp_path), {}, "suite", tmp_path)
    assert label == "BAD"
    assert "b" in why


def test_a_candidate_that_does_not_build_is_excluded_not_bad(tmp_path):
    """Spec §20: a compilation failure is not an acceptable BAD implementation."""
    a = FakeAdapter(SuiteResult(status="build_error", cases={}, rc=1, seconds=0.1))
    assert harvest.label(a, _task(tmp_path), {}, "suite", tmp_path)[0] == "EXCLUDE"


def test_a_seeded_fault_the_oracle_cannot_see_is_excluded_as_equivalent(tmp_path):
    """Spec §3.5: an equivalent mutant is unkillable and would depress every arm."""
    a = FakeAdapter(SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1))
    assert harvest.confirm_bad(a, _task(tmp_path), {}, "suite", tmp_path) is False
