from additional_datasets import harvest
from additional_datasets.spec import SuiteResult, Task


def _task(tmp_path):
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="Create-Inc/refactor-bench",
                source_commit="deadbeef", transformation="Decompose it.",
                results_root=tmp_path)


def _task_obj():
    """Task object for tests that don't need tmp_path."""
    from pathlib import Path
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="Create-Inc/refactor-bench",
                source_commit="deadbeef", transformation="Decompose it.",
                results_root=Path("/tmp"))


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


def test_an_equivalent_fault_is_not_labelled_bad(tmp_path):
    """Spec §3.5: a fault the oracle cannot see is unkillable by anyone.

    Leaving it in the denominator silently depresses every arm's detection rate while
    looking like a fair difficulty increase, so it must not come back BAD.
    """
    a = FakeAdapter(SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1))
    assert harvest.label(a, _task(tmp_path), {}, "suite", tmp_path)[0] == "GOOD"


def test_propose_candidate_merges_the_models_files_over_the_originals(monkeypatch):
    monkeypatch.setattr(harvest, "ask", lambda *a, **k:
                        '```json\n{"app/index.js": "export const x = 2;"}\n```')
    sources = {"app/index.js": "export const x = 1;", "app/util.js": "export const y = 3;"}
    got = harvest.propose_candidate(_task_obj(), sources, "app/index.js", "m", 1.0, "t")
    assert got["sources"]["app/index.js"] == "export const x = 2;"
    # Untouched files must survive the merge -- a candidate is the whole tree, not a patch.
    assert got["sources"]["app/util.js"] == "export const y = 3;"


def test_propose_candidate_returns_none_on_an_unusable_reply(monkeypatch):
    sources = {"app/index.js": "export const x = 1;"}
    for reply in ("no json here at all",
                  '```json\n["a list, not a map"]\n```',
                  '```json\n{}\n```'):
        monkeypatch.setattr(harvest, "ask", lambda *a, **k: reply)
        assert harvest.propose_candidate(_task_obj(), sources, "app/index.js",
                                         "m", 1.0, "t") is None, reply


def test_propose_candidate_prompt_never_sees_the_oracle_or_a_label(monkeypatch):
    """Arm N's whole value is that its candidates were generated blind.

    If oracle text or a GOOD/BAD label could reach this prompt, the candidates would be
    contaminated and Arm N would stop being independent evidence.
    """
    seen = {}
    monkeypatch.setattr(harvest, "ask", lambda system, user, **k: seen.update(user=user) or "")
    harvest.propose_candidate(_task_obj(), {"app/index.js": "export const x = 1;"},
                              "app/index.js", "m", 1.0, "t")
    for forbidden in ("expect(", "describe(", "GOOD", "BAD", "oracle"):
        assert forbidden not in seen["user"], f"{forbidden!r} leaked into the Arm N prompt"
