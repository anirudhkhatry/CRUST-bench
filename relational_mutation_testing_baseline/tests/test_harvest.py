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

    def run_suites(self, workspace, suite_texts, suite_name):
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


def test_a_case_p0_itself_fails_cannot_condemn_a_candidate(tmp_path):
    """The label is decided on the admissible subset -- the cases green on P0.

    A RefactorBench holdout also asserts post-refactor behaviour, so it is normally NOT
    green on P0. Judging on whole-suite greenness made GOOD unreachable for every
    candidate on every fixture, including a faithful refactor.
    """
    a = FakeAdapter(SuiteResult(status="test_failed",
                                cases={"adm": "ok", "not_on_p0": "FAILED"},
                                rc=1, seconds=0.1))
    assert harvest.label(a, _task(tmp_path), {}, "suite", tmp_path,
                         admissible=["adm"]) == ("GOOD", None)


def test_an_admissible_case_that_vanished_from_the_report_is_bad(tmp_path):
    """Not reported is not passed: a candidate that breaks the file carrying a case
    would otherwise be labelled GOOD for having silenced its own judge."""
    a = FakeAdapter(SuiteResult(status="test_failed", cases={"other": "FAILED"},
                                rc=1, seconds=0.1))
    label, why = harvest.label(a, _task(tmp_path), {}, "suite", tmp_path,
                               admissible=["adm", "other"])
    assert label == "BAD"
    assert "adm" in why


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


def test_propose_candidate_forwards_the_model_and_temperature_it_was_given(monkeypatch):
    """Change 3: the sweep is only real if both values reach `ask` and are recorded.

    `rmt.llm.ask` gained `model=` and applied `temperature=` at 9176dcb. Fails on the
    previous code, which passed neither model nor an honest record of what was used --
    every entry in the sweep then produced the same call and the spread was noise.
    """
    seen = {}
    monkeypatch.setattr(harvest, "ask", lambda system, user, **kw: seen.update(kw) or
                        '```json\n{"app/index.js": "export const x = 2;"}\n```')
    got = harvest.propose_candidate(_task_obj(), {"app/index.js": "export const x = 1;"},
                                    "app/index.js", "claude-haiku-4-5-20251001", 0.7, "t")
    assert seen["model"] == "claude-haiku-4-5-20251001"
    assert seen["temperature"] == 0.7
    # ...and it reports back what it used, so the Candidate cannot record a setting that
    # was never applied.
    assert got["model"] == "claude-haiku-4-5-20251001" and got["temperature"] == 0.7


def test_two_temperatures_do_not_share_a_cache_tag(monkeypatch):
    """`ask` names its cache entry `<tag>_<hash>.json`. Two draws that shared a tag AND a
    payload would return the identical candidate for free, collapsing the sweep.

    Fails if the temperature is dropped from the tag while also being dropped from the
    payload -- which is precisely the state the code was in before 9176dcb.
    """
    tags = []
    monkeypatch.setattr(harvest, "ask", lambda system, user, **kw:
                        tags.append(kw["tag"]) or '```json\n{"a.js": "x"}\n```')
    for temp in (0.0, 1.0):
        harvest.propose_candidate(_task_obj(), {"app/index.js": "export const x = 1;"},
                                  "app/index.js", "m", temp, "t")
    assert len(set(tags)) == 2, tags


def test_the_sweep_ends_with_tight_budget_weak_model_draws(monkeypatch):
    """The only source of genuine harvested BAD candidates must survive refactoring.

    With a full output budget 11 of 12 draws come back GOOD and Arm S admitted 0 BAD from
    8 seeded faults, so without these trailing draws every fixture is GOOD-only and BRR is
    n/a for every arm.
    """
    from additional_datasets.benchmarks import refactorbench_js as rjs
    tail = rjs.CANDIDATE_SWEEP[-len(rjs.CONSTRAINED):]
    assert tail and all(b == "tight" and m == rjs.WEAK_MODEL for m, _, b in tail)
    assert all(b == "full" for _, _, b in rjs.CANDIDATE_SWEEP[:-len(rjs.CONSTRAINED)])
    assert harvest.BUDGETS["tight"][1] < harvest.BUDGETS["full"][1]
