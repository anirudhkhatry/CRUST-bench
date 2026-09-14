import json
from pathlib import Path

from additional_datasets.spec import Candidate, SuiteResult, Task, read_metadata, write_metadata


def _task(tmp_path):
    return Task(
        task_id="binary_search",
        benchmark="refactorbench_js",
        language="javascript",
        repository="Create-Inc/refactor-bench",
        source_commit="deadbeef",
        transformation="Decompose app/index.js into smaller modules, preserving behaviour.",
        results_root=tmp_path,
    )


def test_task_root_follows_the_spec_layout(tmp_path):
    t = _task(tmp_path)
    assert t.root == tmp_path / "refactorbench_js" / "binary_search"
    assert t.generated_tests_dir == t.root / "generated_tests"
    assert t.hidden_oracle_dir == t.root / "hidden_oracle"


def test_candidate_dir_is_under_candidates(tmp_path):
    t = _task(tmp_path)
    c = Candidate(candidate_id="bad_01", task_id=t.task_id, label="BAD", bad_arm="S",
                  fault_category="ordering difference", model="claude-opus-5",
                  model_version="claude-opus-5", generation_parameters={"temperature": 1.0},
                  build_status="ok", hidden_oracle_failure="3 cases regressed")
    assert c.dir(t) == t.root / "candidates" / "bad_01"


def test_suite_result_passed_and_failed_case_names():
    r = SuiteResult(status="test_failed", cases={"a": "ok", "b": "FAILED"}, rc=1, seconds=0.5)
    assert r.passed == ["a"]
    assert r.failed == ["b"]


def test_metadata_round_trips_every_spec_9_field(tmp_path):
    t = _task(tmp_path)
    c = Candidate(candidate_id="good_01", task_id=t.task_id, label="GOOD", bad_arm=None,
                  fault_category=None, model="claude-opus-5", model_version="claude-opus-5",
                  generation_parameters={"temperature": 0.0}, build_status="ok",
                  hidden_oracle_failure=None)
    write_metadata(t, [c], extra={"suite_hashes": {"naive": "abc"}})
    blob = read_metadata(t)
    assert blob["task_id"] == "binary_search"
    assert blob["repository"] == "Create-Inc/refactor-bench"
    assert blob["source_commit"] == "deadbeef"
    assert blob["language"] == "javascript"
    assert blob["suite_hashes"] == {"naive": "abc"}
    row = blob["candidates"][0]
    for field in ("candidate_id", "ground_truth_label", "build_status",
                  "hidden_oracle_failure", "fault_category", "model", "model_version",
                  "generation_parameters", "bad_arm", "candidate_commit"):
        assert field in row, field
    assert row["ground_truth_label"] == "GOOD"


def test_candidate_commit_round_trips_when_set(tmp_path):
    t = _task(tmp_path)
    c = Candidate(candidate_id="bad_01", task_id=t.task_id, label="BAD", bad_arm="S",
                  fault_category="ordering difference", model="claude-opus-5",
                  model_version="claude-opus-5", generation_parameters={},
                  build_status="ok", hidden_oracle_failure="2 cases regressed",
                  candidate_commit="sha256:abcd1234")
    write_metadata(t, [c])
    assert read_metadata(t)["candidates"][0]["candidate_commit"] == "sha256:abcd1234"
