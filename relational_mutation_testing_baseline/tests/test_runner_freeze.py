import json

import pytest

from additional_datasets.runner import LeakageError, assert_frozen, assert_no_leak, freeze
from additional_datasets.spec import Task, write_metadata


def _task(tmp_path):
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="Create-Inc/refactor-bench",
                source_commit="deadbeef", transformation="Decompose it.",
                results_root=tmp_path)


def test_freeze_writes_suites_and_records_their_hashes(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    hashes = freeze(t, {"mutation": "test('a', () => {})", "naive": "test('b', () => {})"})
    assert set(hashes) == {"mutation", "naive"}
    assert (t.generated_tests_dir / "mutation" / "suite.js").exists()
    blob = json.loads((t.root / "metadata.json").read_text())
    assert blob["suite_hashes"] == hashes


def test_freeze_refuses_once_candidates_exist(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    (t.candidates_dir / "good_01").mkdir(parents=True)
    with pytest.raises(LeakageError, match="candidates already exist"):
        freeze(t, {"naive": "test('a', () => {})"})


def test_assert_frozen_detects_a_changed_suite(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    freeze(t, {"naive": "test('a', () => {})"})
    assert_frozen(t, {"naive": "test('a', () => {})"})          # unchanged: fine
    with pytest.raises(LeakageError, match="changed after freezing"):
        assert_frozen(t, {"naive": "test('a', () => {}); // tweaked"})


def test_assert_no_leak_catches_candidate_text_in_a_prompt():
    assert_no_leak("Write tests for this library.", ["const SECRET_BAD_CODE = 1;"])
    with pytest.raises(LeakageError, match="leaked"):
        assert_no_leak("Here is the candidate: const SECRET_BAD_CODE = 1;",
                       ["const SECRET_BAD_CODE = 1;"])


def test_assert_no_leak_ignores_trivially_short_fragments():
    # A three-character fragment appears in any English prompt; only meaningful
    # fragments are checked, or the guard would fire constantly and get disabled.
    assert_no_leak("Write a test for the app.", ["app"])
