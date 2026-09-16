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
    assert (t.generated_tests_dir / "mutation" / "suite_0.js").exists()
    blob = json.loads((t.root / "metadata.json").read_text())
    assert blob["suite_hashes"] == hashes


def test_freeze_writes_one_file_per_suite_file_and_hashes_the_whole_list(tmp_path):
    """An arm's suite is a LIST of test files, so all of them must land on disk and the
    recorded hash must cover all of them -- hashing only the first would let a later file
    be swapped after the freeze without the gate firing."""
    t = _task(tmp_path)
    write_metadata(t, [])
    hashes = freeze(t, {"naive": ["test('a', () => {})", "test('b', () => {})"]})
    d = t.generated_tests_dir / "naive"
    assert (d / "suite_0.js").read_text() == "test('a', () => {})"
    assert (d / "suite_1.js").read_text() == "test('b', () => {})"
    assert_frozen(t, {"naive": ["test('a', () => {})", "test('b', () => {})"]})
    with pytest.raises(LeakageError, match="changed after freezing"):
        assert_frozen(t, {"naive": ["test('a', () => {})", "test('b', () => {}) // edit"]})
    # Order is part of the suite's identity: `run_suites` names cases by file index, so a
    # reordered list is a different suite as far as every admissible set is concerned.
    with pytest.raises(LeakageError, match="changed after freezing"):
        assert_frozen(t, {"naive": ["test('b', () => {})", "test('a', () => {})"]})
    # A one-file suite passed as a bare string must hash the same as the one-element list,
    # or ARM A (a single developer file) could never be frozen and scored consistently.
    t2 = _task(tmp_path / "other")
    write_metadata(t2, [])
    assert freeze(t2, {"existing": "test('a', () => {})"})["existing"] == \
        freeze(_fresh(tmp_path / "third"), {"existing": ["test('a', () => {})"]})["existing"]
    assert hashes["naive"] != freeze(_fresh(tmp_path / "fourth"),
                                     {"naive": ["test('a', () => {})"]})["naive"]


def _fresh(root):
    t = Task(task_id="binary_search", benchmark="refactorbench_js", language="javascript",
             repository="Create-Inc/refactor-bench", source_commit="deadbeef",
             transformation="Decompose it.", results_root=root)
    write_metadata(t, [])
    return t


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


def test_assert_frozen_refuses_when_nothing_was_ever_frozen(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])                       # metadata exists, freeze never called
    with pytest.raises(LeakageError, match="never called"):
        assert_frozen(t, {"naive": "test('a', () => {})"})


def test_assert_frozen_refuses_an_arm_that_was_not_frozen(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    freeze(t, {"naive": "test('a', () => {})"})
    with pytest.raises(LeakageError, match="never frozen"):
        assert_frozen(t, {"naive": "test('a', () => {})",
                          "mutation": "test('b', () => {})"})


def test_assert_frozen_refuses_when_there_is_no_metadata_at_all(tmp_path):
    t = _task(tmp_path)
    t.root.mkdir(parents=True, exist_ok=True)   # no metadata.json written
    with pytest.raises(LeakageError, match="never frozen"):
        assert_frozen(t, {"naive": "test('a', () => {})"})


def test_freeze_refuses_an_unknown_language(tmp_path):
    t = Task(task_id="x", benchmark="b", language="Javascript", repository="r",
             source_commit="c", transformation="t", results_root=tmp_path)
    write_metadata(t, [])
    with pytest.raises(LeakageError, match="unknown language"):
        freeze(t, {"naive": "test('a', () => {})"})


def test_freeze_refuses_a_suite_that_is_not_one_or_more_test_files(tmp_path):
    """An arm that produced no suite must not reach `write_text`.

    Without this guard, `write_text(None)` raises `TypeError` from deep inside the freeze
    gate -- a crash that reads like a harness bug rather than "this arm has no suite".
    An EMPTY LIST is the same mistake spelt differently now that a suite is a file list,
    and a list holding a non-string is how a caller passing `contributors` (dicts) instead
    of their texts would present itself.
    """
    t = _task(tmp_path)
    write_metadata(t, [])
    for bad in (None, [], [None], [{"suite": "x"}]):
        with pytest.raises(LeakageError, match="not one or more test files"):
            freeze(t, {"naive": bad})

