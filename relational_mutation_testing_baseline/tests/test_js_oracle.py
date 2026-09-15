import pytest

from additional_datasets.adapters import js_oracle as O
from additional_datasets.adapters import js_vitest as JS

pytestmark = pytest.mark.skipif(not JS.VITEST.exists(),
                                reason="js runtime not installed")


def test_prepare_oracle_splits_binary_search_into_two_verified_halves(tmp_path):
    got = O.prepare_oracle("binary_search", tmp_path / "ws")
    assert got["arm_a_usable"] is True, got["reason"]
    assert got["admissible_oracle"] and got["admissible_existing"]
    assert not (set(got["admissible_oracle"]) & set(got["admissible_existing"]))


def test_the_two_halves_reconstruct_the_whole_suite(tmp_path):
    got = O.prepare_oracle("binary_search", tmp_path / "ws")
    ws = JS.stage("binary_search", tmp_path / "full")
    whole = JS.run_suite(ws, JS.holdout_text("binary_search"), "whole")
    assert whole.status == "green"
    union = set(got["admissible_oracle"]) | set(got["admissible_existing"])
    assert union == set(whole.cases)


def test_an_unsplittable_suite_marks_arm_a_unusable_rather_than_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(JS, "holdout_text",
                        lambda n: "import { test, expect } from 'vitest';\n"
                                  "test('only', () => { expect(1).toBe(1); });\n")
    got = O.prepare_oracle("binary_search", tmp_path / "ws")
    assert got["arm_a_usable"] is False
    assert "unsplittable" in got["reason"]
    # The oracle still works -- only Baseline A is lost.
    assert got["admissible_oracle"]
