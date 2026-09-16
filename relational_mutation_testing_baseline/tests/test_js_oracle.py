"""The oracle is the WHOLE developer suite, and it is verified by execution on P0.

Since spec §4.4 there is no split: `prepare_oracle` returns the holdout verbatim and the
subset of its cases that pass on the fixture's own untouched sources. The splitter itself
is still correct and still covered -- see tests/test_js_split.py -- it is simply no longer
used to manufacture an independent Baseline A.
"""
import pytest

from additional_datasets.adapters import js_oracle as O
from additional_datasets.adapters import js_vitest as JS

pytestmark = pytest.mark.skipif(not JS.VITEST.exists(),
                                reason="js runtime not installed")


def test_the_oracle_is_the_whole_holdout_suite_unsplit(tmp_path):
    """Fails on the old splitting implementation: it returned a skip-masked half, whose
    text differs from `holdout_text` and whose admissible set is a strict subset of the
    whole suite's cases."""
    got = O.prepare_oracle("binary_search", tmp_path / "ws")
    whole = JS.holdout_text("binary_search")
    assert got["oracle_text"] == whole
    assert got["reason"] == ""

    ws = JS.stage("binary_search", tmp_path / "full")
    run = JS.run_suites(ws, [whole], "whole")
    assert run.status == "green"
    # Every case of the whole suite labels candidates, not half of them.
    assert got["admissible_oracle"] == sorted(run.cases)
    assert len(got["admissible_oracle"]) > 1
    # The names come from `run_suites`, so they are file-index prefixed -- the SAME
    # namespace a candidate is scored in. Using `run_suite` here instead would hand back
    # bare names, every one of which would then read as "missing" during scoring.
    assert all(c.startswith("0::") for c in got["admissible_oracle"])


def test_a_holdout_that_is_not_green_on_p0_is_reported_as_such(tmp_path, monkeypatch):
    """The P0-green verification, which is what proves a holdout can label anything.

    Fails if `prepare_oracle` ever stops executing the suite and just returns it: the
    substituted holdout below asserts 1 === 2, so a green `reason` and a non-empty
    `admissible_oracle` are both only obtainable by not running it.
    """
    monkeypatch.setattr(JS, "holdout_text",
                        lambda n: "import { test, expect } from 'vitest';\n"
                                  "test('wrong', () => { expect(1).toBe(2); });\n")
    got = O.prepare_oracle("binary_search", tmp_path / "ws")
    assert "not green on P0" in got["reason"]
    assert got["admissible_oracle"] == []


def test_a_single_case_holdout_is_still_a_usable_oracle(tmp_path, monkeypatch):
    """What used to be fatal ("unsplittable: fewer than two leaf cases") no longer is:
    with no split, one passing case is a perfectly good labeller. Fails if the
    unsplittable guard is ever reinstated, which would empty `admissible_oracle`."""
    monkeypatch.setattr(JS, "holdout_text",
                        lambda n: "import { test, expect } from 'vitest';\n"
                                  "test('only', () => { expect(1).toBe(1); });\n")
    got = O.prepare_oracle("binary_search", tmp_path / "ws")
    assert got["admissible_oracle"] == ["0::only"]
    assert got["reason"] == ""
