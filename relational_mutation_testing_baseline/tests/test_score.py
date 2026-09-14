import pytest

from additional_datasets.score import (ACCEPT, BEHAVIOURAL, ERROR, LOAD_FAILED, REJECT,
                                       STRUCTURAL, confusion, metrics, verdict)
from additional_datasets.spec import SuiteResult


def test_green_suite_accepts():
    r = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=1.0)
    assert verdict(r, {"a"}) == (ACCEPT, "")


def test_failing_admissible_case_is_a_behavioural_rejection():
    r = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=1.0)
    assert verdict(r, {"a"}) == (REJECT, BEHAVIOURAL)


def test_a_case_outside_the_admissible_set_cannot_reject():
    # A case that already failed the trusted P0 would "reject" every candidate, so it
    # must not fire. `a` is present and green, so the suite did run to completion.
    r = SuiteResult(status="test_failed", cases={"a": "ok", "b": "FAILED"}, rc=1,
                    seconds=1.0)
    assert verdict(r, {"a"}) == (ACCEPT, "")


def test_a_run_cut_short_before_an_admissible_case_is_a_detection():
    # A SIGSEGV or stack overflow in mutated code takes the process down mid-run and the
    # remaining cases are never reported. That is a detection, not a miss.
    r = SuiteResult(status="test_failed", cases={"a": "ok"}, rc=139, seconds=1.0)
    assert verdict(r, {"a", "b"}) == (REJECT, BEHAVIOURAL)


def test_build_error_is_structural_not_behavioural():
    r = SuiteResult(status="build_error", cases={}, rc=1, seconds=1.0)
    assert verdict(r, {"a"}) == (REJECT, STRUCTURAL)


def test_timeout_rejects_behaviourally():
    r = SuiteResult(status="timeout", cases={}, rc=124, seconds=600.0)
    assert verdict(r, {"a"}) == (REJECT, BEHAVIOURAL)


def test_no_tests_is_unscorable():
    r = SuiteResult(status="no_tests", cases={}, rc=0, seconds=0.1)
    assert verdict(r, set())[0] == ERROR


def test_confusion_counts_reject_as_the_positive_class():
    rows = [{"label": "BAD", "verdict": REJECT},    # TP
            {"label": "GOOD", "verdict": REJECT},   # FP
            {"label": "BAD", "verdict": ACCEPT},    # FN
            {"label": "GOOD", "verdict": ACCEPT},   # TN
            {"label": "GOOD", "verdict": LOAD_FAILED}]  # excluded
    assert confusion(rows) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "unscorable": 1}


def test_metrics_match_the_spec_12_definitions():
    rows = ([{"label": "BAD", "verdict": REJECT}] * 8
            + [{"label": "BAD", "verdict": ACCEPT}] * 2
            + [{"label": "GOOD", "verdict": ACCEPT}] * 9
            + [{"label": "GOOD", "verdict": REJECT}] * 1)
    m = metrics(rows)
    assert m["brr"] == 0.8           # 8 of 10 BAD rejected
    assert m["gar"] == 0.9           # 9 of 10 GOOD accepted
    assert m["frr"] == pytest.approx(0.1)          # 1 - gar
    assert m["ap"] == pytest.approx(9 / 11)        # 9 GOOD accepted / 11 accepted
    assert m["precision"] == pytest.approx(8 / 9)  # 8 TP / 9 rejections
    assert m["recall"] == 0.8
    assert m["accuracy"] == pytest.approx(0.85)
    assert m["balanced_accuracy"] == pytest.approx(0.85)
    assert m["n_good"] == 10 and m["n_bad"] == 10


def test_metrics_are_none_not_zero_when_a_class_is_empty():
    m = metrics([{"label": "GOOD", "verdict": ACCEPT}])
    assert m["brr"] is None
    assert m["gar"] == 1.0
