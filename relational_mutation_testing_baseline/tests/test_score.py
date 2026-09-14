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


def test_f1_is_zero_not_none_when_nothing_was_caught():
    # tp=0 with both classes non-empty: precision and recall are a real 0.0, not undefined.
    rows = ([{"label": "BAD", "verdict": ACCEPT}] * 4
            + [{"label": "GOOD", "verdict": REJECT}] * 3
            + [{"label": "GOOD", "verdict": ACCEPT}] * 2)
    m = metrics(rows)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_f1_is_none_when_a_class_is_empty():
    m = metrics([{"label": "GOOD", "verdict": ACCEPT}])
    assert m["recall"] is None
    assert m["f1"] is None


from additional_datasets.score import by_arm, by_fault_class, report


def _row(arm, label, v, bad_arm=None, fault=None):
    return {"arm": arm, "label": label, "verdict": v, "bad_arm": bad_arm,
            "fault_category": fault}


def test_by_arm_splits_seeded_from_harvested_and_never_pools_them():
    rows = [_row("mutation", "BAD", REJECT, "S", "ordering difference"),
            _row("mutation", "BAD", ACCEPT, "N", "wrong default value"),
            _row("mutation", "GOOD", ACCEPT)]
    out = by_arm(rows)
    assert out["mutation"]["S"]["brr"] == 1.0
    assert out["mutation"]["N"]["brr"] == 0.0
    # The GOOD rows are shared by both BAD populations, so GAR is reported once.
    assert out["mutation"]["overall"]["gar"] == 1.0
    assert "pooled" not in out["mutation"]


def test_by_fault_class_reports_detection_rate_per_category():
    rows = [_row("mutation", "BAD", REJECT, "S", "encoding difference"),
            _row("mutation", "BAD", REJECT, "S", "encoding difference"),
            _row("mutation", "BAD", ACCEPT, "S", "ordering difference")]
    out = {r["fault_category"]: r for r in by_fault_class(rows)}
    assert out["encoding difference"]["number"] == 2
    assert out["encoding difference"]["detected"] == 2
    assert out["encoding difference"]["detection_rate"] == 1.0
    assert out["ordering difference"]["detection_rate"] == 0.0


def test_report_renders_and_flags_the_matched_gar_comparison():
    rows = [_row("mutation", "BAD", REJECT, "S", "ordering difference"),
            _row("mutation", "GOOD", ACCEPT),
            _row("existing", "BAD", ACCEPT, "S", "ordering difference"),
            _row("existing", "GOOD", ACCEPT)]
    text = report(rows, title="pilot")
    assert "pilot" in text
    assert "BRR" in text and "GAR" in text
    # Baseline A must be marked, never silently compared on GAR (spec §3.6).
    assert "ARM A" in text
    assert "GAR is definitional" in text
