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
    out = {(r["arm"], r["bad_arm"], r["fault_category"]): r for r in by_fault_class(rows)}
    assert out[("mutation", "S", "encoding difference")]["number"] == 2
    assert out[("mutation", "S", "encoding difference")]["detected"] == 2
    assert out[("mutation", "S", "encoding difference")]["detection_rate"] == 1.0
    assert out[("mutation", "S", "ordering difference")]["detection_rate"] == 0.0


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


def test_arm_a_renders_both_gar_and_brr_as_definitional():
    """Spec §4.4: ARM A is the oracle that assigned the labels, so BOTH figures are 1.0
    by construction and neither may render as a measurement.

    The ARM A rows below are exactly what the oracle produces -- it accepts the GOOD
    candidate and rejects the BAD one -- so under the previous code, which special-cased
    GAR only, ARM A's BRR cell read "100%" and invited comparison with ARM M's 0%. That is
    the input this assertion fails on. ARM M's own BRR must still be a percentage, or the
    fix has swallowed the comparison it was protecting.
    """
    rows = [_row("existing", "BAD", REJECT, "S", "ordering difference"),
            _row("existing", "GOOD", ACCEPT),
            _row("mutation", "BAD", ACCEPT, "S", "ordering difference"),
            _row("mutation", "GOOD", ACCEPT)]
    lines = report(rows, title="t").split("\n")
    a_line = [l for l in lines if "ARM A" in l and "seeded" in l][0]
    m_line = [l for l in lines if "ARM M" in l and "seeded" in l][0]
    assert a_line.split("|")[6].strip() == "definitional"      # GAR
    assert a_line.split("|")[7].strip() == "definitional"      # BRR
    assert m_line.split("|")[7].strip() == "0%"
    assert "so is its BRR" in report(rows, title="t")


def test_report_never_emits_a_pooled_bad_figure():
    # S is caught, N is missed. A pooled BRR would read 50% and erase the divergence.
    rows = [_row("mutation", "BAD", REJECT, "S", "ordering difference"),
            _row("mutation", "BAD", ACCEPT, "N", "wrong default value"),
            _row("mutation", "GOOD", ACCEPT)]
    per = by_arm(rows)
    assert "brr" not in per["mutation"]["overall"]
    assert per["mutation"]["S"]["brr"] == 1.0
    assert per["mutation"]["N"]["brr"] == 0.0
    text = report(rows, title="t")
    # The "all" row is gone; S and N rows show 100% and 0% BRR respectively.
    assert "| all |" not in text
    # Verify the individual BRR values appear correctly without pooling. Substring
    # matching the whole line is not enough: "0%" is a substring of "100%", so a table
    # where N's BRR cell wrongly read "100%" (the S/N figures swapped) would still pass
    # `"0%" in n_line`. Parse the BRR cell out by position instead.
    lines = text.split('\n')
    s_line = [l for l in lines if 'seeded' in l and 'ARM M' in l][0]
    n_line = [l for l in lines if 'harvested' in l and 'ARM M' in l][0]
    brr_s = s_line.split('|')[7].strip()
    brr_n = n_line.split('|')[7].strip()
    # Fails if the S/N BRR figures were ever swapped or pooled, e.g. brr_s == "0%".
    assert brr_s == "100%"
    assert brr_n == "0%"


def test_by_fault_class_does_not_blend_arms():
    rows = [_row("mutation", "BAD", REJECT, "S", "ordering difference"),
            _row("existing", "BAD", ACCEPT, "S", "ordering difference")]
    got = {(r["arm"], r["fault_category"]): r for r in by_fault_class(rows)}
    assert got[("mutation", "ordering difference")]["detection_rate"] == 1.0
    assert got[("existing", "ordering difference")]["detection_rate"] == 0.0


# --- single-label fixtures and the matched arms ---------------------------------------

def _srow(task, arm, label, v, single_label=None):
    return {"task_id": task, "arm": arm, "label": label, "verdict": v,
            "rejection_kind": "" if v == ACCEPT else BEHAVIOURAL,
            "bad_arm": None if label == "GOOD" else "N", "fault_category": None,
            "single_label": single_label}


def test_a_single_label_task_is_excluded_from_the_tables_and_listed():
    """A GOOD-only fixture gives every arm BRR n/a; pooling it in still moves GAR.

    `t_good_only` contributes two GOOD candidates that ARM N accepts. If it were pooled,
    ARM N's GAR denominator would be 3 instead of 1 and the reported GAR would change from
    0% to 67% -- a fixture that measures nothing would be improving an arm's headline
    number. Fails on any implementation that does not filter by `single_label`.
    """
    from additional_datasets.score import report, single_label_tasks
    rows = [_srow("t_ok", "naive", "GOOD", REJECT),
            _srow("t_ok", "naive", "BAD", REJECT),
            _srow("t_good_only", "naive", "GOOD", ACCEPT, single_label="GOOD"),
            _srow("t_good_only", "naive", "GOOD", ACCEPT, single_label="GOOD")]
    assert single_label_tasks(rows) == {"t_good_only": "GOOD"}
    text = report(rows, title="x")
    # Listed, with the reason.
    assert "t_good_only" in text
    assert "every candidate is GOOD" in text
    # ...and excluded: ARM N rejected its only comparable GOOD candidate, so GAR is 0%.
    naive_line = [ln for ln in text.splitlines()
                  if ln.startswith("| ARM N (naive LLM) |")][0]
    assert "| 1 |" in naive_line, naive_line     # n GOOD counted, not 3
    assert "| 0% |" in naive_line


def test_a_report_with_no_single_label_tasks_says_nothing_about_them():
    """Fails if the exclusion banner is emitted unconditionally, which would make every
    clean report look like it had unusable fixtures."""
    from additional_datasets.score import report
    text = report([_srow("t_ok", "naive", "GOOD", ACCEPT),
                   _srow("t_ok", "naive", "BAD", REJECT)], title="x")
    assert "UNUSABLE" not in text


def test_the_matched_arms_are_labelled_and_ordered_ahead_of_the_raw_ones():
    """The matched comparison is the headline, so it must be legible as such.

    Fails if `*_matched` arms fall through to the raw arm name (both pairs would render as
    "mutation_matched" with no explanation of what they are), or if they are ordered after
    the raw pair.
    """
    from additional_datasets.score import ARM_LABEL, ARMS, MATCHED_ARMS, report
    assert set(MATCHED_ARMS) <= set(ARM_LABEL)
    assert ARMS.index("mutation_matched") < ARMS.index("mutation")
    text = report([_srow("t", "mutation_matched", "GOOD", ACCEPT),
                   _srow("t", "mutation_matched", "BAD", REJECT),
                   _srow("t", "mutation", "GOOD", ACCEPT),
                   _srow("t", "mutation", "BAD", REJECT)], title="x")
    body = text.splitlines()
    first = [i for i, ln in enumerate(body) if "ARM M (mutation-guided, matched count)" in ln]
    later = [i for i, ln in enumerate(body) if ln.startswith("| ARM M (mutation-guided) |")]
    assert first and later and first[0] < later[0]


def test_report_lists_fixtures_that_have_no_matched_comparison():
    """A fixture where one arm admitted nothing has no matched pair, and the report must
    say which fixture and why -- otherwise the `*_matched` rows silently cover fewer
    fixtures than the raw rows.

    Fails if `matched_skipped` is not surfaced: `t_one_arm` would be absent from the text
    even though only its raw ARM N rows exist. The reason string is checked too, so a
    report that merely names the fixture (it already appears in the coverage lines) does
    not pass.
    """
    from additional_datasets.score import matched_skipped_tasks, report
    rows = [_srow("t_ok", "naive_matched", "GOOD", ACCEPT),
            _srow("t_ok", "naive_matched", "BAD", REJECT),
            {**_srow("t_one_arm", "naive", "GOOD", ACCEPT),
             "matched_skipped": "ARM N kept nothing"},
            {**_srow("t_one_arm", "naive", "BAD", REJECT),
             "matched_skipped": "ARM N kept nothing"}]
    assert matched_skipped_tasks(rows) == {"t_one_arm": "ARM N kept nothing"}
    text = report(rows, title="x")
    assert "NO matched comparison" in text
    assert "`t_one_arm`: ARM N kept nothing." in text


def test_a_report_with_no_skipped_matched_pairs_says_nothing_about_them():
    """Fails if the banner is unconditional, which would make every clean report claim a
    missing matched comparison."""
    from additional_datasets.score import report
    text = report([_srow("t_ok", "naive_matched", "GOOD", ACCEPT),
                   _srow("t_ok", "naive_matched", "BAD", REJECT)], title="x")
    assert "NO matched comparison" not in text
