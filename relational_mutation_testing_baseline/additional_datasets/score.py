"""Verdicts and the spec's §12 metrics.

A suite is a classifier over candidates: every test passes -> ACCEPT, one fails -> REJECT.
Two refinements matter, both learned from `pmb/validate.py` and the 2026-09-08 analysis:

- Only *admissible* cases can reject. A case that already failed the trusted P0 would
  reject every candidate including the correct ones, so it is evidence about the test,
  not about the candidate.
- A structural rejection (the suite would not build or load) is a different claim from a
  behavioural one (an assertion failed). Separating them moved recall from 0.74 to 0.23
  in the C<->Rust study; it is not optional.
"""
ACCEPT, REJECT, LOAD_FAILED, ERROR = "accept", "reject", "load_failed", "error"
# LOAD_FAILED is never returned by `verdict()`; the runner assigns it directly when a
# suite could not be loaded against a candidate at all. It is unscorable, like ERROR.
BEHAVIOURAL, STRUCTURAL = "behavioural", "structural"


def verdict(result, admissible):
    """(verdict, rejection_kind). `admissible` = case names that passed the trusted P0."""
    if result.status == "green":
        return ACCEPT, ""
    if result.status == "build_error":
        return REJECT, STRUCTURAL
    if result.status == "timeout":
        return REJECT, BEHAVIOURAL
    if result.status in ("no_tests", "error"):
        return ERROR, ""
    fired = [c for c in result.failed if c in admissible]
    if fired:
        return REJECT, BEHAVIOURAL
    # Ran but stopped early: a crash took the process down mid-run, which is a detection.
    missing = [c for c in admissible if c not in result.cases]
    if missing and result.rc:
        return REJECT, BEHAVIOURAL
    return ACCEPT, ""


def _scorable(rows):
    return [r for r in rows if r["verdict"] in (ACCEPT, REJECT)]


def confusion(rows):
    """REJECT is the positive class: the suite claims the candidate is wrong."""
    s = _scorable(rows)
    return {"tp": sum(1 for r in s if r["label"] == "BAD" and r["verdict"] == REJECT),
            "fp": sum(1 for r in s if r["label"] == "GOOD" and r["verdict"] == REJECT),
            "fn": sum(1 for r in s if r["label"] == "BAD" and r["verdict"] == ACCEPT),
            "tn": sum(1 for r in s if r["label"] == "GOOD" and r["verdict"] == ACCEPT),
            "unscorable": len(rows) - len(s)}


def _ratio(num, den):
    return None if not den else num / den


def metrics(rows):
    c = confusion(rows)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n_good, n_bad, n = tn + fp, tp + fn, tp + fp + fn + tn
    gar, brr = _ratio(tn, n_good), _ratio(tp, n_bad)
    bal = None if gar is None or brr is None else (gar + brr) / 2
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, n_bad)
    # F1 is None only when precision or recall is *undefined* (an empty class). When both
    # are legitimately 0.0 -- nothing rejected, or nothing to reject -- F1 is 0.0, the
    # sklearn convention. The explicit zero guard avoids 0/0 in the harmonic mean.
    if prec is None or rec is None:
        f1 = None
    elif prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {**c, "n_good": n_good, "n_bad": n_bad, "n": n,
            "gar": gar, "brr": brr, "ap": _ratio(tn, tn + fn),
            "frr": None if gar is None else 1 - gar,
            "accuracy": _ratio(tp + tn, n), "balanced_accuracy": bal,
            "precision": prec, "recall": rec, "f1": f1}
