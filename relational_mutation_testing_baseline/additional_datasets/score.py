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


# --- reporting -------------------------------------------------------------------
# Arm S (seeded faults) and Arm N (harvested LLM migrations) are reported separately and
# never pooled. If an arm beats the naive baseline on S but not on N, that is the finding:
# the advantage would be mutation-family resemblance rather than detection power.

ARMS = ("mutation", "naive", "existing", "ceiling")
ARM_LABEL = {"mutation": "ARM M (mutation-guided)", "naive": "ARM N (naive LLM)",
             "existing": "ARM A (developer tests)", "ceiling": "ARM C (sees the diff)"}


def by_arm(rows):
    """{arm: {"S": metrics, "N": metrics, "overall": metrics}}."""
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows if r["arm"] == arm]
        good = [r for r in mine if r["label"] == "GOOD"]
        entry = {"overall": metrics(mine)}
        for pop in ("S", "N"):
            bad = [r for r in mine if r["label"] == "BAD" and r.get("bad_arm") == pop]
            if bad:
                entry[pop] = metrics(bad + good)
        out[arm] = entry
    return out


def by_fault_class(rows):
    """Spec §16: detection rate per semantic fault category, BAD candidates only."""
    bad = [r for r in rows if r["label"] == "BAD" and r.get("fault_category")]
    out = []
    for cat in sorted({r["fault_category"] for r in bad}):
        mine = [r for r in bad if r["fault_category"] == cat]
        hit = [r for r in mine if r["verdict"] == REJECT]
        out.append({"fault_category": cat, "number": len(mine), "detected": len(hit),
                    "detection_rate": _ratio(len(hit), len(mine))})
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def report(rows, title):
    per = by_arm(rows)
    lines = [f"# Good-vs-bad classification: {title}", "",
             "The headline is BRR at matched GAR -- an arm that rejects more BAD while also",
             "rejecting more GOOD has not won anything. ARM A is excluded from GAR",
             "comparison: its GAR is definitional, since GOOD admission required its suite",
             "to pass (spec §3.6).", "",
             "| arm | population | n GOOD | n BAD | GAR | BRR | AP | precision | F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in [a for a in ARMS if a in per] + [a for a in per if a not in ARMS]:
        for pop in ("S", "N", "overall"):
            m = per[arm].get(pop)
            if not m:
                continue
            name = {"S": "seeded", "N": "harvested", "overall": "all"}[pop]
            gar = "definitional" if arm == "existing" else _pct(m["gar"])
            lines.append(f"| {ARM_LABEL.get(arm, arm)} | {name} | {m['n_good']} | "
                         f"{m['n_bad']} | {gar} | {_pct(m['brr'])} | {_pct(m['ap'])} | "
                         f"{_pct(m['precision'])} | {_pct(m['f1'])} |")
    lines += ["", "## Detection rate by fault category (spec §16)", "",
              "| fault category | number | detected | detection rate |",
              "|---|---:|---:|---:|"]
    for r in by_fault_class(rows):
        lines.append(f"| {r['fault_category']} | {r['number']} | {r['detected']} | "
                     f"{_pct(r['detection_rate'])} |")
    return "\n".join(lines) + "\n"
