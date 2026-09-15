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
# LOAD_FAILED is RESERVED and currently unproduced: neither `verdict()` nor the runner
# assigns it, because the adapters report a suite that will not load as `build_error`,
# which `verdict()` turns into a STRUCTURAL rejection. It is kept in the vocabulary (and
# out of `_scorable`) so that a row carrying it -- from an older rows.json, or from an
# adapter that does distinguish the two -- is excluded rather than silently counted.
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


def is_structural(row):
    """A rejection driven by shape rather than behaviour."""
    return row["verdict"] == REJECT and row.get("rejection_kind") == STRUCTURAL


def _rejects(row, behavioural_only):
    """Does this row count as a rejection under the requested reading?

    Under `behavioural_only`, a STRUCTURAL rejection is counted as an ACCEPT rather than
    dropped. Dropping it would move the row into `unscorable` and shrink the denominator,
    which silently changes GAR and AP as well -- the point of the mode is to change what
    counts as *evidence of misbehaviour*, not what counts as a scored candidate.
    """
    if row["verdict"] != REJECT:
        return False
    return not (behavioural_only and row.get("rejection_kind") == STRUCTURAL)


def n_structural(rows):
    """How many of these rows are structural rejections (spec §5's separated count)."""
    return sum(1 for r in _scorable(rows) if is_structural(r))


def confusion(rows, behavioural_only=False):
    """REJECT is the positive class: the suite claims the candidate is wrong.

    With `behavioural_only`, a structural rejection ("the suite would not load against
    this candidate") is not evidence about behaviour and is counted as an acceptance.
    """
    s = _scorable(rows)
    rej = [r for r in s if _rejects(r, behavioural_only)]
    acc = [r for r in s if not _rejects(r, behavioural_only)]
    return {"tp": sum(1 for r in rej if r["label"] == "BAD"),
            "fp": sum(1 for r in rej if r["label"] == "GOOD"),
            "fn": sum(1 for r in acc if r["label"] == "BAD"),
            "tn": sum(1 for r in acc if r["label"] == "GOOD"),
            "unscorable": len(rows) - len(s)}


def _ratio(num, den):
    return None if not den else num / den


def metrics(rows, behavioural_only=False):
    c = confusion(rows, behavioural_only)
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
            "n_structural": n_structural(rows),
            "n_tasks": len({r.get("task_id") for r in rows}),
            "behavioural_only": behavioural_only,
            "gar": gar, "brr": brr, "ap": _ratio(tn, tn + fn),
            "frr": None if gar is None else 1 - gar,
            "accuracy": _ratio(tp + tn, n), "balanced_accuracy": bal,
            "precision": prec, "recall": rec, "f1": f1}


# --- reporting -------------------------------------------------------------------
# Arm S (seeded faults) and Arm N (harvested LLM migrations) are reported separately and
# never pooled. If an arm beats the naive baseline on S but not on N, that is the finding:
# the advantage would be mutation-family resemblance rather than detection power.

ARMS = ("mutation_matched", "naive_matched", "mutation", "naive", "existing", "ceiling")
# ARM A is the developer suite, which since spec §4.4 is the hidden oracle itself: it
# assigned every GOOD/BAD label, so its verdicts are definitions rather than measurements.
# It is reported for completeness and excluded from every comparison.
DEFINITIONAL_ARM = "existing"
ARM_LABEL = {"mutation": "ARM M (mutation-guided)", "naive": "ARM N (naive LLM)",
             "mutation_matched": "ARM M (mutation-guided, matched count)",
             "naive_matched": "ARM N (naive LLM, matched count)",
             "existing": "ARM A (developer tests)", "ceiling": "ARM C (sees the diff)"}
# The comparison the user asked for: "at the same number of generated tests". The `*_raw`
# arms keep everything each arm's own gate admitted, so ARM M's and ARM N's suites differ
# in SIZE as well as in how they were selected; the `*_matched` arms truncate both to
# min(kept_M, kept_N) before assembling. Both are reported; the matched pair is the
# headline.
MATCHED_ARMS = ("mutation_matched", "naive_matched")


def by_arm(rows, behavioural_only=False):
    """{arm: {"S": metrics, "N": metrics, "overall": {...}}}.

    `overall` also carries the arm's task coverage. Arms are omitted PER TASK when they
    produced no scorable suite, and that omission is not random -- ARM N's suite is one
    monolithic file while ARM M assembles independently-compiled fragments -- so two arms
    in this table may be measured over different task sets. The count is reported so that
    comparison is visible rather than implicit (see `report`).
    """
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows if r["arm"] == arm]
        good = [r for r in mine if r["label"] == "GOOD"]
        # GAR is arm-wide: it describes how this arm treats GOOD candidates and does not
        # depend on which BAD population is scored beside it. Every BAD-derived figure is
        # deliberately absent here -- pooling seeded and harvested BADs into one BRR would
        # hide the divergence this benchmark exists to measure (design §5).
        arm_wide = metrics(good, behavioural_only)
        entry = {"overall": {"n_good": arm_wide["n_good"], "gar": arm_wide["gar"],
                             "frr": arm_wide["frr"],
                             "n_tasks": len({r.get("task_id") for r in mine}),
                             "tasks": sorted({str(r.get("task_id")) for r in mine}),
                             "n_structural": n_structural(mine)}}
        for pop in ("S", "N"):
            bad = [r for r in mine if r["label"] == "BAD" and r.get("bad_arm") == pop]
            if bad:
                entry[pop] = metrics(bad + good, behavioural_only)
        out[arm] = entry
    return out


def by_fault_class(rows, behavioural_only=False):
    """Design §16: detection rate per fault category, split by arm and BAD population.

    Nothing is pooled on either axis. Blending arms would hide which arm detects a class,
    which is the comparison this table exists to make; blending S and N would hide whether
    an advantage is mutation-family resemblance.
    """
    bad = [r for r in rows if r["label"] == "BAD" and r.get("fault_category")]
    out = []
    for key in sorted({(r["arm"], r.get("bad_arm"), r["fault_category"]) for r in bad},
                      key=lambda k: (k[0], str(k[1]), k[2])):
        arm, pop, cat = key
        mine = [r for r in bad if (r["arm"], r.get("bad_arm"), r["fault_category"]) == key]
        hit = [r for r in mine if _rejects(r, behavioural_only)]
        out.append({"arm": arm, "bad_arm": pop, "fault_category": cat,
                    "number": len(mine), "detected": len(hit),
                    "detection_rate": _ratio(len(hit), len(mine))})
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _ordered_arms(per):
    return [a for a in ARMS if a in per] + [a for a in per if a not in ARMS]


HEADER = ("| arm | population | n tasks | n GOOD | n BAD | GAR | BRR | AP | precision "
          "| F1 |")
RULE = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"


def _table(rows, behavioural_only):
    per = by_arm(rows, behavioural_only)
    lines = [HEADER, RULE]
    for arm in _ordered_arms(per):
        for pop in ("S", "N"):
            m = per[arm].get(pop)
            if not m:
                continue
            name = {"S": "seeded", "N": "harvested"}[pop]
            arm_wide = per[arm]["overall"]
            # ARM A *is* the oracle that assigned the labels (spec §4.4), so BOTH of its
            # headline figures are 1.0 by construction -- a GOOD candidate is one its
            # suite passed and a BAD one is one its suite failed. Rendering either as a
            # percentage invites it to be read as a measurement and compared against the
            # generated arms, which is precisely the comparison it cannot enter.
            gar = "definitional" if arm == DEFINITIONAL_ARM else _pct(arm_wide["gar"])
            brr = "definitional" if arm == DEFINITIONAL_ARM else _pct(m["brr"])
            lines.append(f"| {ARM_LABEL.get(arm, arm)} | {name} | "
                         f"{arm_wide['n_tasks']} | {m['n_good']} | "
                         f"{m['n_bad']} | {gar} | {brr} | {_pct(m['ap'])} | "
                         f"{_pct(m['precision'])} | {_pct(m['f1'])} |")
    return lines


def _coverage_lines(per):
    """State plainly whether the arms in the table were measured over the same tasks."""
    cover = {arm: per[arm]["overall"]["tasks"] for arm in _ordered_arms(per)}
    counts = ", ".join(f"{ARM_LABEL.get(a, a)}: {len(v)}" for a, v in cover.items())
    if len({tuple(v) for v in cover.values()}) <= 1:
        return [f"All arms cover the same {len(next(iter(cover.values()), []))} task(s) "
                f"({counts}).", ""]
    out = [f"**The arms do NOT cover the same tasks** ({counts}). An arm is omitted from a "
           "task when it produced no scorable suite, and that omission is not random -- so "
           "the rows below are not a like-for-like comparison and the per-arm task sets "
           "must be read first.", ""]
    for arm, tasks in cover.items():
        missing = sorted(set().union(*cover.values()) - set(tasks))
        if missing:
            out.append(f"- {ARM_LABEL.get(arm, arm)} is absent from: {', '.join(missing)}")
    return out + [""]


def _structural_lines(per):
    counts = {arm: per[arm]["overall"]["n_structural"] for arm in _ordered_arms(per)}
    body = ", ".join(f"{ARM_LABEL.get(a, a)}: {n}" for a, n in counts.items())
    return [f"Structural rejections (the suite would not load against the candidate), "
            f"per arm -- counted in the first table, not in the second: {body}.", ""]


def single_label_tasks(rows):
    """{task_id: "GOOD"|"BAD"} for tasks whose candidates all carried one label.

    Such a task cannot measure the comparison: with only GOOD candidates every arm's BRR
    is n/a, with only BAD every arm's GAR is n/a. Pooling its rows into the tables does not
    make it neutral -- it adds a population that only one half of the confusion matrix can
    see, which flatters whichever arm rejects more (or less). So it is excluded from the
    tables and listed instead.
    """
    return {r["task_id"]: r["single_label"] for r in rows if r.get("single_label")}


def comparable(rows):
    """The rows that can enter a table: everything from a task with BOTH labels."""
    return [r for r in rows if not r.get("single_label")]


def _single_label_lines(rows):
    single = single_label_tasks(rows)
    if not single:
        return []
    out = [f"> **{len(single)} fixture(s) are UNUSABLE for the comparison and are EXCLUDED "
           "from every table below.** Each produced candidates of only one label, so one "
           "half of the confusion matrix does not exist for it:", ""]
    for task, lbl in sorted(single.items()):
        missing = "BRR is n/a for every arm" if lbl == "GOOD" else "GAR is n/a for every arm"
        out.append(f"> - `{task}`: every candidate is {lbl} -- {missing}.")
    return out + [""]


def report(rows, title, failures=()):
    """The spec §5 tables. `failures` are tasks that errored and produced no rows.

    Two tables are emitted, not one. A rejection because the suite would not load against
    a candidate is a claim about the candidate's SHAPE; a failed assertion is a claim about
    its BEHAVIOUR. The 2026-09-08 analysis moved recall from 0.74 to 0.23 on exactly this
    distinction, so reporting only the first table would overstate every arm, and reporting
    only the second would hide how much of each arm's detection was structural.
    """
    all_rows, rows = rows, comparable(rows)
    per = by_arm(rows)
    lines = [f"# Good-vs-bad classification: {title}", ""]
    lines += _single_label_lines(all_rows)
    if failures:
        # A run with failed tasks must not render as a clean report: the arms below were
        # measured over the tasks that survived, which is a different population.
        lines += [f"> **{len(failures)} task(s) failed and contributed no rows.** The "
                  "tables below describe only the tasks that completed.", ""]
        for f in failures:
            lines.append(f"> - `{f.get('task')}`: {f.get('error')}")
        lines.append("")
    lines += ["**The headline rows are the two `matched count` arms**: ARM M keeps only",
              "killing tests and ARM N keeps green ones, so their raw suites differ in SIZE",
              "as well as in selection. The matched arms truncate both to the same number of",
              "kept contributors -- min(kept_M, kept_N), earliest first -- and re-assemble",
              "through the same shared helper, which is the comparison 'at the same number",
              "of generated tests'. The unmatched `ARM M`/`ARM N` rows are reported beside",
              "them: if the two comparisons disagree, the raw advantage was suite size.",
              "",
              "The headline is BRR at matched GAR -- an arm that rejects more BAD while also",
              "rejecting more GOOD has not won anything. The comparison is ARM M against",
              "ARM N. **ARM A is excluded from every comparison: its GAR is definitional AND",
              "so is its BRR**, because ARM A is the developer suite, which is the hidden",
              "oracle that assigned the labels -- a GOOD candidate is one it passed and a BAD",
              "one is one it failed (spec §4.4, superseding §3.6's claim that only its GAR was",
              "definitional). Its AP, precision and F1 are definitional for the same reason;",
              "they are printed as numbers only because they are shared table columns.",
              "`combined_tests_pass` is None throughout for the same reason: combining an arm",
              "with the oracle that defined the labels answers nothing.", ""]
    lines += _coverage_lines(per)
    lines += _structural_lines(per)
    lines += ["## All rejections", ""]
    lines += _table(rows, behavioural_only=False)
    lines += ["", "## Behavioural rejections only", "",
              "A structural rejection is counted here as an acceptance, not dropped: the "
              "candidate was still scored, so the denominators -- and therefore GAR and AP "
              "-- are the same as above (spec §5).", ""]
    lines += _table(rows, behavioural_only=True)
    lines += ["", "## Detection rate by fault category (spec §16), all rejections", "",
              "| arm | population | fault category | number | detected | detection rate |",
              "|---|---|---|---:|---:|---:|"]
    for r in by_fault_class(rows):
        arm_label = ARM_LABEL.get(r["arm"], r["arm"])
        pop_name = {"S": "seeded", "N": "harvested", None: "n/a"}[r.get("bad_arm")]
        lines.append(f"| {arm_label} | {pop_name} | {r['fault_category']} | {r['number']} | "
                     f"{r['detected']} | {_pct(r['detection_rate'])} |")
    return "\n".join(lines) + "\n"
