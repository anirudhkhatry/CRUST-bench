"""run_task's arm-registration logic, isolated from real LLM/vitest calls.

Every external dependency of `run_task` is monkeypatched so these tests exercise only the
control flow: which arms get registered into `suites`/`admissible` (and therefore reach
`freeze`/`score_task`), not the arms' own generation logic (covered by test_arms.py) or
oracle preparation (covered by test_js_oracle.py).
"""
import json

import pytest

from additional_datasets.benchmarks import refactorbench_js as rjs


def _oracle(admissible_oracle=True):
    return {"oracle_text": "ORACLE_SUITE",
            "admissible_oracle": ["case_a"] if admissible_oracle else [],
            "reason": ""}


def _nrec(suite="", admissible=(), contributors=(), draws=6, composite="green"):
    """A complete ARM N record, as `naive.generate` really returns one."""
    return {"suite": suite, "n_draws": draws, "n_kept": len(contributors),
            "kept": [c["index"] for c in contributors],
            "contributors": [dict(c) for c in contributors],
            "rejected": [], "compiles": bool(suite), "on_p0": None,
            "admissible": list(admissible),
            "composite_status": composite if suite else "empty",
            "dropped_for_collision": []}


def _mrec(suite="", admissible=(), contributors=(), draws=6, composite="green"):
    """A complete ARM M record, as `mutation.generate` really returns one."""
    return {"suite": suite, "n_mutants": len(contributors),
            "n_mutants_requested": draws, "n_kept": len(contributors), "kept": [],
            "contributors": [dict(c) for c in contributors], "rejected": [],
            "translate_outcomes": {}, "public_surface": [], "on_p0": None,
            "admissible": list(admissible),
            "composite_status": composite if suite else "empty",
            "dropped_for_collision": []}


def _wire_common(monkeypatch):
    monkeypatch.setattr(rjs, "_sources", lambda name: {"app/index.js": "export const x = 1;"})
    monkeypatch.setattr(rjs.JS, "fixture_config", lambda name: {"targetFile": "app/index.js"})
    monkeypatch.setattr(rjs.JS, "import_path", lambda name: "./app/index.js")
    monkeypatch.setattr(rjs.JS, "stage", lambda name, dest, sources=None: dest)
    monkeypatch.setattr(rjs.js_oracle, "prepare_oracle", lambda name, ws: _oracle())
    monkeypatch.setattr(rjs, "propose_candidate", lambda *a, **k: None)  # no candidates
    monkeypatch.setattr(rjs, "seed_faults", lambda *a, **k: [])

    frozen, scored = {}, {}

    def fake_freeze(task, suites):
        frozen.update(suites)
        return {k: "hash" for k in suites}

    def fake_score_task(task, candidates, suites, admissible, adapter, workspace, **kw):
        scored.update(suites=dict(suites), admissible=dict(admissible), kw=dict(kw))
        return []

    monkeypatch.setattr(rjs, "freeze", fake_freeze)
    monkeypatch.setattr(rjs, "score_task", fake_score_task)
    return frozen, scored


def test_arm_n_is_omitted_when_it_compiles_but_admits_nothing(tmp_path, monkeypatch):
    """A suite that compiles but fails every assertion on P0 has an empty admissible set.

    If run_task registered ARM N unconditionally (the bug this guards against), "naive"
    would appear in both `frozen` and `scored["suites"]` below, and the run would go on to
    hand `score_task` an arm with an empty admissible set -- which the real `score_task`
    refuses outright (`LeakageError`), discarding every paid call already spent on this
    task's candidates. This test fails on that unconditional-registration code because
    "naive" would be present in `frozen`/`scored["suites"]`/`scored["admissible"]`.
    """
    frozen, scored = _wire_common(monkeypatch)
    # ARM N: compiles, but every case failed on P0 -- admissible is empty.
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: (
        "test('a', () => {});",
        _nrec("test('a', () => {});", admissible=[], composite="test_failed")))
    # ARM M: produces nothing either, so only ARM N's registration is under test here.
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))

    rows = rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert "naive" not in frozen
    assert "naive" not in scored["suites"]
    assert "naive" not in scored["admissible"]
    # `rows == []` was dropped: it only confirmed that the faked `score_task`'s return
    # value ([]) is forwarded unchanged, which is true no matter what run_task's own logic
    # does -- the fake makes it impossible for this assertion to fail.


def test_arm_n_is_registered_when_it_admits_at_least_one_case(tmp_path, monkeypatch):
    """The positive case: a suite with a non-empty admissible set is still registered.

    This fails if the fix over-corrects into omitting ARM N unconditionally -- "naive"
    would then be absent from `frozen`/`scored["suites"]` even though it has a real,
    scorable admissible set.
    """
    frozen, scored = _wire_common(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: (
        "test('a', () => { expect(1).toBe(1); });",
        _nrec("test('a', () => { expect(1).toBe(1); });", admissible=["a"])))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert frozen["naive"] == "test('a', () => { expect(1).toBe(1); });"
    assert scored["suites"]["naive"] == "test('a', () => { expect(1).toBe(1); });"
    assert scored["admissible"]["naive"] == ["a"]


def test_arm_a_is_the_whole_oracle_suite_frozen_and_scored(tmp_path, monkeypatch):
    """Spec §4.4: the developer suite IS the oracle, so ARM A is that same whole text.

    This fails on the superseded splitting code, which froze `existing_text` -- a
    skip-masked HALF of the holdout, and only when `arm_a_usable` was true. Under
    `_oracle()` (which no longer reports either field) ARM A would then be absent from
    `frozen` entirely, and `existing_text`/`existing_admissible` would not reach
    `score_task`.
    """
    frozen, scored = _wire_common(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: ("", _nrec()))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert frozen["existing"] == "ORACLE_SUITE"
    assert scored["kw"]["existing_text"] == "ORACLE_SUITE"
    assert scored["kw"]["existing_admissible"] == ["case_a"]


# --- matched test count, the draws knob, and single-label fixtures ---------------------

def _wire_run_suite(monkeypatch, cases=("a",)):
    """`assemble` runs the composite through the adapter, so the matched rebuild needs a
    real-shaped SuiteResult back."""
    from additional_datasets.spec import SuiteResult
    monkeypatch.setattr(rjs.JS, "run_suite", lambda ws, text, name: SuiteResult(
        status="green", cases={c: "ok" for c in cases}, rc=0, seconds=0.1))


def _contrib(i):
    return {"index": i, "suite": f"test('c{i}', () => {{ expect(1).toBe(1); }});"}


def _meta(tmp_path):
    return json.loads(
        (tmp_path / "refactorbench_js" / "fake_fixture" / "metadata.json").read_text())


def test_both_arms_are_truncated_to_the_smaller_kept_count(tmp_path, monkeypatch):
    """The user's headline comparison: equal numbers of generated tests.

    ARM N keeps 3 contributors, ARM M keeps 1. The matched arms must both assemble from
    exactly 1, earliest first. Fails if truncation is missing (`naive_matched` would hold
    all three contributors and `n_kept_matched` would be 3), if it truncates to the max
    instead of the min, or if it reorders (the matched suite would be `c2`, not `c0`).
    """
    frozen, scored = _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    n_contribs = [_contrib(0), _contrib(1), _contrib(2)]
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: (
        "N_SUITE", _nrec("N_SUITE", admissible=["a"], contributors=n_contribs)))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: (
        "M_SUITE", _mrec("M_SUITE", admissible=["a"], contributors=[_contrib(9)])))

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert scored["suites"]["naive_matched"] == n_contribs[0]["suite"]
    assert scored["suites"]["mutation_matched"] == _contrib(9)["suite"]
    # The raw arms are still scored beside the matched ones -- both variants are reported.
    assert scored["suites"]["naive"] == "N_SUITE"
    assert scored["suites"]["mutation"] == "M_SUITE"
    # ...and every one of the four goes through the same freeze gate.
    assert set(frozen) == {"naive", "mutation", "naive_matched", "mutation_matched",
                           "existing"}
    arms = _meta(tmp_path)["arms"]
    assert arms["naive"]["n_kept_raw"] == 3 and arms["naive"]["n_kept_matched"] == 1
    assert arms["mutation"]["n_kept_raw"] == 1 and arms["mutation"]["n_kept_matched"] == 1
    # The size actually executed on P0, which is what "same number of tests" means.
    assert arms["naive"]["n_cases_p0_matched"] == 1
    assert arms["mutation"]["n_cases_p0_matched"] == 1


def test_an_arm_that_kept_nothing_leaves_both_matched_arms_empty(tmp_path, monkeypatch):
    """min(kept_M, kept_N) == 0: neither matched arm can be scored, and neither is
    registered empty.

    Fails if the truncation falls back to "use everything" when one arm is empty, which
    would put ARM N's whole suite up against no ARM M suite at all.
    """
    frozen, scored = _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: (
        "N_SUITE", _nrec("N_SUITE", admissible=["a"], contributors=[_contrib(0)])))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert "naive_matched" not in scored["suites"]
    assert "mutation_matched" not in scored["suites"]
    assert scored["suites"]["naive"] == "N_SUITE"


def test_the_draws_knob_sets_both_arms(tmp_path, monkeypatch):
    """One parameter, both arms. Fails if `run_task` stops threading it (the arms would be
    called with their defaults, 6, and a cheap pilot could not shrink them)."""
    _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    seen = {}

    def fake_naive(*a, **k):
        seen["n_draws"] = k.get("n_draws")
        return "", _nrec(draws=k.get("n_draws"))

    def fake_mut(*a, **k):
        seen["n_mutants"] = k.get("n_mutants")
        seen["target"] = k.get("target")
        return "", _mrec(draws=k.get("n_mutants"))

    monkeypatch.setattr(rjs.naive, "generate", fake_naive)
    monkeypatch.setattr(rjs.mutation, "generate", fake_mut)

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, draws=2,
                 log=lambda *a: None)

    assert seen["n_draws"] == seen["n_mutants"] == 2
    # ARM M also needs the target file, whose exports are the translate step's surface.
    assert seen["target"] == "app/index.js"
    assert _meta(tmp_path)["draws"] == 2


def test_arms_given_different_draw_counts_abort_the_task(tmp_path, monkeypatch):
    """The equality of the two counts IS the matched control, so it is asserted.

    Fails if the assertion is removed: `run_task` would quietly compare 6 ARM N draws
    against 3 ARM M mutants and report the result as a controlled comparison.
    """
    _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: ("", _nrec(draws=6)))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec(draws=3)))

    with pytest.raises(AssertionError, match="matched control is broken"):
        rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, draws=6,
                     log=lambda *a: None)


def _wire_candidates(monkeypatch, labels):
    """Hand `run_task` a scripted sequence of candidate labels.

    Each draw produces a distinct source tree (so none is mistaken for P0) and the fake
    oracle labels it from `labels`, exhausting into None (= unusable reply).
    """
    from additional_datasets.spec import SuiteResult
    seq = list(labels)
    drawn = []

    def fake_propose(task, sources, target, model, temperature, tag):
        if not seq:
            return None
        lbl = seq.pop(0)
        drawn.append({"model": model, "temperature": temperature, "label": lbl})
        # +100 so no draw coincides with P0 (`export const x = 1;`) and gets discarded by
        # the identity guard -- that guard has its own test.
        return {"sources": {**sources,
                            "app/index.js": f"export const x = {len(drawn) + 100};"},
                "model": model, "temperature": temperature}

    def fake_label(adapter, task, sources, oracle_text, ws):
        return drawn[-1]["label"], None

    monkeypatch.setattr(rjs, "propose_candidate", fake_propose)
    monkeypatch.setattr(rjs, "label", fake_label)
    monkeypatch.setattr(rjs, "score_ceiling", None, raising=False)
    monkeypatch.setattr(rjs.ceiling, "score_ceiling", lambda *a, **k: [])
    return drawn


def test_a_good_only_fixture_is_recorded_as_unusable(tmp_path, monkeypatch):
    """A fixture with no BAD candidate gives every arm BRR n/a.

    Fails if `single_label` is not recorded -- the fixture would be pooled into the
    headline table as if it were a measurement, silently flattering whichever arm rejects
    least.
    """
    _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: ("", _nrec()))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))
    _wire_candidates(monkeypatch, ["GOOD"] * len(rjs.CANDIDATE_SWEEP))
    # Seeding is the BAD fallback; here it yields nothing, so GOOD-only survives.
    monkeypatch.setattr(rjs, "seed_faults", lambda *a, **k: [])

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert _meta(tmp_path)["single_label"] == "GOOD"


def test_a_fixture_with_both_labels_is_not_flagged(tmp_path, monkeypatch):
    """The positive case. Fails if `single_label` is set unconditionally, which would
    exclude every fixture from the report."""
    _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: ("", _nrec()))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))
    _wire_candidates(monkeypatch, ["GOOD", "BAD"])
    monkeypatch.setattr(rjs, "seed_faults", lambda *a, **k: [])

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    meta = _meta(tmp_path)
    assert meta["single_label"] is None
    assert meta["candidate_routes"]["harvested_good"] == 1
    assert meta["candidate_routes"]["harvested_bad"] == 1
    # The sweep stopped as soon as both floors and both quotas were met, rather than
    # paying for all 12 draws. Fails if the early-exit condition is dropped.
    assert meta["candidate_routes"]["sweep_draws_used"] == 2


def test_the_sweep_keeps_drawing_until_both_labels_exist(tmp_path, monkeypatch):
    """With n_good=n_bad=1, a run of GOODs must not stop at the first GOOD.

    Fails on a loop that breaks once `n_good` is satisfied without checking that a BAD
    exists: `sweep_draws_used` would be 1 and the fixture would be single-labelled.
    """
    _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: ("", _nrec()))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))
    _wire_candidates(monkeypatch, ["GOOD", "GOOD", "GOOD", "BAD"])
    monkeypatch.setattr(rjs, "seed_faults", lambda *a, **k: [])

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    meta = _meta(tmp_path)
    assert meta["single_label"] is None
    assert meta["candidate_routes"]["sweep_draws_used"] == 4


def test_p0_returned_unchanged_is_never_admitted_as_a_candidate(tmp_path, monkeypatch):
    """"A correct implementation is a good implementation" must not become "use P0".

    P0 passes the oracle trivially and every arm's suite is already gated green-on-P0, so
    an identity candidate hands every arm GAR 100% for free. Fails if the identity guard
    is removed: the candidate would be labelled GOOD and recorded.
    """
    _wire_common(monkeypatch)
    _wire_run_suite(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: ("", _nrec()))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: ("", _mrec()))
    monkeypatch.setattr(rjs.ceiling, "score_ceiling", lambda *a, **k: [])
    monkeypatch.setattr(rjs, "seed_faults", lambda *a, **k: [])
    monkeypatch.setattr(rjs, "label", lambda *a, **k: ("GOOD", None))
    # The model echoes P0 back verbatim, every time.
    monkeypatch.setattr(rjs, "propose_candidate", lambda task, sources, target, model,
                        temperature, tag: {"sources": dict(sources), "model": model,
                                           "temperature": temperature})

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert _meta(tmp_path)["candidates"] == []


def test_the_candidate_sweep_is_one_model_at_several_temperatures():
    """Change 3: a temperature sweep, not a model mix -- so GOOD vs BAD is a property of
    the sampling temperature. Fails if a second model is reintroduced into the list, or if
    the list collapses to a single temperature (which would give no spread)."""
    assert {m for m, _ in rjs.MODELS} == {rjs.CANDIDATE_MODEL}
    assert "haiku" in rjs.CANDIDATE_MODEL
    assert len({t for _, t in rjs.MODELS}) >= 3
