"""run_task's arm-registration logic, isolated from real LLM/vitest calls.

Every external dependency of `run_task` is monkeypatched so these tests exercise only the
control flow: which arms get registered into `suites`/`admissible` (and therefore reach
`freeze`/`score_task`), not the arms' own generation logic (covered by test_arms.py) or
the oracle split (covered by test_js_oracle.py).
"""
from additional_datasets.benchmarks import refactorbench_js as rjs


def _oracle(admissible_oracle=True, arm_a_usable=False):
    return {"oracle_text": "ORACLE_SUITE", "existing_text": "", "assignment": {},
            "admissible_oracle": ["case_a"] if admissible_oracle else [],
            "admissible_existing": [], "arm_a_usable": arm_a_usable, "reason": ""}


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
        scored.update(suites=dict(suites), admissible=dict(admissible))
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
        "test('a', () => {});", {"suite": "test('a', () => {});", "compiles": True,
                                 "attempts": [], "on_p0": None, "admissible": []}))
    # ARM M: produces nothing either, so only ARM N's registration is under test here.
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: (
        "", {"suite": "", "n_mutants": 0, "n_kept": 0, "kept": [], "rejected": [],
             "composite_status": "empty", "dropped_for_collision": []}))

    rows = rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert "naive" not in frozen
    assert "naive" not in scored["suites"]
    assert "naive" not in scored["admissible"]
    assert rows == []


def test_arm_n_is_registered_when_it_admits_at_least_one_case(tmp_path, monkeypatch):
    """The positive case: a suite with a non-empty admissible set is still registered.

    This fails if the fix over-corrects into omitting ARM N unconditionally -- "naive"
    would then be absent from `frozen`/`scored["suites"]` even though it has a real,
    scorable admissible set.
    """
    frozen, scored = _wire_common(monkeypatch)
    monkeypatch.setattr(rjs.naive, "generate", lambda *a, **k: (
        "test('a', () => { expect(1).toBe(1); });",
        {"suite": "test('a', () => { expect(1).toBe(1); });", "compiles": True,
         "attempts": [], "on_p0": None, "admissible": ["a"]}))
    monkeypatch.setattr(rjs.mutation, "generate", lambda *a, **k: (
        "", {"suite": "", "n_mutants": 0, "n_kept": 0, "kept": [], "rejected": [],
             "composite_status": "empty", "dropped_for_collision": []}))

    rjs.run_task("fake_fixture", tmp_path, n_good=1, n_bad=1, log=lambda *a: None)

    assert frozen["naive"] == "test('a', () => { expect(1).toBe(1); });"
    assert scored["suites"]["naive"] == "test('a', () => { expect(1).toBe(1); });"
    assert scored["admissible"]["naive"] == ["a"]
