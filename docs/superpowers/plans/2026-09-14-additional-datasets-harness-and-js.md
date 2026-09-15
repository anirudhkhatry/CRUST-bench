# additional_datasets: harness + RefactorBench-JS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared good-vs-bad benchmark harness and prove it end to end on the 18 pure-logic RefactorBench-JS fixtures, producing the spec's §19 evaluation table and §5 metrics for four arms.

**Architecture:** A language-agnostic core (`spec.py`, `score.py`, `runner.py`) plus one adapter per benchmark. The core freezes and hashes every generated test suite *before* candidate directories exist, then runs each arm's frozen suite against every candidate. The adapter is the only code that knows about vitest or Maven. This plan builds the core plus the `js_vitest` adapter; the `java_mvn` adapter is a separate plan that reuses everything here unchanged.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), pytest, node v24.21.0 + npm, vitest + jsdom, `rmt/llm.py` for all model calls (disk cache, cost ledger, concurrency cap).

**Spec:** `docs/superpowers/specs/2026-09-14-additional-datasets-design.md`

## Global Constraints

- **Working dir** is `relational_mutation_testing_baseline/`. All paths below are relative to it unless they start with `docs/` or `datasets/`, which are relative to the repo root `/nas/CRUST-bench-COLM/CRUST-bench`.
- **Python is `.venv/bin/python`**, never bare `python3`. The venv has `anthropic`; this plan adds `pytest`.
- **Never fork `rmt/llm.py`.** Import it: `from rmt.llm import ask, fenced, json_block`. It carries the disk cache, the cost ledger and the 16-request concurrency cap.
- **Prompts live in `additional_datasets/prompts/*.md`**, loaded by `<<NAME>>` substitution, mirroring `rmt/prompts/__init__.py`. Never inline a prompt in Python.
- **`seed.py` must not import from, or share prompt text with, `rmt/stage1_mutate.py`** (spec §7.4). Its fault taxonomy is the migration-hazard list in spec §3.4, kept in its own prompt file.
- **Arm S and Arm N results are never pooled** in any reported metric (spec §5).
- **Baseline A (ARM A) is excluded from every GAR comparison** (spec §3.6) — on Benchmark A its GAR is 1.0 by construction.
- **Results root:** `results/additional_datasets/<benchmark>/<task_id>/`, layout exactly as spec §6.
- Commit after every task. Prefix: `feat(ad):`, `test(ad):`, `fix(ad):`.

---

### Task 1: Package skeleton, dataclasses, and the test runner

**Files:**
- Create: `additional_datasets/__init__.py`, `additional_datasets/spec.py`
- Create: `tests/__init__.py`, `tests/test_spec.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: `Task`, `Candidate`, `SuiteResult` dataclasses; `Task.root`, `Candidate.dir`, `write_metadata(task, candidates, extra) -> Path`, `read_metadata(task) -> dict`. Every later task uses these types.

- [ ] **Step 1: Install pytest into the project venv**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench/relational_mutation_testing_baseline
.venv/bin/pip install pytest
.venv/bin/python -m pytest --version
```
Expected: prints a pytest version. There is currently no pytest anywhere on this machine, so this step is required before any test can run.

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_spec.py`:

```python
import json
from pathlib import Path

from additional_datasets.spec import Candidate, SuiteResult, Task, read_metadata, write_metadata


def _task(tmp_path):
    return Task(
        task_id="binary_search",
        benchmark="refactorbench_js",
        language="javascript",
        repository="Create-Inc/refactor-bench",
        source_commit="deadbeef",
        transformation="Decompose app/index.js into smaller modules, preserving behaviour.",
        results_root=tmp_path,
    )


def test_task_root_follows_the_spec_layout(tmp_path):
    t = _task(tmp_path)
    assert t.root == tmp_path / "refactorbench_js" / "binary_search"
    assert t.generated_tests_dir == t.root / "generated_tests"
    assert t.hidden_oracle_dir == t.root / "hidden_oracle"


def test_candidate_dir_is_under_candidates(tmp_path):
    t = _task(tmp_path)
    c = Candidate(candidate_id="bad_01", task_id=t.task_id, label="BAD", bad_arm="S",
                  fault_category="ordering difference", model="claude-opus-5",
                  model_version="claude-opus-5", generation_parameters={"temperature": 1.0},
                  build_status="ok", hidden_oracle_failure="3 cases regressed")
    assert c.dir(t) == t.root / "candidates" / "bad_01"


def test_suite_result_passed_and_failed_case_names():
    r = SuiteResult(status="test_failed", cases={"a": "ok", "b": "FAILED"}, rc=1, seconds=0.5)
    assert r.passed == ["a"]
    assert r.failed == ["b"]


def test_metadata_round_trips_every_spec_9_field(tmp_path):
    t = _task(tmp_path)
    c = Candidate(candidate_id="good_01", task_id=t.task_id, label="GOOD", bad_arm=None,
                  fault_category=None, model="claude-opus-5", model_version="claude-opus-5",
                  generation_parameters={"temperature": 0.0}, build_status="ok",
                  hidden_oracle_failure=None)
    write_metadata(t, [c], extra={"suite_hashes": {"naive": "abc"}})
    blob = read_metadata(t)
    assert blob["task_id"] == "binary_search"
    assert blob["repository"] == "Create-Inc/refactor-bench"
    assert blob["source_commit"] == "deadbeef"
    assert blob["language"] == "javascript"
    assert blob["suite_hashes"] == {"naive": "abc"}
    row = blob["candidates"][0]
    for field in ("candidate_id", "ground_truth_label", "build_status",
                  "hidden_oracle_failure", "fault_category", "model", "model_version",
                  "generation_parameters", "bad_arm"):
        assert field in row, field
    assert row["ground_truth_label"] == "GOOD"
```

- [ ] **Step 4: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets'`

- [ ] **Step 5: Implement `additional_datasets/spec.py`**

Create `additional_datasets/__init__.py` containing only a docstring, then `spec.py`:

```python
"""Task and candidate records, and the metadata.json the spec's §9 requires.

One benchmark instance is a Task (a trusted P0 plus the migration spec M) and a set of
Candidates (compiling post-migration implementations, each labelled GOOD or BAD by a
hidden oracle the test-generating arms never see).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Task:
    task_id: str
    benchmark: str          # "refactorbench_js" | "migrationbench_java"
    language: str           # "javascript" | "java"
    repository: str
    source_commit: str
    transformation: str     # the migration spec M, as prose shown to the arms
    results_root: Path

    @property
    def root(self) -> Path:
        return Path(self.results_root) / self.benchmark / self.task_id

    @property
    def original_dir(self) -> Path:
        return self.root / "original"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def generated_tests_dir(self) -> Path:
        return self.root / "generated_tests"

    @property
    def hidden_oracle_dir(self) -> Path:
        return self.root / "hidden_oracle"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str               # "good_01" | "bad_03"
    task_id: str
    label: str                      # "GOOD" | "BAD"
    bad_arm: str | None             # "S" (seeded) | "N" (harvested) | None for GOOD
    fault_category: str | None      # spec §16 class, BAD only
    model: str | None
    model_version: str | None
    generation_parameters: dict
    build_status: str               # "ok" | "build_error"
    hidden_oracle_failure: str | None   # why the oracle called it BAD; None for GOOD

    def dir(self, task: Task) -> Path:
        return task.candidates_dir / self.candidate_id

    def as_row(self) -> dict:
        return {"candidate_id": self.candidate_id, "ground_truth_label": self.label,
                "bad_arm": self.bad_arm, "fault_category": self.fault_category,
                "model": self.model, "model_version": self.model_version,
                "generation_parameters": self.generation_parameters,
                "build_status": self.build_status,
                "hidden_oracle_failure": self.hidden_oracle_failure}


@dataclass(frozen=True)
class SuiteResult:
    """One test suite's outcome on one implementation."""
    status: str                     # green|test_failed|build_error|timeout|no_tests|error
    cases: dict                     # case name -> "ok" | "FAILED"
    rc: int
    seconds: float
    out: str = ""

    @property
    def passed(self) -> list:
        return sorted(k for k, v in self.cases.items() if v == "ok")

    @property
    def failed(self) -> list:
        return sorted(k for k, v in self.cases.items() if v == "FAILED")

    def as_row(self) -> dict:
        return {"status": self.status, "n_cases": len(self.cases),
                "n_passed": len(self.passed), "n_failed": len(self.failed),
                "rc": self.rc, "seconds": self.seconds}


def write_metadata(task: Task, candidates, extra: dict | None = None) -> Path:
    task.root.mkdir(parents=True, exist_ok=True)
    blob = {"task_id": task.task_id, "benchmark": task.benchmark,
            "language": task.language, "repository": task.repository,
            "source_commit": task.source_commit, "transformation": task.transformation,
            "candidates": [c.as_row() for c in candidates]}
    blob.update(extra or {})
    out = task.root / "metadata.json"
    out.write_text(json.dumps(blob, indent=2))
    return out


def read_metadata(task: Task) -> dict:
    return json.loads((task.root / "metadata.json").read_text())
```

- [ ] **Step 6: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_spec.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 7: Commit**

```bash
git add pytest.ini additional_datasets/__init__.py additional_datasets/spec.py tests/
git commit -m "feat(ad): task/candidate records and the spec-9 metadata schema"
```

---

### Task 2: Verdicts and the core metrics

**Files:**
- Create: `additional_datasets/score.py`
- Create: `tests/test_score.py`

**Interfaces:**
- Consumes: `SuiteResult` from Task 1.
- Produces: constants `ACCEPT`, `REJECT`, `LOAD_FAILED`, `ERROR`, `BEHAVIOURAL`, `STRUCTURAL`; `verdict(result, admissible) -> (str, str)`; `confusion(rows) -> dict`; `metrics(rows) -> dict` with keys `gar`, `brr`, `ap`, `frr`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `n_good`, `n_bad`.

A `row` throughout is a dict carrying at least `label` ("GOOD"/"BAD") and `verdict` (ACCEPT/REJECT); unscorable rows carry `verdict` LOAD_FAILED or ERROR and are excluded from every rate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_score.py`:

```python
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.score'`

- [ ] **Step 3: Implement `additional_datasets/score.py`**

```python
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
    f1 = None if not prec or not rec else 2 * prec * rec / (prec + rec)
    return {**c, "n_good": n_good, "n_bad": n_bad, "n": n,
            "gar": gar, "brr": brr, "ap": _ratio(tn, tn + fn),
            "frr": None if gar is None else 1 - gar,
            "accuracy": _ratio(tp + tn, n), "balanced_accuracy": bal,
            "precision": prec, "recall": rec, "f1": f1}
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add additional_datasets/score.py tests/test_score.py
git commit -m "feat(ad): verdicts and the spec-12 metrics, behavioural vs structural"
```

---

### Task 3: The report — per-fault-class, arms side by side, Arm S and N never pooled

**Files:**
- Modify: `additional_datasets/score.py` (append)
- Modify: `tests/test_score.py` (append)

**Interfaces:**
- Consumes: `metrics`, `confusion` from Task 2.
- Produces: `by_arm(rows) -> dict[str, dict]`, `by_fault_class(rows) -> list[dict]`, `report(rows, title) -> str`. Rows now additionally carry `arm` ("mutation"|"naive"|"existing"|"ceiling"), `bad_arm` ("S"|"N"|None) and `fault_category`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_score.py`:

```python
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_score.py -k "by_arm or fault_class or report" -v`
Expected: FAIL — `ImportError: cannot import name 'by_arm'`

- [ ] **Step 3: Append the implementation to `additional_datasets/score.py`**

```python
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
```

- [ ] **Step 4: Run the whole score suite**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add additional_datasets/score.py tests/test_score.py
git commit -m "feat(ad): per-arm and per-fault-class report, seeded and harvested kept apart"
```

---

### Task 4: The freeze — leakage control enforced by hashing, not convention

**Files:**
- Create: `additional_datasets/runner.py`
- Create: `tests/test_runner_freeze.py`

**Interfaces:**
- Consumes: `Task` from Task 1.
- Produces: `freeze(task, suites) -> dict[str, str]` (arm -> sha256, writes each suite to `generated_tests/<arm>/suite.<ext>` and the hashes into `metadata.json`); `assert_frozen(task, suites) -> None` (raises `LeakageError`); `assert_no_leak(prompt, forbidden) -> None`; exception `LeakageError`.

This is spec §7. The freeze must happen while `candidates/` does not exist, so the ordering is a property of the filesystem rather than a promise in a docstring.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_freeze.py`:

```python
import json

import pytest

from additional_datasets.runner import LeakageError, assert_frozen, assert_no_leak, freeze
from additional_datasets.spec import Task, write_metadata


def _task(tmp_path):
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="Create-Inc/refactor-bench",
                source_commit="deadbeef", transformation="Decompose it.",
                results_root=tmp_path)


def test_freeze_writes_suites_and_records_their_hashes(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    hashes = freeze(t, {"mutation": "test('a', () => {})", "naive": "test('b', () => {})"})
    assert set(hashes) == {"mutation", "naive"}
    assert (t.generated_tests_dir / "mutation" / "suite.js").exists()
    blob = json.loads((t.root / "metadata.json").read_text())
    assert blob["suite_hashes"] == hashes


def test_freeze_refuses_once_candidates_exist(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    (t.candidates_dir / "good_01").mkdir(parents=True)
    with pytest.raises(LeakageError, match="candidates already exist"):
        freeze(t, {"naive": "test('a', () => {})"})


def test_assert_frozen_detects_a_changed_suite(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    freeze(t, {"naive": "test('a', () => {})"})
    assert_frozen(t, {"naive": "test('a', () => {})"})          # unchanged: fine
    with pytest.raises(LeakageError, match="changed after freezing"):
        assert_frozen(t, {"naive": "test('a', () => {}); // tweaked"})


def test_assert_no_leak_catches_candidate_text_in_a_prompt():
    assert_no_leak("Write tests for this library.", ["const SECRET_BAD_CODE = 1;"])
    with pytest.raises(LeakageError, match="leaked"):
        assert_no_leak("Here is the candidate: const SECRET_BAD_CODE = 1;",
                       ["const SECRET_BAD_CODE = 1;"])


def test_assert_no_leak_ignores_trivially_short_fragments():
    # A three-character fragment appears in any English prompt; only meaningful
    # fragments are checked, or the guard would fire constantly and get disabled.
    assert_no_leak("Write a test for the app.", ["app"])
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_runner_freeze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.runner'`

- [ ] **Step 3: Implement `additional_datasets/runner.py`**

```python
"""The experimental protocol, and the leakage controls that make it mean anything.

The spec's §15 is the difference between measuring pre-migration test generation and
accidentally measuring fault localisation. Convention is not enough, so the ordering is
enforced against the filesystem: suites are written and hashed while `candidates/` does
not yet exist, and scoring refuses to proceed if a suite's bytes changed afterwards.
"""
import hashlib
import json

from .spec import Task

EXT = {"javascript": "js", "java": "java"}
MIN_LEAK_FRAGMENT = 24     # shorter fragments match innocent prose


class LeakageError(RuntimeError):
    """A leakage control fired. The run is not scoreable and must not be reported."""


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def freeze(task: Task, suites: dict) -> dict:
    """Write every arm's suite, hash it, record the hashes. Candidates must not exist."""
    if task.candidates_dir.exists() and any(task.candidates_dir.iterdir()):
        raise LeakageError(
            f"{task.task_id}: candidates already exist -- suites must be frozen before "
            f"any candidate is on disk (spec §7)")
    ext = EXT.get(task.language, "txt")
    hashes = {}
    for arm, text in sorted(suites.items()):
        d = task.generated_tests_dir / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / f"suite.{ext}").write_text(text)
        hashes[arm] = _sha(text)
    meta = task.root / "metadata.json"
    blob = json.loads(meta.read_text()) if meta.exists() else {}
    blob["suite_hashes"] = hashes
    meta.write_text(json.dumps(blob, indent=2))
    return hashes


def assert_frozen(task: Task, suites: dict) -> None:
    recorded = json.loads((task.root / "metadata.json").read_text()).get("suite_hashes", {})
    for arm, text in sorted(suites.items()):
        if arm in recorded and recorded[arm] != _sha(text):
            raise LeakageError(f"{task.task_id}/{arm}: suite changed after freezing")


def assert_no_leak(prompt: str, forbidden) -> None:
    """No candidate source, ground-truth label or hidden-oracle text in a prompt."""
    for frag in forbidden:
        frag = (frag or "").strip()
        if len(frag) >= MIN_LEAK_FRAGMENT and frag in prompt:
            raise LeakageError(f"leaked into a test-generation prompt: {frag[:60]!r}...")
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_runner_freeze.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add additional_datasets/runner.py tests/test_runner_freeze.py
git commit -m "feat(ad): freeze suites before candidates exist, and fail loudly on leakage"
```

---

### Task 5: The oracle split — leaf-case skip-masking

**Files:**
- Create: `additional_datasets/adapters/__init__.py`, `additional_datasets/adapters/js_split.py`
- Create: `tests/test_js_split.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `find_leaf_cases(text) -> list[dict]` (each `{"title", "start", "end", "kind"}` where `kind` is `"test"` or `"it"` and `start`/`end` bracket the callee token); `split_suite(text, seed) -> (half_a, half_b, assignment)` where `assignment` is `{title: "A"|"B"}`.

This implements the corrected spec §4.4. Do **not** extract text into new files: mask one half with `.skip` so imports, nested `describe`s, `beforeEach` and shared fixture constants survive in both halves.

- [ ] **Step 1: Write the failing test**

Create `tests/test_js_split.py`:

```python
from additional_datasets.adapters.js_split import find_leaf_cases, split_suite

SUITE = """import { describe, test, expect } from 'vitest';
import { search } from './app/index.js';

describe('Binary Search', () => {
  const sorted = [1, 3, 5];

  describe('search', () => {
    test('finds an element', () => {
      expect(search(sorted, 3)).toBe(1);
    });

    test('misses an absent element', () => {
      expect(search(sorted, 4)).toBe(-1);
    });

    it('handles the empty array', () => {
      expect(search([], 1)).toBe(-1);
    });
  });
});
"""


def test_find_leaf_cases_finds_test_and_it_but_not_describe():
    cases = find_leaf_cases(SUITE)
    assert [c["title"] for c in cases] == [
        "finds an element", "misses an absent element", "handles the empty array"]
    assert [c["kind"] for c in cases] == ["test", "test", "it"]


def test_split_masks_each_half_with_skip_and_keeps_everything_else():
    a, b, assign = split_suite(SUITE, seed="binary_search")
    for half in (a, b):
        # Shared setup survives in both halves -- this is why we mask instead of extract.
        assert "import { search } from './app/index.js';" in half
        assert "const sorted = [1, 3, 5];" in half
        assert "describe('Binary Search'" in half
        assert half.count("test(") + half.count("it(") + half.count(".skip(") >= 3
    assert set(assign.values()) == {"A", "B"}


def test_every_case_lands_in_exactly_one_half():
    a, b, assign = split_suite(SUITE, seed="binary_search")
    for title, side in assign.items():
        live, masked = (a, b) if side == "A" else (b, a)
        assert f"'{title}'" in live and f"'{title}'" in masked
    # Counting skips: each half skips exactly the cases assigned to the other.
    n_b = sum(1 for v in assign.values() if v == "B")
    assert a.count(".skip(") == n_b


def test_the_split_is_deterministic_for_a_given_seed():
    assert split_suite(SUITE, seed="x")[2] == split_suite(SUITE, seed="x")[2]


def test_different_seeds_can_produce_different_assignments():
    seeds = [split_suite(SUITE, seed=s)[2] for s in ("a", "b", "c", "d", "e", "f")]
    assert any(s != seeds[0] for s in seeds)


def test_a_single_case_suite_is_reported_as_unsplittable():
    one = "describe('x', () => { test('only', () => { expect(1).toBe(1); }); });"
    a, b, assign = split_suite(one, seed="x")
    assert assign == {}      # caller marks ARM A n/a rather than scoring a corrupt half
    assert a == b == one
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_js_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.adapters'`

- [ ] **Step 3: Implement the splitter**

Create `additional_datasets/adapters/__init__.py` with only a docstring, then
`additional_datasets/adapters/js_split.py`:

```python
"""Split a vitest suite into two disjoint halves without cutting anything out.

Baseline A and the hidden oracle would otherwise be the same file, which makes Baseline A
a tautology (spec §4.4). Measuring the pilot corpus killed the obvious split: 16 of the 18
pure-logic fixtures have exactly one top-level `describe`, so the only granularity that
exists in quantity is the leaf `test`/`it` case.

So both halves are the *whole* original file, with the other half's leaf calls rewritten
`test(` -> `test.skip(`. Nothing is extracted, so imports, nested describes, `beforeEach`
and shared fixture constants -- `const sorted = [1, 3, 5]` in the outer describe body --
survive in both halves rather than having to be re-derived, which is where a text-cutting
split would quietly go wrong.
"""
import hashlib
import re

# A leaf case opens with `test(` or `it(` followed by a quoted title. `describe(` is
# deliberately absent, and `.skip`/`.only`/`.each` are excluded by the `(` right after
# the token, so an already-masked file is never double-masked.
CASE_RE = re.compile(r"""\b(?P<kind>test|it)\(\s*(?P<q>['"`])(?P<title>(?:\\.|(?!(?P=q)).)*)(?P=q)""")


def find_leaf_cases(text):
    """Every leaf `test`/`it` call, in source order."""
    return [{"title": m.group("title"), "kind": m.group("kind"),
             "start": m.start("kind"), "end": m.start("kind") + len(m.group("kind"))}
            for m in CASE_RE.finditer(text)]


def _side(seed, index):
    """Alternate, with the starting parity seeded so neither half is always the earlier
    tests across the corpus."""
    off = int(hashlib.sha256(seed.encode()).hexdigest(), 16) & 1
    return "A" if (index + off) % 2 == 0 else "B"


def _mask(text, cases, drop):
    """Rewrite the named cases' callee token to `<kind>.skip`, right to left."""
    out = text
    for c in sorted(cases, key=lambda c: -c["start"]):
        if c["title"] in drop:
            out = out[:c["end"]] + ".skip" + out[c["end"]:]
    return out


def split_suite(text, seed):
    """(half_a, half_b, {title: "A"|"B"}). An unsplittable suite returns ({}, text, text)."""
    cases = find_leaf_cases(text)
    if len(cases) < 2:
        return text, text, {}
    assignment = {}
    for i, c in enumerate(cases):
        assignment.setdefault(c["title"], _side(seed, i))
    a_titles = {t for t, s in assignment.items() if s == "A"}
    b_titles = {t for t, s in assignment.items() if s == "B"}
    if not a_titles or not b_titles:          # degenerate after de-duplicating titles
        return text, text, {}
    return _mask(text, cases, b_titles), _mask(text, cases, a_titles), assignment
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_js_split.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add additional_datasets/adapters/ tests/test_js_split.py
git commit -m "feat(ad): split a vitest suite by leaf case via skip-masking"
```

---

### Task 6: The JS runtime — one shared vitest install

**Files:**
- Create: `additional_datasets/adapters/js_runtime/package.json`
- Create: `additional_datasets/adapters/js_runtime/vitest.config.mjs`
- Create: `additional_datasets/adapters/js_vitest.py`
- Create: `tests/test_js_vitest.py`
- Create: `scripts/setup_js_runtime.sh`

**Interfaces:**
- Consumes: `SuiteResult` from Task 1.
- Produces: `FIXTURES` (Path to the corpus), `fixture_config(name) -> dict` with keys `targetFile`, `testFile`; `holdout_text(name) -> str`; `src_prefix(name) -> str` (`""` or `"src"`); `import_path(name) -> str`; `pure_logic_fixtures() -> list[str]`; `stage(name, dest, sources=None) -> Path`; `staged_prefix(workspace) -> str`; `run_suite(workspace, suite_text, suite_name) -> SuiteResult`; `parse_vitest_json(blob) -> (status, cases)`.

The adapter protocol every benchmark must satisfy is exactly `stage(name, dest, sources=None)` and `run_suite(workspace, suite_text, suite_name)`. The per-fixture source prefix is deliberately NOT a parameter of `run_suite` — it is recorded into the staged workspace by `stage` and read back — so that the Java adapter can satisfy the same two signatures.

The benchmark ships no `package.json` — its `HARNESS_INTERFACE.md` says evaluators bring their own package installation — so this task builds that runtime once and reuses it for all 123 fixtures.

- [ ] **Step 1: Fetch the corpus**

The corpus lives at `/nas/CRUST-bench-COLM/CRUST-bench/datasets/refactor-bench`, **outside
this worktree**. A worktree-isolated session is refused any `git` command targeting a path
outside its own worktree, so this is a tarball fetch, not a `git clone` — the corpus needs
no history:

```bash
mkdir -p /nas/CRUST-bench-COLM/CRUST-bench/datasets
cd /nas/CRUST-bench-COLM/CRUST-bench/datasets
if [ ! -d refactor-bench ]; then
  curl -sL https://codeload.github.com/Create-Inc/refactor-bench/tar.gz/refs/heads/main \
       -o /tmp/rbjs.tar.gz
  mkdir -p refactor-bench
  tar xzf /tmp/rbjs.tar.gz -C refactor-bench --strip-components=1
fi
ls refactor-bench/refactoring/data | wc -l
```
Expected: `123`. This may already be present — the controller pre-fetched it; the guard
above makes the step a no-op in that case.

- [ ] **Step 2: Create the shared runtime**

`additional_datasets/adapters/js_runtime/package.json`:

```json
{
  "name": "ad-js-runtime",
  "private": true,
  "type": "module",
  "devDependencies": {
    "vitest": "^2.1.8",
    "jsdom": "^25.0.1",
    "@testing-library/react": "^16.1.0",
    "@testing-library/jest-dom": "^6.6.3",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  }
}
```

`additional_datasets/adapters/js_runtime/vitest.setup.mjs`:

```javascript
// Registers the jest-dom matchers (toBeInTheDocument, toHaveValue, ...) that the
// RefactorBench-JS holdout suites assume are available. Without this, a React fixture
// fails with "Invalid Chai property: toBeInTheDocument" on nearly every case —
// 89 of travel_itinerary_planner's 105 failures were exactly this.
import '@testing-library/jest-dom/vitest';
```

`additional_datasets/adapters/js_runtime/vitest.config.mjs`:

Every line of this config was forced by a measured failure. Do not simplify it.

```javascript
import { defineConfig } from 'vitest/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.AD_WORKSPACE || process.cwd();

// The corpus is NOT uniform about where a holdout test expects its sources, so the adapter
// detects the layout per fixture and passes it in AD_SRC_PREFIX:
//   ''    — sources flat at the workspace root; the test imports `./app/index.js`
//   'src' — sources under src/;                the test imports `./src/app/route.js`
// `@` must follow the same prefix, because `@/app/api/utils/sql` is relative to the SOURCE
// root, not the workspace root. Measured over the 18 pure-logic fixtures: 7 flat,
// 4 src-prefixed, 7 unusable for unrelated reasons. Guessing one layout for all gives
// "Failed to resolve import" on whichever group you guessed against.
const srcRoot = path.join(root, process.env.AD_SRC_PREFIX || '');

export default defineConfig({
  root,
  // The automatic JSX runtime. Fixture components and holdout tests use JSX without
  // importing React, so the classic runtime fails every case with
  // "ReferenceError: React is not defined".
  esbuild: { jsx: 'automatic' },
  resolve: { alias: { '@': srcRoot } },
  test: {
    environment: 'jsdom',
    // `globals: true` is required, not a convenience: several holdout suites call
    // `beforeEach`/`vi` without importing them from 'vitest'. With globals off,
    // dijkstra_shortest_path dies with "beforeEach is not defined" and reports zero cases.
    globals: true,
    include: ['*.test.js', '*.test.jsx'],
    setupFiles: [path.join(here, 'vitest.setup.mjs')],
  },
});
```

`scripts/setup_js_runtime.sh`:

```bash
#!/usr/bin/env bash
# Install the shared vitest runtime once; every fixture run reuses it.
set -euo pipefail
cd "$(dirname "$0")/../additional_datasets/adapters/js_runtime"
npm install --no-audit --no-fund
./node_modules/.bin/vitest --version
```

- [ ] **Step 3: Run the install and confirm vitest works**

```bash
chmod +x scripts/setup_js_runtime.sh && ./scripts/setup_js_runtime.sh
```
Expected: prints a vitest version. If npm cannot reach the registry, stop — nothing downstream is meaningful.

- [ ] **Step 4: Write the failing test**

Create `tests/test_js_vitest.py`:

```python
import pytest

from additional_datasets.adapters import js_vitest as JS

pytestmark = pytest.mark.skipif(not JS.FIXTURES.is_dir(),
                                reason="RefactorBench-JS corpus not cloned")


def test_parse_vitest_json_reads_case_names_and_statuses():
    # Real vitest `fullName` values are the describe/test titles joined by a SPACE, e.g.
    # "Binary Search binarySearchRecursive should find existing element" -- verified
    # against a real run. The parser passes the string through untouched, so the exact
    # separator does not matter to it; these names are shaped like the real ones.
    blob = {"testResults": [{"assertionResults": [
        {"fullName": "Binary Search search finds an element", "status": "passed"},
        {"fullName": "Binary Search search misses it", "status": "failed"},
        {"fullName": "Binary Search search skipped one", "status": "skipped"}]}]}
    status, cases = JS.parse_vitest_json(blob)
    assert status == "test_failed"
    assert cases["Binary Search search finds an element"] == "ok"
    assert cases["Binary Search search misses it"] == "FAILED"
    # A skipped case is masked out by the oracle split and must not be scored either way.
    assert "Binary Search search skipped one" not in cases


def test_parse_vitest_json_green_when_nothing_failed():
    blob = {"testResults": [{"assertionResults": [
        {"fullName": "a", "status": "passed"}]}]}
    assert JS.parse_vitest_json(blob)[0] == "green"


def test_parse_vitest_json_no_tests_when_everything_is_skipped():
    blob = {"testResults": [{"assertionResults": [
        {"fullName": "a", "status": "skipped"}]}]}
    assert JS.parse_vitest_json(blob)[0] == "no_tests"


def test_parse_vitest_json_reports_a_load_failure_as_build_error():
    # A file that will not load reports status "failed" with ZERO assertions. That is a
    # structural rejection, not "this suite has no tests" -- conflating them corrupts the
    # structural-vs-behavioural split the design depends on.
    blob = {"testResults": [{"status": "failed", "assertionResults": [],
                             "message": 'Failed to resolve import "./app/index.js"'}]}
    assert JS.parse_vitest_json(blob)[0] == "build_error"


def test_src_prefix_detects_both_corpus_layouts():
    # Measured: 7 of the 18 pure-logic fixtures are flat, 4 reach sources through src/.
    assert JS.src_prefix("binary_search") == ""
    assert JS.src_prefix("seo_content_data") == "src"


def test_import_path_carries_the_prefix():
    # Handing an arm `targetFile` alone yields './app/page.jsx' for an src-prefixed
    # fixture, which never resolves. The arms are told this string verbatim.
    assert JS.import_path("binary_search") == "./app/index.js"
    assert JS.import_path("seo_content_data").startswith("./src/")


def test_stage_records_the_prefix_for_run_suite(tmp_path):
    ws = JS.stage("seo_content_data", tmp_path / "ws")
    assert JS.staged_prefix(ws) == "src"
    assert (ws / "src").is_dir()
    flat = JS.stage("binary_search", tmp_path / "flat")
    assert JS.staged_prefix(flat) == ""
    assert (flat / "app").is_dir()


def test_pure_logic_fixtures_is_the_18_fixture_pilot_corpus():
    # 18 is the SYNTACTIC count -- fixtures whose holdout needs no DOM. Only 11 of them
    # actually verify end to end (holdout green on P0, split disjoint and complete); the
    # other 7 are corpus defects, and Task 7's verification is what discovers that.
    # Do not hardcode 11 here; this function does not run anything.
    names = JS.pure_logic_fixtures()
    assert len(names) == 18
    assert "binary_search" in names and "merge_sort" in names


def test_fixture_config_names_the_target_and_the_holdout_suite():
    cfg = JS.fixture_config("binary_search")
    assert cfg["targetFile"] == "app/index.js"
    assert cfg["testFile"].endswith(".test.js")


def test_the_holdout_suite_runs_green_against_the_untouched_fixture(tmp_path):
    """The toolchain smoke test from spec §8 step 1. If this fails, stop.

    Verified by hand before this plan was written: binary_search reports exactly 19
    cases, all green. A `build_error` here with "Failed to resolve import './app/index.js'"
    means `stage` nested the sources under `src/` instead of putting their contents at the
    workspace root — re-read `stage`'s docstring.
    """
    ws = JS.stage("binary_search", tmp_path / "ws")
    suite = (JS.FIXTURES / "binary_search" / JS.fixture_config("binary_search")["testFile"]).read_text()
    res = JS.run_suite(ws, suite, "oracle")
    assert res.status == "green", res.out[-2000:]
    assert len(res.cases) == 19
```

- [ ] **Step 5: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_js_vitest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.adapters.js_vitest'`

- [ ] **Step 6: Implement `additional_datasets/adapters/js_vitest.py`**

```python
"""RefactorBench-JS access and execution.

The benchmark ships fixtures and a holdout test suite but no package.json -- its
HARNESS_INTERFACE.md hands package installation and test execution to the evaluator. So
there is one shared vitest install under `js_runtime/`, reused by every fixture, and each
run stages a throwaway copy of the fixture so the corpus is never mutated.
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..spec import SuiteResult

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench/datasets/refactor-bench")
FIXTURES = ROOT / "refactoring/data"
RUNTIME = Path(__file__).resolve().parent / "js_runtime"
VITEST = RUNTIME / "node_modules/.bin/vitest"
CONFIG = RUNTIME / "vitest.config.mjs"
TIMEOUT = 300


def fixture_config(name):
    return json.loads((FIXTURES / name / "refactoring_eval.config.json").read_text())


def holdout_text(name):
    return (FIXTURES / name / fixture_config(name)["testFile"]).read_text()


def pure_logic_fixtures():
    """Fixtures whose holdout suite needs no DOM -- the spec §4.3 pilot corpus."""
    out = []
    for d in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
        tests = sorted(d.glob("*.test.js")) + sorted(d.glob("*.test.jsx"))
        if not tests:
            continue
        text = tests[0].read_text()
        if "@testing-library/react" in text or "react-native" in text:
            continue
        out.append(d.name)
    return out


SRC_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*['"]\./src/""")
PREFIX_MARKER = ".ad_src_prefix"


def src_prefix(name):
    """"src" if this fixture's holdout reaches sources through src/, else "" (flat).

    The corpus is not uniform and there is no metadata field for this, so it is read off
    the holdout's own imports. The pattern deliberately matches DYNAMIC imports too
    (`await import('./src/auth.js')`) -- basic_web_with_auth has no static import of its
    target at all, and a `from`-only pattern silently classifies it as flat and then fails
    to resolve every import.
    """
    return "src" if SRC_IMPORT.search(holdout_text(name)) else ""


def import_path(name):
    """How a suite at the workspace root reaches the target file, prefix included.

    The arms must be told this verbatim. Handing a model `targetFile` alone makes it write
    `./app/page.jsx` for an src-prefixed fixture, which never resolves.
    """
    pre = src_prefix(name)
    rel = fixture_config(name)["targetFile"]
    return f"./{pre}/{rel}" if pre else f"./{rel}"


def stage(name, dest, sources=None):
    """Copy a fixture into a writable workspace. `sources` overrides files by relpath.

    Where `src/` lands depends on the fixture: flat fixtures get its CONTENTS at the
    workspace root (their holdout imports `./app/index.js`, so `app/` must be a sibling of
    the test file); src-prefixed fixtures keep it at `src/`. Override paths are always
    relative to `src/`, matching `targetFile` in `refactoring_eval.config.json`.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    pre = src_prefix(name)
    target = dest / pre if pre else dest
    shutil.copytree(FIXTURES / name / "src", target, dirs_exist_ok=True)
    for rel, text in (sources or {}).items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    # The prefix is a property of the staged workspace, so record it here rather than
    # threading the fixture name through run_suite. That keeps the adapter protocol
    # -- stage(name, dest, sources) / run_suite(workspace, text, suite_name) -- identical
    # for every benchmark, which the Java adapter also has to satisfy.
    (dest / PREFIX_MARKER).write_text(pre)
    return dest


def staged_prefix(workspace):
    """The source prefix recorded by `stage` for this workspace."""
    marker = Path(workspace) / PREFIX_MARKER
    return marker.read_text().strip() if marker.exists() else ""


def parse_vitest_json(blob):
    """(status, {case name: "ok"|"FAILED"}). Skipped cases are not scored either way.

    A file that fails to LOAD reports `status: "failed"` with zero assertionResults. That
    is a build error -- a structural rejection -- not "this suite has no tests", which is
    unscorable. Conflating them corrupts the structural-vs-behavioural split the design
    requires, and it is the most likely misreading of this report format.
    """
    cases = {}
    files = blob.get("testResults") or []
    for f in files:
        for a in f.get("assertionResults") or []:
            name = a.get("fullName") or a.get("title") or ""
            st = a.get("status")
            if st == "passed":
                cases[name] = "ok"
            elif st == "failed":
                cases[name] = "FAILED"
            # A skipped case simply does not enter `cases`: the oracle split masks half
            # the suite with .skip, and those must not be scored either way.
    if not cases:
        file_failed = any(f.get("status") == "failed" for f in files)
        return ("build_error" if file_failed else "no_tests"), cases
    return ("test_failed" if any(v == "FAILED" for v in cases.values()) else "green"), cases


def run_suite(workspace, suite_text, suite_name):
    """Run one suite against a staged workspace. The suite is written in, then removed.

    Callers stage a workspace once and run several suites against it, so both the test
    file and the report are cleaned up on every exit path. Deleting the report BEFORE
    running matters most: the build-error check below is `not out_file.exists()`, so a
    stale report from an earlier run of the same suite_name would be indistinguishable
    from this run's output and a crashed vitest would silently return the PREVIOUS
    result as if it were current.
    """
    workspace = Path(workspace)
    test_file = workspace / f"{suite_name}.test.jsx"
    test_file.write_text(suite_text)
    out_file = workspace / f"{suite_name}.results.json"
    out_file.unlink(missing_ok=True)
    env = {**os.environ, "AD_WORKSPACE": str(workspace), "CI": "1",
           "AD_SRC_PREFIX": staged_prefix(workspace)}
    t0 = time.time()
    try:
        try:
            p = subprocess.run(
                [str(VITEST), "run", "--config", str(CONFIG), "--reporter=json",
                 f"--outputFile={out_file}", test_file.name],
                cwd=workspace, env=env, capture_output=True, text=True, timeout=TIMEOUT)
            rc, out = p.returncode, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            return SuiteResult(status="timeout", cases={}, rc=124,
                               seconds=round(time.time() - t0, 1), out=str(e)[-3000:])
        secs = round(time.time() - t0, 1)
        if not out_file.exists():
            # vitest never produced a report: the suite did not build or load.
            return SuiteResult(status="build_error", cases={}, rc=rc, seconds=secs,
                               out=out[-4000:])
        status, cases = parse_vitest_json(json.loads(out_file.read_text()))
        return SuiteResult(status=status, cases=cases, rc=rc, seconds=secs,
                           out=out[-4000:])
    finally:
        test_file.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)
```

- [ ] **Step 7: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_js_vitest.py -v`
Expected: PASS, 6 tests. The last one is the spec §8 toolchain smoke test — if it fails, fix the runtime before continuing.

- [ ] **Step 8: Commit**

```bash
git add additional_datasets/adapters/js_runtime/package.json \
        additional_datasets/adapters/js_runtime/vitest.config.mjs \
        additional_datasets/adapters/js_vitest.py scripts/setup_js_runtime.sh \
        tests/test_js_vitest.py
git commit -m "feat(ad): shared vitest runtime and RefactorBench-JS fixture execution"
```

Note: do **not** commit `js_runtime/node_modules`. Add it to `.gitignore` in this step if it is not already covered.

---

### Task 7: Verify the oracle split against P0 before it is allowed to score

**Files:**
- Create: `additional_datasets/adapters/js_oracle.py`
- Create: `tests/test_js_oracle.py`

**Interfaces:**
- Consumes: `split_suite` (Task 5), `stage`/`run_suite`/`holdout_text`/`fixture_config` (Task 6).
- Produces: `prepare_oracle(name, workspace) -> dict` with keys `oracle_text`, `existing_text`, `assignment`, `admissible_oracle`, `admissible_existing`, `arm_a_usable` (bool), `reason`.

Spec §4.4 requires three properties checked against the unmodified P0: both halves green, their case-name union equals the full suite's, and their intersection is empty. A fixture failing any of them is excluded from ARM A scoring — marked `n/a`, never zero.

- [ ] **Step 1: Write the failing test**

Create `tests/test_js_oracle.py`:

```python
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_js_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.adapters.js_oracle'`

- [ ] **Step 3: Implement `additional_datasets/adapters/js_oracle.py`**

```python
"""Prepare a fixture's hidden oracle and its Baseline A half, and verify the split.

Baseline A and the hidden oracle are the same file until it is split, which would make
Baseline A a tautology. The split is only trustworthy if it is checked: both halves green
on the untouched P0, their case names union to the whole suite's, and their intersection
empty. A fixture that fails any of the three keeps its oracle and loses only Baseline A.
"""
from . import js_vitest as JS
from .js_split import split_suite


def prepare_oracle(name, workspace):
    whole = JS.holdout_text(name)
    oracle_text, existing_text, assignment = split_suite(whole, seed=name)
    ws = JS.stage(name, workspace)

    full = JS.run_suite(ws, whole, "whole")
    if full.status != "green":
        return {"oracle_text": whole, "existing_text": "", "assignment": {},
                "admissible_oracle": full.passed, "admissible_existing": [],
                "arm_a_usable": False,
                "reason": f"holdout suite is not green on P0: {full.status}"}

    if not assignment:
        return {"oracle_text": whole, "existing_text": "", "assignment": {},
                "admissible_oracle": full.passed, "admissible_existing": [],
                "arm_a_usable": False, "reason": "unsplittable: fewer than two leaf cases"}

    a = JS.run_suite(ws, oracle_text, "oracle")
    b = JS.run_suite(ws, existing_text, "existing")
    got = {"oracle_text": oracle_text, "existing_text": existing_text,
           "assignment": assignment, "admissible_oracle": a.passed,
           "admissible_existing": b.passed, "arm_a_usable": True, "reason": ""}

    overlap = set(a.passed) & set(b.passed)
    union = set(a.passed) | set(b.passed)
    if a.status != "green" or b.status != "green":
        got |= {"arm_a_usable": False,
                "reason": f"a half is not green on P0: {a.status}/{b.status}"}
    elif overlap:
        got |= {"arm_a_usable": False,
                "reason": f"halves overlap on {len(overlap)} cases"}
    elif union != set(full.cases):
        missing = set(full.cases) - union
        got |= {"arm_a_usable": False,
                "reason": f"halves lose {len(missing)} cases from the whole suite"}
    if not got["arm_a_usable"]:
        # Keep the whole suite as the oracle; only Baseline A is forfeited.
        got |= {"oracle_text": whole, "existing_text": "",
                "admissible_oracle": full.passed, "admissible_existing": []}
    return got
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_js_oracle.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Check the split across the whole pilot corpus**

```bash
.venv/bin/python -c "
from pathlib import Path
import tempfile
from additional_datasets.adapters import js_oracle as O, js_vitest as JS
for n in JS.pure_logic_fixtures():
    with tempfile.TemporaryDirectory() as d:
        g = O.prepare_oracle(n, Path(d)/'ws')
        print(f\"{n:28} arm_a={g['arm_a_usable']!s:5} \"
              f\"oracle={len(g['admissible_oracle']):3} \"
              f\"existing={len(g['admissible_existing']):3} {g['reason']}\")
"
```
Expected: a line per fixture. Record how many have `arm_a=True`; this is the real Baseline A coverage and belongs in the pilot write-up. Fixtures with `arm_a=False` are not a bug — they are honestly reported as `n/a`.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/adapters/js_oracle.py tests/test_js_oracle.py
git commit -m "feat(ad): verify the oracle split against P0 before letting it score"
```

---

### Task 8: ARM N (naive) and ARM M (mutation-guided) test generation

**Files:**
- Create: `additional_datasets/prompts/__init__.py`
- Create: `additional_datasets/prompts/naive_testgen.md`, `additional_datasets/prompts/mutate.md`, `additional_datasets/prompts/mutation_testgen.md`, `additional_datasets/prompts/repair.md`
- Create: `additional_datasets/arms/__init__.py`, `additional_datasets/arms/naive.py`, `additional_datasets/arms/mutation.py`
- Create: `tests/test_arms.py`

**Interfaces:**
- Consumes: `SuiteResult`, `assert_no_leak`, `js_vitest.run_suite`/`stage`.
- Produces: `naive.generate(task, sources, adapter, workspace, import_line, framework="vitest", retries=3, tag="naive") -> (suite_text, record)`; `mutation.propose_mutants(sources, n, tag) -> list[dict]` (each `{"file", "original_snippet", "mutated_snippet", "semantic_change", "fault_class"}`); `mutation.apply_mutant(sources, rec) -> dict | None`; `mutation.kills(on_p0, on_mutant) -> bool`; `mutation.generate(task, sources, adapter, workspace, import_line, framework="vitest", n_mutants=6, retries=3, tag="mut") -> (suite_text, record)`.

`import_line` is supplied by the caller (the benchmark module, from the adapter's `import_path`). Both arms receive it identically — neither is advantaged by knowing where the code lives, and without it a model invents a path that does not resolve and every suite build-errors. This was measured: in a pilot, handing the model the target path relative to the fixture rather than to the staged workspace made **every** suite fail to build, twice, through two repair rounds.

The two arms differ in exactly one thing: ARM M's tests must pass an **executed** kill gate (green on P0, red on the mutant) before being kept. Everything else — model, effort, repair budget, output format — is held identical, or the comparison measures the wrong variable.

- [ ] **Step 1: Write the prompts**

`additional_datasets/prompts/__init__.py` — copy the loader pattern from `rmt/prompts/__init__.py`:

```python
"""Prompt texts, kept out of the pipeline code so they can be diffed run to run.

Substitution is plain `<<NAME>>` replacement, not str.format -- the prompts embed source
code, which is full of braces.
"""
from pathlib import Path

_DIR = Path(__file__).parent


def load(name, **subs):
    text = (_DIR / f"{name}.md").read_text()
    for k, v in subs.items():
        text = text.replace(f"<<{k.upper()}>>", str(v))
    return text
```

`additional_datasets/prompts/naive_testgen.md`:

```markdown
You are given a program and a transformation that will be applied to it. Write a test
suite that will be run against the transformed program to check that the transformation
preserved observable behaviour.

## The program

<<SOURCES>>

## The transformation that will be applied

<<TRANSFORMATION>>

## What to write

Write a single <<LANGUAGE>> test file using <<FRAMEWORK>>.

- Import the code under test with EXACTLY this line, character for character:
  `<<IMPORT_LINE>>`
  Do not alter that path. It is where the file actually sits in the test workspace, and a
  path you invent instead will not resolve.
- Assert exact values. A test that only checks a value is non-null, or that two calls
  agree with each other, cannot detect a behavioural change.
- Test only the behaviour visible through the program's public surface, using the names
  the program already exports. The transformation may reorganise internals; a test that
  reaches into internals will fail a correct transformation.
- Cover edge cases: empty inputs, boundaries, error paths, ordering, and any value the
  code treats specially.
- Never write a test that could fail the program exactly as given to you above.

Return only the test file in a single fenced code block.
```

`additional_datasets/prompts/mutate.md`:

```markdown
You are producing mutation-testing fixtures. Propose <<N>> faults to inject into the
program below. Each fault must be a small, plausible mistake a developer could make --
not a random token swap, and not something that breaks the program outright.

## The program

<<SOURCES>>

## Output

Return a single fenced `json` block: a list of <<N>> objects, each with

- `file`: the path exactly as labelled above
- `original_snippet`: a verbatim span copied from that file, unique within it
- `mutated_snippet`: the replacement
- `semantic_change`: one sentence describing the behavioural difference in
  language-agnostic terms, without naming any syntax
- `fault_class`: one of arithmetic, boundary, condition, return_value, state_update,
  control_flow, edge_case, ordering, invariant, error_path

`original_snippet` must appear exactly once in the named file, or the fault is discarded.
```

`additional_datasets/prompts/mutation_testgen.md`:

```markdown
Write a test that detects the following fault in the program below.

## The program

<<SOURCES>>

## The fault

<<SEMANTIC_CHANGE>>

## What to write

Write a single <<LANGUAGE>> test file using <<FRAMEWORK>> that **passes** on the program
exactly as shown above and **fails** on a version of it carrying that fault.

- Import the code under test with EXACTLY this line, character for character:
  `<<IMPORT_LINE>>`
  Do not alter that path. It is where the file actually sits in the test workspace, and a
  path you invent instead will not resolve.
- Assert exact values. The test is executed against both versions and kept only if it
  separates them, so a weak assertion will simply be rejected.
- Reach the fault through the program's public surface, using the names it already
  exports. A later transformation may reorganise internals.
- Do not describe the fault in a comment; just detect it.

Return only the test file in a single fenced code block.
```

`additional_datasets/prompts/repair.md`:

````markdown
Your test file did not build or load. Fix only what prevents it from running.

## The errors

```
<<ERRORS>>
```

## Your previous file

```
<<PREV>>
```

Do not weaken, delete or loosen any assertion. Fix imports, names and syntax only. Return
the corrected file in a single fenced code block.
````

- [ ] **Step 2: Write the failing test**

Create `tests/test_arms.py`:

```python
from additional_datasets.arms import mutation, naive
from additional_datasets.prompts import load
from additional_datasets.spec import SuiteResult


class FakeAdapter:
    """Records what it was asked to run, and replays scripted results."""

    def __init__(self, results):
        self.results, self.calls = list(results), []

    def stage(self, name, dest, sources=None):
        return dest

    def run_suite(self, workspace, suite_text, suite_name):
        self.calls.append((suite_name, suite_text))
        return self.results.pop(0)


def test_naive_prompt_carries_the_program_the_transformation_and_the_import():
    text = load("naive_testgen", sources="SRC_HERE", transformation="MIGRATE_HERE",
                import_line="import * as mod from './src/app/page.jsx';",
                language="JavaScript", framework="vitest")
    assert "SRC_HERE" in text and "MIGRATE_HERE" in text
    assert "./src/app/page.jsx" in text
    assert "<<" not in text


def test_mutation_prompt_carries_the_fault_and_the_import():
    text = load("mutation_testgen", sources="SRC_HERE", semantic_change="FAULT_HERE",
                import_line="import * as mod from './app/index.js';",
                language="JavaScript", framework="vitest")
    assert "SRC_HERE" in text and "FAULT_HERE" in text
    assert "./app/index.js" in text
    assert "<<" not in text


def test_mutation_kill_gate_keeps_a_test_that_separates_p0_from_the_mutant():
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    assert mutation.kills(green, red) is True


def test_mutation_kill_gate_rejects_a_test_that_fails_p0():
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    assert mutation.kills(red, red) is False


def test_mutation_kill_gate_rejects_a_test_that_passes_both():
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    assert mutation.kills(green, green) is False


def test_apply_mutant_requires_a_unique_snippet():
    sources = {"app/index.js": "const a = 1;\nconst b = 2;\n"}
    out = mutation.apply_mutant(sources, {"file": "app/index.js",
                                          "original_snippet": "const a = 1;",
                                          "mutated_snippet": "const a = 2;"})
    assert out["app/index.js"].startswith("const a = 2;")

    assert mutation.apply_mutant(sources, {"file": "app/index.js",
                                           "original_snippet": "const",
                                           "mutated_snippet": "let"}) is None
    assert mutation.apply_mutant(sources, {"file": "nope.js",
                                           "original_snippet": "x",
                                           "mutated_snippet": "y"}) is None
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_arms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.arms'`

- [ ] **Step 4: Implement the two arms**

`additional_datasets/arms/__init__.py` — docstring only.

`additional_datasets/arms/naive.py`:

```python
"""ARM N -- the naive baseline: tests written from P0 and the migration spec, no mutant.

This is the arm the pipeline's claim is measured against, so everything it is denied must
be denied for a reason. It gets the same model, the same effort, the same repair budget
and the same output format as ARM M. The single difference is that no mutant exists to
force its assertions to prove they discriminate.
"""
from rmt.llm import ask, fenced

from ..prompts import load

SYSTEM = ("You are a precise programmer writing tests. You assert exact values, you use "
          "only APIs you have been shown, and you never write a test that could fail a "
          "correct implementation.")


def generate(task, sources, adapter, workspace, import_line, framework="vitest",
             retries=3, tag="naive"):
    """(suite_text, record). Repair is build-errors-only; assertions may not weaken.

    `import_line` is the exact import statement the suite must use to reach the code under
    test. It is passed in rather than derived, because only the adapter knows where the
    file lands in a staged workspace -- handing a model the target's path relative to the
    fixture makes it write an import that does not resolve, and every suite build-errors.
    """
    base = load("naive_testgen", sources=_bundle(sources),
                transformation=task.transformation, import_line=import_line,
                language=task.language.capitalize(), framework=framework)
    return _generate_with_repair(base, adapter, workspace, retries, tag)


def _bundle(sources, max_chars=180_000):
    parts, used = [], 0
    for rel, text in sorted(sources.items()):
        if used + len(text) > max_chars:
            parts.append(f"### `{rel}` (omitted, prompt budget)\n")
            continue
        used += len(text)
        parts.append(f"### `{rel}`\n\n```\n{text}\n```\n")
    return "\n".join(parts)


def _generate_with_repair(base, adapter, workspace, retries, tag):
    attempts, user, suite = [], base, None
    for i in range(retries + 1):
        reply = ask(SYSTEM, user, effort="high",
                    max_tokens=32000 if i == 0 else 64000, tag=f"{tag}_{i}")
        suite = fenced(reply)
        if not suite:
            attempts.append({"attempt": i, "outcome": "no_code"})
            user = base + ("\n\nYour reply ran out of room before the code block. Return "
                           "only the test file, and if the suite would be long write the "
                           "12 highest-value tests rather than 25.")
            continue
        res = adapter.run_suite(workspace, suite, f"{tag}_{i}")
        attempts.append({"attempt": i, "outcome": res.status,
                         "n_passed": len(res.passed), "n_failed": len(res.failed)})
        if res.status != "build_error":
            return suite, {"suite": suite, "compiles": True, "attempts": attempts,
                           "on_p0": res.as_row(), "admissible": res.passed}
        user = base + "\n\n---\n\n" + load("repair", errors=res.out[-4000:], prev=suite)
    return suite, {"suite": suite, "compiles": False, "attempts": attempts,
                   "on_p0": None, "admissible": []}
```

`additional_datasets/arms/mutation.py`:

```python
"""ARM M -- mutation-guided: a test kept only because it provably separates P0 from P0'.

The kill gate is executed, not asserted: the candidate test runs against the untouched
program and against the mutated one, and survives only if it is green on the first and red
on the second. That gate is also the value grounding -- an asserted literal survives only
if the real program produces it, so no expected value is ever a model's guess.
"""
import json

from rmt.llm import ask, fenced, json_block

from ..prompts import load
from .naive import SYSTEM, _bundle, _generate_with_repair

MUTATE_SYSTEM = ("You are a precise programmer producing mutation-testing fixtures. You "
                 "follow output formats exactly and never invent code you have not read.")


def propose_mutants(sources, n, tag, retries=2):
    """Ask for n faults. Returns raw records; validity is decided by `apply_mutant`."""
    user = load("mutate", sources=_bundle(sources), n=n)
    for i in range(retries + 1):
        reply = ask(MUTATE_SYSTEM, user, effort="high", max_tokens=48000, tag=f"{tag}_m{i}")
        try:
            recs = json_block(reply)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(recs, list) and recs:
            return recs
    return []


def apply_mutant(sources, rec):
    """New {path: text} with the fault applied, or None if the snippet is not unique."""
    rel = rec.get("file", "")
    if rel not in sources:
        return None
    old, new = rec.get("original_snippet", ""), rec.get("mutated_snippet", "")
    if not old or sources[rel].count(old) != 1 or old == new:
        return None
    return {**sources, rel: sources[rel].replace(old, new)}


def kills(on_p0, on_mutant):
    """The gate: green on the trusted program, red on the mutated one."""
    if on_p0.status != "green":
        return False
    return on_mutant.status in ("test_failed", "timeout") or (
        on_mutant.status == "build_error")


def generate(task, sources, adapter, workspace, import_line, framework="vitest",
             n_mutants=6, retries=3, tag="mut"):
    """(suite_text, record). Concatenates every test that passed its kill gate.

    `import_line` is the exact import statement the suite must use -- see naive.generate.
    Both arms are given it identically, so neither is advantaged by knowing where the code
    lives.
    """
    mutants = propose_mutants(sources, n_mutants, tag)
    kept, rejected = [], []
    for k, rec in enumerate(mutants):
        mutated = apply_mutant(sources, rec)
        if mutated is None:
            rejected.append({"index": k, "why": "snippet not unique or file unknown"})
            continue
        base = load("mutation_testgen", sources=_bundle(sources),
                    semantic_change=rec.get("semantic_change", ""),
                    import_line=import_line,
                    language=task.language.capitalize(), framework=framework)
        p0_ws = adapter.stage(task.task_id, workspace / f"p0_{k}")
        suite, srec = _generate_with_repair(base, adapter, p0_ws, retries, f"{tag}_{k}")
        if not srec["compiles"]:
            rejected.append({"index": k, "why": "test never built"})
            continue
        mut_ws = adapter.stage(task.task_id, workspace / f"mut_{k}", sources=mutated)
        on_mutant = adapter.run_suite(mut_ws, suite, f"{tag}_{k}_mut")
        on_p0 = adapter.run_suite(p0_ws, suite, f"{tag}_{k}_p0")
        if kills(on_p0, on_mutant):
            kept.append({"index": k, "suite": suite, "fault_class": rec.get("fault_class"),
                         "semantic_change": rec.get("semantic_change")})
        else:
            rejected.append({"index": k, "why": "test did not kill the mutant",
                             "on_p0": on_p0.status, "on_mutant": on_mutant.status})
    suite_text = "\n\n".join(k["suite"] for k in kept)
    return suite_text, {"suite": suite_text, "n_mutants": len(mutants),
                        "n_kept": len(kept), "kept": [
                            {x: k[x] for x in ("index", "fault_class", "semantic_change")}
                            for k in kept],
                        "rejected": rejected}
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_arms.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/prompts/ additional_datasets/arms/ tests/test_arms.py
git commit -m "feat(ad): naive and mutation-guided test generation, identical but for the kill gate"
```

---

### Task 9: Arm S — the seeded fault injector

**Files:**
- Create: `additional_datasets/prompts/seed_fault.md`
- Create: `additional_datasets/seed.py`
- Create: `tests/test_seed.py`

**Interfaces:**
- Consumes: `apply_mutant` (Task 8), `Candidate` (Task 1).
- Produces: `FAULT_CATEGORIES` (the spec §16 list), `HAZARDS` (per-language hazard prose), `seed_faults(task, good_sources, n, tag) -> list[dict]` each `{"sources", "fault_category", "description"}`.

**Spec §7.4 is binding here:** this module must not import from `rmt/stage1_mutate.py` and its prompt must not reuse `rmt/prompts/mutate.md`. Its taxonomy is migration-hazard-specific.

- [ ] **Step 1: Write the prompt**

`additional_datasets/prompts/seed_fault.md`:

```markdown
You are constructing a benchmark of *incorrect but plausible* migrations. Below is a
migration that is known to be correct. Introduce exactly one behavioural fault into it.

## The correct migration

<<SOURCES>>

## The transformation it was supposed to perform

<<TRANSFORMATION>>

## Hazards specific to this kind of migration

<<HAZARDS>>

## Requirements

The result must:

- still build and load without error;
- still look like a competent attempt at the migration -- a reviewer skimming the diff
  should not immediately see a bug;
- differ in *observable behaviour* on at least one realistic input.

Do not introduce a syntax error, delete a function, or make the program obviously broken.
The fault must be the kind of mistake a careful engineer makes under time pressure.

## Output

Return a single fenced `json` block with

- `file`: the path exactly as labelled above
- `original_snippet`: a verbatim span from that file, unique within it
- `mutated_snippet`: the replacement
- `fault_category`: one of <<CATEGORIES>>
- `description`: one sentence on the behavioural difference, for the benchmark's records
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_seed.py`:

```python
from pathlib import Path

from additional_datasets import seed


def test_fault_categories_are_the_spec_16_list():
    assert "encoding difference" in seed.FAULT_CATEGORIES
    assert "ordering difference" in seed.FAULT_CATEGORIES
    assert "partial migration" in seed.FAULT_CATEGORIES
    assert len(seed.FAULT_CATEGORIES) >= 15


def test_every_language_has_hazard_prose():
    assert "javascript" in seed.HAZARDS and "java" in seed.HAZARDS
    assert "Locale" in seed.HAZARDS["java"]


def test_seed_does_not_import_the_mutation_arms_operators():
    """Spec §7.4: sharing a taxonomy with stage1_mutate would make Arm S circular."""
    src = Path("additional_datasets/seed.py").read_text()
    assert "stage1_mutate" not in src
    assert "from rmt import" not in src or "stage1" not in src
    prompt = Path("additional_datasets/prompts/seed_fault.md").read_text()
    rmt_prompt = Path("rmt/prompts/mutate.md").read_text()
    # No long verbatim span shared between the two prompts.
    for i in range(0, len(rmt_prompt) - 120, 60):
        assert rmt_prompt[i:i + 120] not in prompt
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_seed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.seed'`

- [ ] **Step 4: Implement `additional_datasets/seed.py`**

```python
"""Arm S -- seeded BAD migrations, from a taxonomy of migration hazards.

This module is deliberately kept apart from `rmt/stage1_mutate.py`: separate prompt,
separate taxonomy, separate model role. If the fault injector drew from the same list the
mutation arm proposes from, then ARM M catching Arm S faults would prove only that both
halves of this repository share a vocabulary, which is the circularity the 2026-09-08
write-up already raised against the C<->Rust result. Arm N exists to cross-check this arm,
and the two are never pooled in a reported metric.
"""
import json

from rmt.llm import ask, json_block

from .arms.mutation import apply_mutant
from .arms.naive import _bundle
from .prompts import load

SYSTEM = ("You construct benchmark fixtures: plausible, compiling, behaviourally "
          "incorrect migrations. You follow output formats exactly.")

# Spec §16.
FAULT_CATEGORIES = [
    "incorrect condition", "incorrect API mapping", "wrong API overload",
    "wrong default value", "incorrect time/unit conversion", "null-handling change",
    "exception-semantics change", "ordering difference", "encoding difference",
    "numeric precision change", "partial migration", "resource-management error",
    "state-management error", "incorrect serialization", "concurrency difference",
    "configuration-induced behaviour change", "data-conversion error",
]

HAZARDS = {
    "java": """- charset defaults: `Files.readAllLines`, `String.getBytes`, `new String(byte[])`
- `Locale` sensitivity in `String.format`, `toUpperCase`, `toLowerCase`
- `SimpleDateFormat` vs `DateTimeFormatter`: leniency, era, zone and week-year semantics
- iteration order of `HashMap`, `Collectors.toMap`, `Set` implementations
- `Integer` cache identity: `==` versus `equals` above 127
- `Math.round` on negative values; `Double.toString` precision
- removed or replaced APIs whose substitute differs on null or on the exception it throws
- partially migrated call sites: one of several related calls left on the old API""",
    "javascript": """- iteration and insertion order of objects, `Map`, and `Set`
- `Array.prototype.sort` default lexicographic comparison versus a numeric comparator
- strict versus loose equality, and `NaN`/`-0`/`undefined` handling
- shallow versus deep copying when state is lifted into a new module
- default parameter values and argument order silently changed during extraction
- async ordering: a promise no longer awaited, or awaited in a different sequence
- error paths: a thrown error becoming a returned `null`, or the message changing
- off-by-one boundaries at slice, splice and index arithmetic""",
}


def seed_faults(task, good_sources, n, tag, retries=2):
    """n BAD variants of a known-good migration. Each carries its §16 fault category."""
    user = load("seed_fault", sources=_bundle(good_sources),
                transformation=task.transformation,
                hazards=HAZARDS.get(task.language, ""),
                categories=", ".join(FAULT_CATEGORIES))
    out, seen = [], set()
    for i in range(n + retries):
        if len(out) >= n:
            break
        reply = ask(SYSTEM, user, effort="high", max_tokens=16000, tag=f"{tag}_s{i}")
        try:
            rec = json_block(reply)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(rec, list):
            rec = rec[0] if rec else None
        if not rec:
            continue
        key = (rec.get("file"), rec.get("original_snippet"))
        if key in seen:
            continue
        mutated = apply_mutant(good_sources, rec)
        if mutated is None:
            continue
        seen.add(key)
        out.append({"sources": mutated,
                    "fault_category": rec.get("fault_category"),
                    "description": rec.get("description", "")})
    return out
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_seed.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/seed.py additional_datasets/prompts/seed_fault.md tests/test_seed.py
git commit -m "feat(ad): arm S seeded faults, taxonomy kept separate from the mutation arm"
```

---

### Task 10: Arm N candidates and hidden-oracle labelling

**Files:**
- Create: `additional_datasets/prompts/refactor_js.md`
- Create: `additional_datasets/harvest.py`
- Create: `tests/test_harvest.py`

**Interfaces:**
- Consumes: `Candidate` (Task 1), `js_vitest` adapter (Task 6).
- Produces: `propose_candidate(task, sources, target, model_tag, temperature, tag) -> dict | None` with key `sources`; `label(adapter, task, sources, oracle_text, workspace) -> (str, str | None)` returning `("GOOD"|"BAD"|"EXCLUDE", hidden_oracle_failure)`.

`EXCLUDE` is for candidates that do not build. Spec §20: a compilation failure is not an acceptable BAD.

- [ ] **Step 1: Write the prompt**

`additional_datasets/prompts/refactor_js.md`:

```markdown
Refactor the target file below into smaller, well-structured modules while preserving all
externally observable behaviour.

## Target file

`<<TARGET>>`

## The source tree

<<SOURCES>>

## Requirements

- Preserve every export the target file currently provides, with the same names and the
  same behaviour. Callers must not have to change.
- Split the target into focused modules; import them back into the target file.
- Do not change behaviour, including edge cases, error messages, ordering and defaults.

## Output

Return one fenced `json` block mapping each file path (relative to `src/`) to its complete
new contents. Include every file you created or modified, and no others.
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_harvest.py`:

```python
from additional_datasets import harvest
from additional_datasets.spec import SuiteResult, Task


def _task(tmp_path):
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="Create-Inc/refactor-bench",
                source_commit="deadbeef", transformation="Decompose it.",
                results_root=tmp_path)


class FakeAdapter:
    def __init__(self, result):
        self.result = result

    def stage(self, name, dest, sources=None):
        return dest

    def run_suite(self, workspace, suite_text, suite_name):
        return self.result


def test_a_candidate_passing_the_hidden_oracle_is_good(tmp_path):
    a = FakeAdapter(SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1))
    assert harvest.label(a, _task(tmp_path), {}, "suite", tmp_path) == ("GOOD", None)


def test_a_candidate_failing_the_hidden_oracle_is_bad_and_records_why(tmp_path):
    a = FakeAdapter(SuiteResult(status="test_failed", cases={"a": "ok", "b": "FAILED"},
                                rc=1, seconds=0.1))
    label, why = harvest.label(a, _task(tmp_path), {}, "suite", tmp_path)
    assert label == "BAD"
    assert "b" in why


def test_a_candidate_that_does_not_build_is_excluded_not_bad(tmp_path):
    """Spec §20: a compilation failure is not an acceptable BAD implementation."""
    a = FakeAdapter(SuiteResult(status="build_error", cases={}, rc=1, seconds=0.1))
    assert harvest.label(a, _task(tmp_path), {}, "suite", tmp_path)[0] == "EXCLUDE"


def test_a_seeded_fault_the_oracle_cannot_see_is_excluded_as_equivalent(tmp_path):
    """Spec §3.5: an equivalent mutant is unkillable and would depress every arm."""
    a = FakeAdapter(SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1))
    assert harvest.confirm_bad(a, _task(tmp_path), {}, "suite", tmp_path) is False
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_harvest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.harvest'`

- [ ] **Step 4: Implement `additional_datasets/harvest.py`**

```python
"""Arm N -- real transformation attempts, labelled by the hidden oracle.

Arm S manufactures faults; this arm collects the ones a model actually makes. It is the
defence against the criticism that a mutation-guided test suite catches seeded mutants
because both are mutants, so its candidates are generated from P0 and the migration spec
alone, with no sight of any test, and labelled only by the held-out oracle.
"""
import json

from rmt.llm import ask, json_block

from .arms.naive import _bundle
from .prompts import load

SYSTEM = "You are a careful software engineer performing a behaviour-preserving refactoring."


def propose_candidate(task, sources, target, model_tag, temperature, tag):
    """One candidate transformation, or None if the reply was unusable."""
    user = load("refactor_js", target=target, sources=_bundle(sources))
    reply = ask(SYSTEM, user, effort="high", max_tokens=32000,
                temperature=temperature, tag=f"{tag}_{model_tag}")
    try:
        files = json_block(reply)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(files, dict) or not files:
        return None
    return {"sources": {**sources, **files}}


def label(adapter, task, sources, oracle_text, workspace):
    """(GOOD|BAD|EXCLUDE, why). EXCLUDE means it never built -- spec §20."""
    ws = adapter.stage(task.task_id, workspace, sources=sources)
    res = adapter.run_suite(ws, oracle_text, "oracle")
    if res.status in ("build_error", "no_tests", "error"):
        return "EXCLUDE", f"hidden oracle could not run: {res.status}"
    if res.status == "green":
        return "GOOD", None
    if res.status == "timeout":
        return "BAD", "hidden oracle timed out"
    return "BAD", f"hidden oracle failed: {', '.join(res.failed[:6])}"


def confirm_bad(adapter, task, sources, oracle_text, workspace):
    """Does the hidden oracle actually see a difference? Guards against equivalent faults."""
    return label(adapter, task, sources, oracle_text, workspace)[0] == "BAD"
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_harvest.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/harvest.py additional_datasets/prompts/refactor_js.md tests/test_harvest.py
git commit -m "feat(ad): arm N candidates, hidden-oracle labelling, equivalent-fault guard"
```

---

### Task 11: Scoring a task — the spec §19 rows

**Files:**
- Modify: `additional_datasets/runner.py` (append)
- Create: `tests/test_runner_score.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `score_task(task, candidates, suites, admissible, adapter, workspace, existing_text=None, existing_admissible=()) -> list[dict]`. Each row carries the spec §19 columns plus `arm`, `bad_arm`, `label`, `verdict`, `rejection_kind`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_score.py`:

```python
from additional_datasets.runner import score_task
from additional_datasets.score import ACCEPT, REJECT
from additional_datasets.spec import Candidate, SuiteResult, Task, write_metadata


def _task(tmp_path):
    return Task(task_id="binary_search", benchmark="refactorbench_js",
                language="javascript", repository="r", source_commit="c",
                transformation="t", results_root=tmp_path)


class ScriptedAdapter:
    """Returns a result keyed by (candidate_id, suite_name)."""

    def __init__(self, table):
        self.table = table

    def stage(self, name, dest, sources=None):
        return dest

    def run_suite(self, workspace, suite_text, suite_name):
        return self.table[suite_name]


GREEN = SuiteResult(status="green", cases={"t1": "ok"}, rc=0, seconds=0.1)
RED = SuiteResult(status="test_failed", cases={"t1": "FAILED"}, rc=1, seconds=0.1)


def test_score_task_emits_a_row_per_arm_per_candidate(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    good = Candidate("good_01", t.task_id, "GOOD", None, None, "m", "m", {}, "ok", None)
    bad = Candidate("bad_01", t.task_id, "BAD", "S", "ordering difference", "m", "m", {},
                    "ok", "oracle failed")
    cands = {"good_01": (good, {}), "bad_01": (bad, {})}
    adapter = ScriptedAdapter({"mutation@good_01": GREEN, "mutation@bad_01": RED,
                               "naive@good_01": GREEN, "naive@bad_01": GREEN})
    rows = score_task(t, cands, suites={"mutation": "M", "naive": "N"},
                      admissible={"mutation": ["t1"], "naive": ["t1"]},
                      adapter=adapter, workspace=tmp_path / "ws")
    assert len(rows) == 4
    by = {(r["arm"], r["candidate_id"]): r for r in rows}
    assert by[("mutation", "bad_01")]["verdict"] == REJECT
    assert by[("naive", "bad_01")]["verdict"] == ACCEPT
    assert by[("mutation", "good_01")]["verdict"] == ACCEPT


def test_rows_carry_every_spec_19_column(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    bad = Candidate("bad_01", t.task_id, "BAD", "N", "ordering difference", "m", "m", {},
                    "ok", "oracle failed")
    rows = score_task(t, {"bad_01": (bad, {})}, suites={"mutation": "M"},
                      admissible={"mutation": ["t1"]},
                      adapter=ScriptedAdapter({"mutation@bad_01": RED}),
                      workspace=tmp_path / "ws")
    r = rows[0]
    for col in ("task_id", "candidate_id", "ground_truth", "compiles",
                "existing_tests_pass", "generated_tests_pass", "number_generated_tests",
                "tests_failed", "fault_category", "execution_time", "arm", "bad_arm"):
        assert col in r, col
    assert r["ground_truth"] == "BAD"
    assert r["generated_tests_pass"] is False
    assert r["bad_arm"] == "N"


def test_existing_tests_are_scored_as_their_own_arm(tmp_path):
    t = _task(tmp_path)
    write_metadata(t, [])
    bad = Candidate("bad_01", t.task_id, "BAD", "S", "ordering difference", "m", "m", {},
                    "ok", "oracle failed")
    rows = score_task(t, {"bad_01": (bad, {})}, suites={"mutation": "M"},
                      admissible={"mutation": ["t1"]},
                      adapter=ScriptedAdapter({"mutation@bad_01": RED,
                                               "existing@bad_01": GREEN}),
                      workspace=tmp_path / "ws",
                      existing_text="E", existing_admissible=["t1"])
    arms = {r["arm"] for r in rows}
    assert arms == {"mutation", "existing"}
    existing = [r for r in rows if r["arm"] == "existing"][0]
    assert existing["verdict"] == ACCEPT
    assert existing["existing_tests_pass"] is True
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_runner_score.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_task'`

- [ ] **Step 3: Append to `additional_datasets/runner.py`**

```python
# --- scoring ---------------------------------------------------------------------
from .score import ACCEPT, REJECT, verdict   # noqa: E402  (kept next to its use)


def _row(task, cand, arm, res, v, kind, n_tests):
    return {"task_id": task.task_id, "benchmark": task.benchmark,
            "candidate_id": cand.candidate_id, "ground_truth": cand.label,
            "label": cand.label, "bad_arm": cand.bad_arm,
            "fault_category": cand.fault_category, "arm": arm,
            "compiles": cand.build_status == "ok",
            "existing_tests_pass": (v == ACCEPT) if arm == "existing" else None,
            "generated_tests_pass": v == ACCEPT,
            "number_generated_tests": n_tests, "tests_failed": len(res.failed),
            "verdict": v, "rejection_kind": kind, "status": res.status,
            "execution_time": res.seconds,
            "model": cand.model, "model_version": cand.model_version,
            "hidden_oracle_failure": cand.hidden_oracle_failure}


def score_task(task, candidates, suites, admissible, adapter, workspace,
               existing_text=None, existing_admissible=()):
    """Run every frozen suite against every candidate. Returns the spec §19 rows.

    `candidates` is {candidate_id: (Candidate, sources)}. The suites are the frozen ones;
    `assert_frozen` is called first so a suite edited after the freeze cannot be scored.
    """
    assert_frozen(task, suites)
    arms = dict(suites)
    if existing_text:
        arms["existing"] = existing_text
    adm = {**{a: list(v) for a, v in admissible.items()},
           **({"existing": list(existing_admissible)} if existing_text else {})}

    rows = []
    for cid, (cand, sources) in sorted(candidates.items()):
        for arm, text in sorted(arms.items()):
            ws = adapter.stage(task.task_id, workspace / f"{arm}_{cid}", sources=sources)
            res = adapter.run_suite(ws, text, f"{arm}@{cid}")
            v, kind = verdict(res, set(adm.get(arm, [])))
            rows.append(_row(task, cand, arm, res, v, kind, len(adm.get(arm, []))))
    return rows
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_runner_score.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/runner.py tests/test_runner_score.py
git commit -m "feat(ad): score every frozen arm against every candidate, spec-19 rows"
```

---

### Task 12: The entrypoint and the pilot run

**Files:**
- Create: `additional_datasets/benchmarks/__init__.py`, `additional_datasets/benchmarks/refactorbench_js.py`
- Create: `run_additional_datasets.py`
- Create: `tests/test_entrypoint.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `refactorbench_js.build_task(name, results_root) -> Task`; `refactorbench_js.run_task(name, results_root, n_good, n_bad, workspace) -> list[dict]`; CLI `select`, `run`, `report`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_entrypoint.py`:

```python
import subprocess
import sys


def test_cli_lists_the_pilot_corpus():
    p = subprocess.run([sys.executable, "run_additional_datasets.py", "select",
                        "--benchmark", "refactorbench_js"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "binary_search" in p.stdout
    assert "18" in p.stdout


def test_cli_rejects_an_unknown_benchmark():
    p = subprocess.run([sys.executable, "run_additional_datasets.py", "select",
                        "--benchmark", "nope"], capture_output=True, text=True)
    assert p.returncode != 0
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_entrypoint.py -v`
Expected: FAIL — `can't open file 'run_additional_datasets.py'`

- [ ] **Step 3: Implement the benchmark module**

`additional_datasets/benchmarks/__init__.py` — docstring only.

`additional_datasets/benchmarks/refactorbench_js.py`:

```python
"""RefactorBench-JS wired into the shared harness.

The order below is the experiment: prepare and verify the oracle split, generate every
arm's suite from P0 alone, freeze them, and only then build candidates. Nothing may
reorder these steps -- `freeze` refuses once candidates exist.
"""
import json

from ..adapters import js_oracle, js_vitest as JS
from ..arms import mutation, naive
from ..harvest import label, propose_candidate
from ..runner import freeze, score_task
from ..seed import seed_faults
from ..spec import Candidate, Task, write_metadata

MODELS = [("claude-opus-5", 1.0), ("claude-haiku-4-5-20251001", 1.0)]
TRANSFORMATION = ("Decompose the target file into smaller, well-structured modules while "
                  "preserving all externally observable behaviour.")


def build_task(name, results_root):
    return Task(task_id=name, benchmark="refactorbench_js", language="javascript",
                repository="Create-Inc/refactor-bench", source_commit="main",
                transformation=TRANSFORMATION, results_root=results_root)


def _sources(name):
    src = JS.FIXTURES / name / "src"
    return {str(p.relative_to(src)): p.read_text()
            for p in sorted(src.rglob("*")) if p.is_file()}


def run_task(name, results_root, n_good=3, n_bad=3, workspace=None, log=print):
    task = build_task(name, results_root)
    task.root.mkdir(parents=True, exist_ok=True)
    ws = workspace or (task.root / "_work")
    ws.mkdir(parents=True, exist_ok=True)
    sources = _sources(name)
    target = JS.fixture_config(name)["targetFile"]

    # 1. Oracle split, verified against P0 (spec §4.4).
    oracle = js_oracle.prepare_oracle(name, ws / "oracle_ws")
    task.hidden_oracle_dir.mkdir(parents=True, exist_ok=True)
    (task.hidden_oracle_dir / "oracle.test.jsx").write_text(oracle["oracle_text"])
    log(f"{name}: arm_a_usable={oracle['arm_a_usable']} {oracle['reason']}")

    # `prepare_oracle` always PRESERVES the oracle, but preserved is not the same as
    # populated: a fixture whose holdout does not build reports zero admissible cases.
    # Such a fixture can label nothing, so every candidate would come back GOOD by
    # default and the task would be silently meaningless -- while still paying for a full
    # set of LLM calls. Bail before spending any of them. Measured: 4 of the 18 pure-logic
    # fixtures fail this way (their holdouts import modules the fixture does not ship).
    if not oracle["admissible_oracle"]:
        log(f"{name}: SKIPPED -- oracle has no admissible case ({oracle['reason']})")
        write_metadata(task, [], extra={"skipped": "oracle has no admissible case",
                                        "oracle_reason": oracle["reason"]})
        return []

    # 2. Every arm's suite, from P0 and the transformation only.
    p0_ws = JS.stage(name, ws / "p0")
    suites, admissible, records = {}, {}, {}
    imp = f"import * as mod from '{JS.import_path(name)}';"
    n_suite, rec = naive.generate(task, sources, JS, p0_ws, imp, tag=f"{name}_naive")
    suites["naive"], admissible["naive"], records["naive"] = n_suite, rec["admissible"], rec
    m_suite, rec = mutation.generate(task, sources, JS, ws / "mut", imp,
                                     tag=f"{name}_mut")
    m_run = JS.run_suite(p0_ws, m_suite, "mut_p0") if m_suite else None
    suites["mutation"] = m_suite
    admissible["mutation"] = m_run.passed if m_run else []
    records["mutation"] = rec

    # 3. Freeze. Candidates must not exist yet.
    hashes = freeze(task, suites)
    log(f"{name}: froze {list(hashes)}")

    # 4. Candidates.
    candidates, good_sources = {}, None
    for i, (model, temp) in enumerate(MODELS * 3):
        if sum(1 for c, _ in candidates.values() if c.label == "GOOD") >= n_good:
            break
        cand = propose_candidate(task, sources, target, model, temp, f"{name}_c{i}")
        if not cand:
            continue
        lbl, why = label(JS, task, cand["sources"], oracle["oracle_text"], ws / f"c{i}")
        if lbl == "EXCLUDE":
            continue
        if lbl == "GOOD":
            cid = f"good_{sum(1 for c, _ in candidates.values() if c.label == 'GOOD') + 1:02d}"
            good_sources = good_sources or cand["sources"]
        else:
            cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        candidates[cid] = (Candidate(cid, name, lbl, None if lbl == "GOOD" else "N",
                                     None, model, model, {"temperature": temp}, "ok", why),
                           cand["sources"])

    # 5. Arm S: seeded faults in a verified-GOOD migration, confirmed by the oracle.
    for j, f in enumerate(seed_faults(task, good_sources or sources, n_bad,
                                      f"{name}_seed") if (good_sources or sources) else []):
        lbl, why = label(JS, task, f["sources"], oracle["oracle_text"], ws / f"s{j}")
        if lbl != "BAD":
            continue          # equivalent fault, or it broke the build: excluded (§3.5)
        cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        candidates[cid] = (Candidate(cid, name, "BAD", "S", f["fault_category"],
                                     "claude-opus-5", "claude-opus-5", {}, "ok", why),
                           f["sources"])

    for cid, (cand, src) in candidates.items():
        d = cand.dir(task)
        d.mkdir(parents=True, exist_ok=True)
        (d / "sources.json").write_text(json.dumps(src, indent=2))

    write_metadata(task, [c for c, _ in candidates.values()],
                   extra={"suite_hashes": hashes, "oracle": {
                       "arm_a_usable": oracle["arm_a_usable"], "reason": oracle["reason"],
                       "assignment": oracle["assignment"]},
                       "arms": {k: {x: v[x] for x in v if x != "suite"}
                                for k, v in records.items()}})

    rows = score_task(task, candidates, suites, admissible, JS, ws / "score",
                      existing_text=oracle["existing_text"] if oracle["arm_a_usable"] else None,
                      existing_admissible=oracle["admissible_existing"])
    (task.root / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"{name}: {len(candidates)} candidates, {len(rows)} rows")
    return rows
```

- [ ] **Step 4: Implement the entrypoint**

`run_additional_datasets.py`:

```python
#!/usr/bin/env python
"""Good-vs-bad migration benchmarks. The only entrypoint.

    ./run_additional_datasets.py select --benchmark refactorbench_js
    ./run_additional_datasets.py run --benchmark refactorbench_js --tasks 3
    ./run_additional_datasets.py report --run pilot

Run it with the project venv: `.venv/bin/python run_additional_datasets.py ...`
"""
import argparse
import json
import sys
from pathlib import Path

# Runs are namespaced, matching the existing `results/<run>/<project>/` convention in
# rmt/ and pmb/, so a task's tree is results/additional_datasets/<run>/<benchmark>/<task>.
RESULTS = Path(__file__).resolve().parent / "results/additional_datasets"
BENCHMARKS = ("refactorbench_js",)


def _module(name):
    if name == "refactorbench_js":
        from additional_datasets.benchmarks import refactorbench_js
        return refactorbench_js
    sys.exit(f"unknown benchmark {name!r}; known: {', '.join(BENCHMARKS)}")


def cmd_select(a):
    _module(a.benchmark)
    from additional_datasets.adapters import js_vitest as JS
    names = JS.pure_logic_fixtures()
    print(f"{len(names)} pilot fixtures")
    for n in names:
        print(" ", n)


def cmd_run(a):
    mod = _module(a.benchmark)
    from additional_datasets.adapters import js_vitest as JS
    names = (a.fixtures or JS.pure_logic_fixtures())[:a.tasks]
    root = RESULTS / a.run if a.run else RESULTS
    rows = []
    for n in names:
        try:
            rows += mod.run_task(n, root, n_good=a.good, n_bad=a.bad)
        except Exception as e:  # noqa: BLE001 - one task must not sink the run
            print(f"  {n}: ERROR {type(e).__name__}: {e}", flush=True)
    out = root / "rows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out} ({len(rows)} rows)")


def cmd_report(a):
    from additional_datasets.score import report
    root = RESULTS / a.run if a.run else RESULTS
    rows = json.loads((root / "rows.json").read_text())
    text = report(rows, title=a.run or "pilot")
    (root / "REPORT.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="list the benchmark's task pool")
    s.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("run", help="run the protocol over the task pool")
    s.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    s.add_argument("--run", default="pilot", help="run name (results/additional_datasets/<run>)")
    s.add_argument("--tasks", type=int, default=3, help="how many tasks")
    s.add_argument("--fixtures", nargs="*", help="restrict to these task ids")
    s.add_argument("--good", type=int, default=3, help="GOOD candidates per task")
    s.add_argument("--bad", type=int, default=3, help="seeded BAD candidates per task")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="the spec §5 metrics table")
    s.add_argument("--run", default="pilot")
    s.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_entrypoint.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Run the whole test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 7: Run a single-task pilot and read the result**

```bash
.venv/bin/python run_additional_datasets.py run --benchmark refactorbench_js \
    --run smoke --fixtures binary_search --good 2 --bad 2
.venv/bin/python run_additional_datasets.py report --run smoke
```
Expected: a REPORT.md with per-arm GAR/BRR/AP and a fault-category table. Check by hand before scaling: the mutation arm's `n_kept` in `metadata.json` is non-zero, the freeze recorded both arms, and at least one BAD candidate exists. If the run produced no BAD candidates, the fixture is too easy — pick another and say so.

- [ ] **Step 8: Run the 3-task pilot**

```bash
.venv/bin/python run_additional_datasets.py run --benchmark refactorbench_js \
    --run pilot --tasks 3 --good 3 --bad 3
.venv/bin/python run_additional_datasets.py report --run pilot
```

- [ ] **Step 9: Commit**

```bash
git add additional_datasets/benchmarks/ run_additional_datasets.py tests/test_entrypoint.py
git commit -m "feat(ad): refactorbench-js wired end to end, plus the CLI"
```

---

### Task 13: ARM C — the ceiling that sees the GOOD→BAD diff

**Files:**
- Create: `additional_datasets/prompts/ceiling_testgen.md`
- Create: `additional_datasets/arms/ceiling.py`
- Modify: `additional_datasets/benchmarks/refactorbench_js.py` (wire ARM C in after scoring)
- Create: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: `SuiteResult` (Task 1), `_generate_with_repair` (Task 8), `Candidate` (Task 1).
- Produces: `diff_of(good_sources, bad_sources) -> str`; `is_unreachable(suite_text) -> bool`; `outcome(on_good, on_bad) -> str`; `generate(task, good_sources, bad_sources, adapter, workspace, import_line, framework="vitest", retries=3, tag="ceil") -> (suite_text, record)`; `score_ceiling(task, candidates, adapter, workspace, good_sources, import_line) -> list[dict]`.

`import_line` threads through for the same reason as in Task 8: a model handed the wrong path writes an import that does not resolve, and the suite build-errors before it can say anything about the fault.

`score.py` already names `ceiling` in `ARMS` and `ARM_LABEL` (Task 3), so without this task the report has a column nothing ever fills.

ARM C is **not a technique and must never be described as one.** It is handed the exact
GOOD→BAD diff, in the language the test is written in, with no translation step — so it is
an upper bound, and `1 - its BRR` is the floor under every other arm's miss rate. It is
therefore exempt from the §7 freeze (it is *given* the candidate by construction) and is
scored per BAD candidate rather than once per task, which is why it does not go through
`score_task`.

- [ ] **Step 1: Write the prompt**

`additional_datasets/prompts/ceiling_testgen.md`:

````markdown
Below is a correct implementation and a diff that introduces a behavioural fault into it.
Write one test that passes on the correct implementation and fails on the faulty one.

## The correct implementation

<<SOURCES>>

## The fault that was introduced

```diff
<<DIFF>>
```

## What to write

Write a single <<LANGUAGE>> test file using <<FRAMEWORK>>.

- Import the code under test with EXACTLY this line, character for character:
  `<<IMPORT_LINE>>`
  Do not alter that path.
- Assert exact values, and reach the fault through the public surface.
- The test is executed against both versions and is kept only if it separates them.
- If the fault is genuinely unreachable through the public API, say so in a comment on the
  first line beginning `// UNREACHABLE:` and write the closest test you can.

Return only the test file in a single fenced code block.
````

- [ ] **Step 2: Write the failing test**

Create `tests/test_ceiling.py`:

```python
from additional_datasets.arms import ceiling
from additional_datasets.spec import SuiteResult


def test_diff_of_renders_a_unified_diff_of_the_changed_file():
    good = {"a.js": "const x = 1;\n", "b.js": "const y = 2;\n"}
    bad = {"a.js": "const x = 2;\n", "b.js": "const y = 2;\n"}
    d = ceiling.diff_of(good, bad)
    assert "a.js" in d
    assert "-const x = 1;" in d and "+const x = 2;" in d
    assert "b.js" not in d          # unchanged files are not in the diff


def test_diff_of_covers_a_file_the_fault_added():
    d = ceiling.diff_of({"a.js": "x\n"}, {"a.js": "x\n", "new.js": "y\n"})
    assert "new.js" in d


def test_unreachable_is_detected_from_the_first_line_marker():
    assert ceiling.is_unreachable("// UNREACHABLE: no public accessor\ntest('x', () => {});")
    assert not ceiling.is_unreachable("test('x', () => {});")


def test_outcome_names_why_a_ceiling_attempt_failed():
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    assert ceiling.outcome(green, red) == "KILLED"
    assert ceiling.outcome(red, red) == "F_UNSOUND_ON_GOOD"
    assert ceiling.outcome(green, green) == "F_NOT_DISCRIMINATING"
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_ceiling.py -v`
Expected: FAIL — `ImportError: cannot import name 'ceiling'`

- [ ] **Step 4: Implement `additional_datasets/arms/ceiling.py`**

```python
"""ARM C -- the ceiling: one test per BAD candidate, written from the GOOD->BAD diff.

This arm is shown exactly what is wrong, in the language the test is written in, with no
translation step to lose it in. So it is an upper bound, not a technique: whatever it
cannot kill is a fault that is hard to reach through the public surface rather than one
the other arms mislaid, and `1 - its BRR` is the floor under every other arm's miss rate.

It is exempt from the §7 freeze because it is handed the candidate by construction, and it
is scored per candidate rather than once per task, so it does not go through `score_task`.
"""
import difflib

from ..prompts import load
from .naive import SYSTEM, _bundle, _generate_with_repair


def diff_of(good_sources, bad_sources):
    """Unified diff over every file the fault touched or added."""
    out = []
    for rel in sorted(set(good_sources) | set(bad_sources)):
        a = good_sources.get(rel, "").splitlines(keepends=True)
        b = bad_sources.get(rel, "").splitlines(keepends=True)
        if a == b:
            continue
        out += list(difflib.unified_diff(a, b, fromfile=f"good/{rel}", tofile=f"bad/{rel}"))
    return "".join(out)


def is_unreachable(suite_text):
    return (suite_text or "").lstrip().startswith("// UNREACHABLE:")


def outcome(on_good, on_bad):
    """KILLED | F_UNSOUND_ON_GOOD | F_NOT_DISCRIMINATING."""
    if on_good.status != "green":
        return "F_UNSOUND_ON_GOOD"
    if on_bad.status in ("test_failed", "timeout", "build_error"):
        return "KILLED"
    return "F_NOT_DISCRIMINATING"


def generate(task, good_sources, bad_sources, adapter, workspace, import_line,
             framework="vitest", retries=3, tag="ceil"):
    base = load("ceiling_testgen", sources=_bundle(good_sources),
                diff=diff_of(good_sources, bad_sources), import_line=import_line,
                language=task.language.capitalize(), framework=framework)
    good_ws = adapter.stage(task.task_id, workspace / "good", sources=good_sources)
    suite, rec = _generate_with_repair(base, adapter, good_ws, retries, tag)
    rec["unreachable"] = is_unreachable(suite)
    return suite, rec


def score_ceiling(task, candidates, adapter, workspace, good_sources, import_line):
    """One row per BAD candidate. GOOD candidates are not scored: ARM C has no GAR."""
    from ..score import ACCEPT, BEHAVIOURAL, REJECT

    rows = []
    for cid, (cand, sources) in sorted(candidates.items()):
        if cand.label != "BAD":
            continue
        ws = workspace / f"ceil_{cid}"
        suite, rec = generate(task, good_sources, sources, adapter, ws, import_line,
                              tag=f"ceil_{cid}")
        if not rec["compiles"]:
            rows.append({"task_id": task.task_id, "benchmark": task.benchmark,
                         "candidate_id": cid, "ground_truth": "BAD", "label": "BAD",
                         "bad_arm": cand.bad_arm, "fault_category": cand.fault_category,
                         "arm": "ceiling", "compiles": True,
                         "existing_tests_pass": None, "generated_tests_pass": True,
                         "number_generated_tests": 0, "tests_failed": 0,
                         "verdict": ACCEPT, "rejection_kind": "",
                         "status": "no_suite", "execution_time": 0.0,
                         "ceiling_outcome": "F_NO_SUITE", "unreachable": rec["unreachable"],
                         "model": cand.model, "model_version": cand.model_version,
                         "hidden_oracle_failure": cand.hidden_oracle_failure})
            continue
        good_ws = adapter.stage(task.task_id, ws / "good2", sources=good_sources)
        bad_ws = adapter.stage(task.task_id, ws / "bad", sources=sources)
        on_good = adapter.run_suite(good_ws, suite, f"ceiling@{cid}_good")
        on_bad = adapter.run_suite(bad_ws, suite, f"ceiling@{cid}")
        oc = outcome(on_good, on_bad)
        killed = oc == "KILLED"
        rows.append({"task_id": task.task_id, "benchmark": task.benchmark,
                     "candidate_id": cid, "ground_truth": "BAD", "label": "BAD",
                     "bad_arm": cand.bad_arm, "fault_category": cand.fault_category,
                     "arm": "ceiling", "compiles": True, "existing_tests_pass": None,
                     "generated_tests_pass": not killed,
                     "number_generated_tests": len(on_good.cases),
                     "tests_failed": len(on_bad.failed),
                     "verdict": REJECT if killed else ACCEPT,
                     "rejection_kind": BEHAVIOURAL if killed else "",
                     "status": on_bad.status, "execution_time": on_bad.seconds,
                     "ceiling_outcome": oc, "unreachable": rec["unreachable"],
                     "model": cand.model, "model_version": cand.model_version,
                     "hidden_oracle_failure": cand.hidden_oracle_failure})
    return rows
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_ceiling.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Wire ARM C into the benchmark**

In `additional_datasets/benchmarks/refactorbench_js.py`, replace the final scoring block:

```python
    rows = score_task(task, candidates, suites, admissible, JS, ws / "score",
                      existing_text=oracle["existing_text"] if oracle["arm_a_usable"] else None,
                      existing_admissible=oracle["admissible_existing"])
    (task.root / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"{name}: {len(candidates)} candidates, {len(rows)} rows")
    return rows
```

with:

```python
    rows = score_task(task, candidates, suites, admissible, JS, ws / "score",
                      existing_text=oracle["existing_text"] if oracle["arm_a_usable"] else None,
                      existing_admissible=oracle["admissible_existing"])
    # ARM C is exempt from the freeze: it is handed the GOOD->BAD diff by construction,
    # which is exactly why it is a ceiling and not a technique.
    if good_sources:
        rows += ceiling.score_ceiling(task, candidates, JS, ws / "ceiling", good_sources,
                                      imp)
    else:
        log(f"{name}: no verified-GOOD candidate, ARM C skipped")
    (task.root / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"{name}: {len(candidates)} candidates, {len(rows)} rows")
    return rows
```

and add `ceiling` to the arms import at the top of the file:

```python
from ..arms import ceiling, mutation, naive
```

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 8: Re-run the smoke pilot and confirm ARM C appears**

```bash
.venv/bin/python run_additional_datasets.py run --benchmark refactorbench_js \
    --run smoke --fixtures binary_search --good 2 --bad 2
.venv/bin/python run_additional_datasets.py report --run smoke
```
Expected: the report now has an `ARM C (sees the diff)` row. Sanity check: ARM C's BRR
should be **at or above** every other arm's. If it is below, something is wrong with the
other arms' scoring, not with ARM C — investigate before scaling.

- [ ] **Step 9: Commit**

```bash
git add additional_datasets/arms/ceiling.py additional_datasets/prompts/ceiling_testgen.md \
        additional_datasets/benchmarks/refactorbench_js.py tests/test_ceiling.py
git commit -m "feat(ad): arm C, the diff-seeing ceiling that floors every other arm's miss rate"
```

---

## Verification checklist

Run before declaring this plan done:

- [ ] `.venv/bin/python -m pytest -q` — all green
- [ ] `results/additional_datasets/pilot/REPORT.md` exists and has non-empty GAR/BRR cells
- [ ] `metadata.json` for each task records `suite_hashes` for both `mutation` and `naive`
- [ ] Every BAD candidate has a non-null `hidden_oracle_failure` (no equivalent faults survived)
- [ ] No BAD candidate has `build_status == "build_error"` (spec §20)
- [ ] Arm S and Arm N appear as separate rows in the report, never summed
- [ ] ARM C's BRR is >= every other arm's (it sees the diff; if not, another arm is mis-scored)
- [ ] `grep -r stage1_mutate additional_datasets/` returns nothing (spec §7.4)
- [ ] `js_runtime/node_modules` is not tracked by git
