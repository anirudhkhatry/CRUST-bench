import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_additional_datasets as rad
from additional_datasets.runner import LeakageError


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


def _run_args(run_name, **kw):
    d = dict(benchmark="refactorbench_js", fixtures=["fake_fixture"], tasks=1,
             run=run_name, good=1, bad=1, draws=2)
    d.update(kw)
    return SimpleNamespace(**d)


def test_a_fired_leakage_control_aborts_the_run_and_writes_no_artifact(tmp_path, monkeypatch):
    """A `LeakageError` must not be swallowed by the per-task handler.

    If `cmd_run` caught it like any other exception, the loop would keep going, an empty
    or partial `rows.json` would be written, and `report` would render a table from it as
    if the run were clean -- exactly the failure mode spec §15 exists to prevent.
    """
    from additional_datasets.benchmarks import refactorbench_js as rjs

    def fake_run_task(name, root, n_good, n_bad, draws):
        raise LeakageError(f"{name}: candidates already exist")

    monkeypatch.setattr(rjs, "run_task", fake_run_task)
    monkeypatch.setattr(rad, "RESULTS", tmp_path)

    with pytest.raises(SystemExit):
        rad.cmd_run(_run_args("r"))
    assert not (tmp_path / "r" / "rows.json").exists()


def test_a_non_leakage_task_failure_is_recorded_in_the_artifact_and_flagged_in_the_report(
        tmp_path, monkeypatch):
    """Any other per-task exception must not just print-and-vanish.

    Before this fix, the only record of a failed task was a line to stdout; `rows.json`
    and `REPORT.md` were written as though the run were complete. The failure must survive
    into the artifact so `report` can state, prominently, that it did not.
    """
    from additional_datasets.benchmarks import refactorbench_js as rjs

    def fake_run_task(name, root, n_good, n_bad, draws):
        raise RuntimeError("vitest exploded")

    monkeypatch.setattr(rjs, "run_task", fake_run_task)
    monkeypatch.setattr(rad, "RESULTS", tmp_path)

    rad.cmd_run(_run_args("r"))
    blob = json.loads((tmp_path / "r" / "rows.json").read_text())
    assert blob["rows"] == []
    assert blob["failures"] == [
        {"task": "fake_fixture", "error": "RuntimeError: vitest exploded"}]

    rad.cmd_report(SimpleNamespace(run="r"))
    text = (tmp_path / "r" / "REPORT.md").read_text()
    assert "1 task(s) failed" in text
    assert "fake_fixture" in text and "vitest exploded" in text
