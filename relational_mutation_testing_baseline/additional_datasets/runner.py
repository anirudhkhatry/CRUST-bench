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
    try:
        ext = EXT[task.language]
    except KeyError:
        raise LeakageError(
            f"{task.task_id}: unknown language {task.language!r}; known: "
            f"{', '.join(sorted(EXT))}") from None
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
    """Every suite must have been frozen, and must still match what was frozen.

    This fails CLOSED on purpose. A missing metadata.json, a missing `suite_hashes` key,
    or an arm absent from it all mean the freeze never happened for that suite -- and a
    leakage gate that returns normally when it was never armed is worse than no gate,
    because the run still produces numbers that look valid.
    """
    meta = task.root / "metadata.json"
    if not meta.exists():
        raise LeakageError(f"{task.task_id}: no metadata.json -- suites were never frozen")
    blob = json.loads(meta.read_text())
    if "suite_hashes" not in blob:
        raise LeakageError(f"{task.task_id}: metadata.json has no suite_hashes -- "
                           f"freeze() was never called")
    recorded = blob["suite_hashes"]
    for arm, text in sorted(suites.items()):
        if arm not in recorded:
            raise LeakageError(f"{task.task_id}/{arm}: never frozen")
        if recorded[arm] != _sha(text):
            raise LeakageError(f"{task.task_id}/{arm}: suite changed after freezing")


def assert_no_leak(prompt: str, forbidden) -> None:
    """No candidate source, ground-truth label or hidden-oracle text in a prompt."""
    for frag in forbidden:
        frag = (frag or "").strip()
        if len(frag) >= MIN_LEAK_FRAGMENT and frag in prompt:
            raise LeakageError(f"leaked into a test-generation prompt: {frag[:60]!r}...")

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
