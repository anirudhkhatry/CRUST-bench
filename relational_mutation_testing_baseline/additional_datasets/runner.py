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
