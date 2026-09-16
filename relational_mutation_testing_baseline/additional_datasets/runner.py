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


def suite_files(suite):
    """An arm's suite as a LIST of test-file texts. A bare string is a one-file suite.

    Arms return lists (`assemble`); ARM A's suite is one developer file and is passed as a
    string. Normalising here rather than at each call site is what keeps the freeze hash
    and the scoring run reading the same thing.
    """
    return [suite] if isinstance(suite, str) else list(suite)


def _suite_sha(files) -> str:
    """Hash over the ORDERED file list. Order is part of the suite's identity: case names
    are file-index prefixed, so two suites with the same files in a different order are
    different suites as far as every admissible set is concerned."""
    return _sha(json.dumps(files))


def freeze(task: Task, suites: dict) -> dict:
    """Write every arm's suite, hash it, record the hashes. Candidates must not exist.

    A value may be one file's text or a list of them; either way the arm's files land as
    `suite_0.<ext>`, `suite_1.<ext>`, ... and the recorded hash covers the whole ordered
    list.
    """
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
    for arm, suite in sorted(suites.items()):
        # An arm that produced no suite (e.g. `naive.generate` keeping nothing) must be
        # omitted by the caller, not passed here. Letting `None` -- or a list holding one --
        # through would otherwise reach `Path.write_text` and raise a bare `TypeError` deep
        # inside the freeze gate: a crash that reads like a harness bug rather than "this
        # arm has no suite". An EMPTY list is the same mistake spelt differently.
        files = suite_files(suite) if suite is not None else None
        if files is None or not files or not all(isinstance(f, str) for f in files):
            raise LeakageError(
                f"{task.task_id}/{arm}: suite is not one or more test files "
                f"({type(suite).__name__}) -- omit arms with no suite rather than "
                f"freezing one")
        d = task.generated_tests_dir / arm
        d.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(files):
            (d / f"suite_{i}.{ext}").write_text(text)
        hashes[arm] = _suite_sha(files)
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
    for arm, suite in sorted(suites.items()):
        if arm not in recorded:
            raise LeakageError(f"{task.task_id}/{arm}: never frozen")
        if recorded[arm] != _suite_sha(suite_files(suite)):
            raise LeakageError(f"{task.task_id}/{arm}: suite changed after freezing")


def assert_no_leak(prompt: str, forbidden) -> None:
    """No candidate source, ground-truth label or hidden-oracle text in a prompt."""
    for frag in forbidden:
        frag = (frag or "").strip()
        if len(frag) >= MIN_LEAK_FRAGMENT and frag in prompt:
            raise LeakageError(f"leaked into a test-generation prompt: {frag[:60]!r}...")

# --- scoring ---------------------------------------------------------------------
from .score import ACCEPT, verdict          # noqa: E402  (kept next to its use)


def _row(task, cand, arm, res, v, kind, combined=None):
    """One §19 row. `combined` is retained as a column but is always None -- see below.

    `number_generated_tests` is the size of the suite as actually RUN, not the admissible
    subset -- the admissible set is filtered to cases that passed the trusted P0, so using
    it here would under-report every suite's size and disagree with the ceiling arm, which
    counts the same column from its executed suite.
    """
    return {"task_id": task.task_id, "benchmark": task.benchmark,
            "candidate_id": cand.candidate_id, "ground_truth": cand.label,
            "label": cand.label, "bad_arm": cand.bad_arm,
            "fault_category": cand.fault_category, "arm": arm,
            "compiles": cand.build_status == "ok",
            "existing_tests_pass": (v == ACCEPT) if arm == "existing" else None,
            "generated_tests_pass": v == ACCEPT,
            "combined_tests_pass": combined,
            "number_generated_tests": len(res.cases), "tests_failed": len(res.failed),
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
    arms = dict(suites)
    if existing_text:
        arms["existing"] = existing_text
    # Baseline A's suite is frozen and hash-checked like every other arm. Validating only
    # `suites` would leave the one arm whose text comes from the fixture outside the gate
    # that is supposed to be a mechanical guarantee rather than a promise.
    assert_frozen(task, arms)

    adm = {**{a: list(v) for a, v in admissible.items()},
           **({"existing": list(existing_admissible)} if existing_text else {})}
    # An arm with no admissible set can never fire a behavioural rejection, so every
    # candidate would be silently ACCEPTed and the run would produce plausible, wrong
    # numbers. Empty counts as missing: passing `existing_text` without
    # `existing_admissible` would otherwise slip through and read as BRR 0 for Baseline A.
    empty = sorted(a for a in arms if not adm.get(a))
    if empty:
        raise LeakageError(f"{task.task_id}: no admissible set for arm(s) "
                           f"{', '.join(empty)} -- every candidate would be "
                           f"accepted by default")

    rows = []
    for cid, (cand, sources) in sorted(candidates.items()):
        results = {}
        for arm, suite in sorted(arms.items()):
            ws = adapter.stage(task.task_id, workspace / f"{arm}_{cid}", sources=sources)
            # Every arm goes through `run_suites`, single-file ones included, so that the
            # case names a candidate is scored on are built the same way as the ones in the
            # admissible set. Mixing the two namespaces would silently make every
            # admissible case "missing".
            res = adapter.run_suites(ws, suite_files(suite), f"{arm}@{cid}")
            results[arm] = (res, *verdict(res, set(adm.get(arm, []))))
        for arm, (res, v, kind) in sorted(results.items()):
            # `combined_tests_pass` (design §14's fifth baseline) is a retained spec §19
            # column and is None on EVERY row. It used to be "Baseline A accepts AND this
            # arm accepts", which asked whether the generated suite catches what the
            # developer suite misses. Since spec §4.4 the developer suite IS the oracle
            # that assigned the labels, so Baseline A accepts exactly the GOOD candidates
            # and rejects exactly the BAD ones by construction: the conjunction would
            # reduce to "this arm accepts, on the GOOD candidates only" and answer
            # nothing. A column of Nones is honest; a column of derived tautologies is
            # not. The key stays so downstream readers of rows.json do not KeyError.
            rows.append(_row(task, cand, arm, res, v, kind))
    return rows
