"""Task and candidate records, and the metadata.json the spec's §9 requires.

One benchmark instance is a Task (a trusted P0 plus the migration spec M) and a set of
Candidates (compiling post-migration implementations, each labelled GOOD or BAD by a
hidden oracle the test-generating arms never see).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def sources_hash(sources) -> str:
    """Content hash of a candidate's file set -- the analogue of `candidate_commit`.

    Candidates are generated file sets, never commits, so §9's commit field is filled with
    something that identifies the same thing: sha256 over the sorted {path: text} mapping.
    Path-sensitive on purpose -- moving a function between two files is a distinct
    candidate even when the concatenated text is unchanged.
    """
    return hashlib.sha256(
        json.dumps(sources, sort_keys=True).encode()).hexdigest()


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
    # Candidates here are generated file sets, not commits, so the analogue of a commit is
    # a content hash of the candidate's sources -- `sources_hash(sources)`, set by the
    # benchmark when the candidate is built. Defaulted so a Candidate built positionally is
    # unaffected; `None` means the producer did not identify its sources, which is a gap in
    # the artifact rather than a valid value.
    candidate_commit: str | None = None

    def dir(self, task: Task) -> Path:
        return task.candidates_dir / self.candidate_id

    def as_row(self) -> dict:
        return {"candidate_id": self.candidate_id, "ground_truth_label": self.label,
                "bad_arm": self.bad_arm, "fault_category": self.fault_category,
                "model": self.model, "model_version": self.model_version,
                "generation_parameters": self.generation_parameters,
                "build_status": self.build_status,
                "hidden_oracle_failure": self.hidden_oracle_failure,
                "candidate_commit": self.candidate_commit}


@dataclass(frozen=True)
class SuiteResult:
    """One test suite's outcome on one implementation."""
    status: str                     # green|test_failed|build_error|timeout|no_tests|error
    cases: dict                     # case name -> "ok" | "FAILED"
    rc: int
    seconds: float
    out: str = ""
    # Which of a MULTI-FILE suite's files failed to load at all, by index. A load failure is
    # a claim about the candidate's SHAPE -- a renamed export, a moved module -- and it is
    # invisible in `cases`, because the file's cases are simply absent. Carrying it
    # separately is what lets `score.verdict` tell "this arm's third file could not resolve
    # its import" (structural) apart from "the suite crashed part-way through" (behavioural).
    # Empty for a single-file run, where `status == "build_error"` already says it.
    load_failures: tuple = ()

    @property
    def passed(self) -> list:
        return sorted(k for k, v in self.cases.items() if v == "ok")

    @property
    def failed(self) -> list:
        return sorted(k for k, v in self.cases.items() if v == "FAILED")

    def as_row(self) -> dict:
        return {"status": self.status, "n_cases": len(self.cases),
                "n_passed": len(self.passed), "n_failed": len(self.failed),
                "n_load_failures": len(self.load_failures),
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
