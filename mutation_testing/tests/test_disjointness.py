"""The transpilation operators must not duplicate cargo-mutants' own.

The completed five-project run left 19 outcomes.json files holding every mutant
cargo-mutants generates for those projects. Regenerating over the same sources and
intersecting on (file, span, replacement) turns "these operators are different" from an
argument into an assertion.
"""

import json
import unittest
from pathlib import Path

import transpilation_operators as tops

ROOT = Path(__file__).resolve().parent.parent / "results/claude/opus5_sample"
OUT = ROOT / "_mutants_out"
WS = ROOT / "_workspaces"
PROJECTS = ["ulidgen", "clog", "emlang", "clhash", "cset"]


def _key(file, span, replacement):
    return (file,
            span["start"]["line"], span["start"]["column"],
            span["end"]["line"], span["end"]["column"],
            replacement)


def cargo_keys():
    keys = set()
    for f in OUT.rglob("mutants.out/outcomes.json"):
        for o in json.loads(f.read_text())["outcomes"]:
            m = o["scenario"]["Mutant"]
            keys.add(_key(m["file"], m["span"], m["replacement"]))
    return keys


def corpus_sources():
    """(project, workspace-relative path, text) for every source the harness mutates."""
    for project in PROJECTS:
        core = WS / project / "core"
        if not core.is_dir():
            continue
        for src in sorted((core / "src").rglob("*.rs")):
            # never mutate test bins or interfaces -- must match list_transpile_mutants,
            # or this guard measures a different population than the harness scores.
            # `#[cfg(test)]` regions are excluded inside generate(), so there is nothing
            # to repeat here.
            parts = src.relative_to(core / "src").parts
            if "bin" in parts or "interfaces" in parts:
                continue
            # cargo-mutants records paths relative to the workspace, e.g. core/src/ulid.rs
            rel = "core/" + str(src.relative_to(core))
            yield project, rel, src.read_text(encoding="utf-8", errors="replace")


def transpile_keys(operators=None):
    keys = set()
    for project, rel, text in corpus_sources():
        for r in tops.generate(text, rel, project, operators=operators):
            keys.add(_key(r["file"], r["span"], r["replacement"]))
    return keys


class TestDisjointness(unittest.TestCase):
    def test_corpus_is_present(self):
        self.assertTrue(cargo_keys(), "no stored cargo-mutants outcomes found")
        self.assertTrue(transpile_keys(), "generator produced nothing on the corpus")

    def test_no_shared_edits(self):
        overlap = cargo_keys() & transpile_keys()
        self.assertEqual(overlap, set(),
                         f"{len(overlap)} edits collide with cargo-mutants: "
                         f"{sorted(overlap)[:5]}")

    def test_operators_do_not_claim_each_others_edits(self):
        """No two operators may generate the same edit.

        ``generate``'s dedup key is (file, span, replacement) with no operator identity,
        so two operators producing one edit silently collapse into a single record whose
        genre is decided by registry order -- and the genre is what the whole per-operator
        breakdown is computed from. Nothing else in the suite defends this, so a twelfth
        operator overlapping cast-width or overflow-discipline would be invisible.

        The key carries the project because every workspace has its own
        ``core/src/lib.rs``, which would otherwise collide across projects.
        """
        per_operator = {}
        for project, rel, text in corpus_sources():
            for op in tops.OPERATORS:
                keys = per_operator.setdefault(op.name, set())
                for r in tops.generate(text, rel, project, operators=[op]):
                    keys.add((project,) + _key(r["file"], r["span"], r["replacement"]))

        union = set().union(*per_operator.values())
        total = sum(len(k) for k in per_operator.values())
        if total != len(union):
            shared = sorted(
                (a, b, len(per_operator[a] & per_operator[b]))
                for i, a in enumerate(sorted(per_operator))
                for b in sorted(per_operator)[i + 1:]
                if per_operator[a] & per_operator[b])
            self.fail(f"{total - len(union)} edits claimed by more than one operator: "
                      f"{shared}")


if __name__ == "__main__":
    unittest.main()
