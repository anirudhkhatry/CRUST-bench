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


def transpile_keys():
    keys = set()
    for project in PROJECTS:
        core = WS / project / "core"
        if not core.is_dir():
            continue
        for src in (core / "src").rglob("*.rs"):
            if "bin" in src.relative_to(core / "src").parts:
                continue
            # cargo-mutants records paths relative to the workspace, e.g. core/src/ulid.rs
            rel = "core/" + str(src.relative_to(core))
            for r in tops.generate(src.read_text(encoding="utf-8", errors="replace"),
                                   rel, project):
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


if __name__ == "__main__":
    unittest.main()
