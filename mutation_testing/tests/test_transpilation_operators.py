"""Unit tests for the transpilation-aware mutation operators.

Every test runs against a string fixture -- no cargo, no corpus, no build -- so the whole
file runs in well under a second and can be run on every edit.
"""

import unittest

import transpilation_operators as tops
from mutate_shared import splice


def gen(text, operators=None):
    """Generate for one fixture, defaulting to the whole registry."""
    return tops.generate(text, "core/src/lib.rs", "demo",
                         operators=operators if operators is not None else tops.OPERATORS)


def replacements(records, genre=None):
    return sorted(r["replacement"] for r in records
                  if genre is None or r["genre"] == genre)


class TestEndianness(unittest.TestCase):
    def test_swaps_method_form(self):
        recs = gen("fn f(x: u32) -> [u8; 4] { x.to_le_bytes() }",
                   [tops.ENDIANNESS])
        self.assertEqual(replacements(recs), ["to_be_bytes", "to_ne_bytes"])

    def test_swaps_path_form(self):
        recs = gen("fn f() -> u32 { u32::from_be_bytes([0, 0, 0, 0]) }",
                   [tops.ENDIANNESS])
        self.assertEqual(replacements(recs), ["from_le_bytes", "from_ne_bytes"])

    def test_ignores_unrelated_methods(self):
        self.assertEqual(gen("fn f(v: Vec<u8>) -> usize { v.len() }",
                             [tops.ENDIANNESS]), [])

    def test_genre_is_prefixed(self):
        recs = gen("fn f(x: u32) -> [u8; 4] { x.to_le_bytes() }", [tops.ENDIANNESS])
        self.assertTrue(all(r["genre"] == "TranspileEndianness" for r in recs))


class TestRecordShape(unittest.TestCase):
    SRC = "fn convert(x: u32) -> [u8; 4] {\n    x.to_le_bytes()\n}\n"

    def test_has_exactly_the_harness_keys(self):
        rec = gen(self.SRC, [tops.ENDIANNESS])[0]
        self.assertEqual(set(rec), {"name", "package", "file", "function",
                                    "span", "replacement", "genre"})

    def test_function_context_is_captured(self):
        rec = gen(self.SRC, [tops.ENDIANNESS])[0]
        self.assertEqual(rec["function"]["function_name"], "convert")
        self.assertEqual(rec["function"]["return_type"], "[u8; 4]")

    def test_span_points_at_the_method_name(self):
        rec = gen(self.SRC, [tops.ENDIANNESS])[0]
        self.assertEqual(rec["span"]["start"], {"line": 2, "column": 7})
        self.assertEqual(rec["span"]["end"], {"line": 2, "column": 18})

    def test_name_matches_cargo_mutants_shape(self):
        rec = gen(self.SRC, [tops.ENDIANNESS])[0]
        self.assertTrue(rec["name"].startswith("core/src/lib.rs:2:7: replace to_le_bytes"))


class TestRoundTrip(unittest.TestCase):
    """Every record must splice to exactly the edit the generator intended.

    tree-sitter counts bytes; mutate_shared._offset counts characters. They agree on ASCII
    and diverge otherwise, so this is the test that stops a silent mis-edit.
    """

    FIXTURES = [
        "fn f(x: u32) -> [u8; 4] { x.to_le_bytes() }",
        "// ééé non-ascii comment\nfn f(x: u32) -> [u8; 4] { x.to_le_bytes() }",
        "fn f() -> &'static str { \"üüü\" }\nfn g(x: u32) -> u32 { x.to_be() }",
    ]

    def test_splice_reproduces_intended_edit(self):
        for text in self.FIXTURES:
            with self.subTest(text=text[:30]):
                for rec in gen(text):
                    spliced = splice(text, rec["span"], rec["replacement"])
                    self.assertNotEqual(spliced, text)
                    self.assertIn(rec["replacement"], spliced)

    def test_non_ascii_prefix_does_not_shift_the_span(self):
        text = "// ééé\nfn f(x: u32) -> u32 { x.to_be() }"
        recs = gen(text, [tops.ENDIANNESS])
        self.assertTrue(recs)
        for rec in recs:
            self.assertEqual(splice(text, rec["span"], rec["replacement"]),
                             text.replace("to_be", rec["replacement"]))


if __name__ == "__main__":
    unittest.main()
