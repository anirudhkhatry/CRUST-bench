"""Unit tests for the transpilation-aware mutation operators.

Every test runs against a string fixture -- no cargo, no corpus, no build -- so the whole
file runs in well under a second and can be run on every edit.
"""

import unittest
from unittest.mock import patch

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


class TestRecordVerification(unittest.TestCase):
    """The splice round-trip check alone is tautological: ``_position`` and
    ``mutate_shared._offset`` are exact inverses for any integer, so it would pass even
    given a wrong byte->char conversion. This exercises the independent ground-truth
    check that ``_record`` must also perform.
    """

    def test_bad_char_index_is_rejected(self):
        text = "// ééé non-ascii comment\nfn f(x: u32) -> [u8; 4] { x.to_le_bytes() }"

        def broken_char_index(src, byte_offset):
            # Simulates forgetting the .decode("utf-8") step: treats a byte offset as
            # though it were already a character offset. Wrong the moment the file has
            # a multi-byte character before the target node.
            return byte_offset

        with patch.object(tops, "_char_index", side_effect=broken_char_index):
            recs = gen(text, [tops.ENDIANNESS])
        self.assertEqual(recs, [])


class TestCastWidth(unittest.TestCase):
    def test_widens_and_narrows_unsigned(self):
        recs = gen("fn f(x: u8) -> u32 { x as u32 }", [tops.CAST_WIDTH])
        self.assertEqual(replacements(recs), ["u16", "u64"])

    def test_stays_within_signedness(self):
        recs = gen("fn f(x: i8) -> i32 { x as i32 }", [tops.CAST_WIDTH])
        self.assertEqual(replacements(recs), ["i16", "i64"])

    def test_usize_maps_to_the_unsigned_ladder(self):
        recs = gen("fn f(n: u64) -> usize { n as usize }", [tops.CAST_WIDTH])
        self.assertIn("u32", replacements(recs))

    def test_ignores_float_casts(self):
        self.assertEqual(gen("fn f(x: i32) -> f64 { x as f64 }", [tops.CAST_WIDTH]), [])


class TestCastSign(unittest.TestCase):
    def test_flips_sign_at_same_width(self):
        recs = gen("fn f(x: u8) -> i32 { x as i32 }", [tops.CAST_SIGN])
        self.assertEqual(replacements(recs), ["u32"])

    def test_flips_unsigned_to_signed(self):
        recs = gen("fn f(x: i8) -> u32 { x as u32 }", [tops.CAST_SIGN])
        self.assertEqual(replacements(recs), ["i32"])

    def test_ignores_usize(self):
        self.assertEqual(gen("fn f(n: u64) -> usize { n as usize }", [tops.CAST_SIGN]), [])


class TestOverflowDiscipline(unittest.TestCase):
    def test_swaps_wrapping_for_saturating(self):
        recs = gen("fn f(a: u32, b: u32) -> u32 { a.wrapping_add(b) }",
                   [tops.OVERFLOW_DISCIPLINE])
        self.assertIn("saturating_add", replacements(recs))

    def test_drops_wrapping_for_the_bare_operator(self):
        recs = gen("fn f(a: u32, b: u32) -> u32 { a.wrapping_add(b) }",
                   [tops.OVERFLOW_DISCIPLINE])
        self.assertIn("a + b", replacements(recs))

    def test_adds_wrapping_to_a_bare_operator(self):
        recs = gen("fn f(a: u32, b: u32) -> u32 { a + b }", [tops.OVERFLOW_DISCIPLINE])
        self.assertEqual(replacements(recs), ["a.wrapping_add(b)"])

    def test_ignores_shifts(self):
        """Shifts belong to shift-discipline, so the two operators cannot double-count."""
        self.assertEqual(gen("fn f(a: u32) -> u32 { a << 2 }",
                             [tops.OVERFLOW_DISCIPLINE]), [])

    def test_ignores_non_arithmetic_operators(self):
        self.assertEqual(gen("fn f(a: u32, b: u32) -> bool { a == b }",
                             [tops.OVERFLOW_DISCIPLINE]), [])


class TestShiftDiscipline(unittest.TestCase):
    def test_wraps_a_left_shift(self):
        recs = gen("fn f(a: u32) -> u32 { a << 2 }", [tops.SHIFT_DISCIPLINE])
        self.assertEqual(replacements(recs), ["a.wrapping_shl(2)"])

    def test_wraps_a_right_shift(self):
        recs = gen("fn f(a: u32) -> u32 { a >> 2 }", [tops.SHIFT_DISCIPLINE])
        self.assertEqual(replacements(recs), ["a.wrapping_shr(2)"])


class TestDisjointGenres(unittest.TestCase):
    def test_every_genre_is_prefixed(self):
        recs = gen("fn f(a: u32, b: u8) -> u32 { a.wrapping_add(b as u32) << 1 }")
        self.assertTrue(recs)
        self.assertTrue(all(r["genre"].startswith("Transpile") for r in recs))


if __name__ == "__main__":
    unittest.main()
