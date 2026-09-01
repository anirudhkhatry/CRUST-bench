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
        self.assertEqual(replacements(recs), ["to_be_bytes"])

    def test_swaps_path_form(self):
        recs = gen("fn f() -> u32 { u32::from_be_bytes([0, 0, 0, 0]) }",
                   [tops.ENDIANNESS])
        self.assertEqual(replacements(recs), ["from_le_bytes"])

    def test_never_emits_a_native_endian_replacement(self):
        """On a little-endian host `ne` is `le`, so a `*_ne_*` replacement is either a
        no-op (from `le`) or a duplicate of the `le` swap (from `be`)."""
        for src in ("fn f(x: u32) -> [u8; 4] { x.to_le_bytes() }",
                    "fn f(x: u32) -> [u8; 4] { x.to_be_bytes() }",
                    "fn f() -> u32 { u32::from_le_bytes([0; 4]) }",
                    "fn f() -> u32 { u32::from_be_bytes([0; 4]) }",
                    "fn f() -> u32 { u32::from_ne_bytes([0; 4]) }"):
            with self.subTest(src=src):
                self.assertNotIn("ne", "".join(replacements(gen(src, [tops.ENDIANNESS]))))

    def test_native_endian_swaps_only_to_big_endian(self):
        """`from_ne_bytes -> from_le_bytes` is the same no-op read backwards; the swap to
        big-endian is genuinely live and is kept."""
        recs = gen("fn f() -> u32 { u32::from_ne_bytes([0; 4]) }", [tops.ENDIANNESS])
        self.assertEqual(replacements(recs), ["from_be_bytes"])

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

    def test_names_are_unique_within_a_file(self):
        """Nested wrapping calls share a start position and a label, so the name alone
        collides. `--only-names-from` filters on the raw name, so it must not."""
        text = ("fn f(a: u64, b: u64, c: u64) -> u64 "
                "{ a.wrapping_add(b).wrapping_add(c) }")
        names = [r["name"] for r in gen(text)]
        self.assertEqual(len(names), len(set(names)), names)
        self.assertTrue(any(n.endswith("#2") for n in names), names)


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
        self.assertIn("(a + b)", replacements(recs))

    def test_adds_wrapping_to_a_bare_operator(self):
        recs = gen("fn f(a: u32, b: u32) -> u32 { a + b }", [tops.OVERFLOW_DISCIPLINE])
        self.assertEqual(replacements(recs), ["a.wrapping_add(b)"])

    def test_parenthesises_a_non_atomic_left_operand(self):
        """`365 * yoe + yoe / 4` must not become `365 * yoe.wrapping_add(yoe / 4)`."""
        recs = gen("fn f(a: u32, b: u32) -> u32 { a * b + b / 4 }",
                   [tops.OVERFLOW_DISCIPLINE])
        self.assertIn("(a * b).wrapping_add(b / 4)", replacements(recs))

    def test_bare_operator_replacement_cannot_escape_its_context(self):
        """`acc.wrapping_mul(P).wrapping_add(z)` must not become `acc * P.wrapping_add(z)`."""
        recs = gen("fn f(a: u32, p: u32, z: u32) -> u32 { a.wrapping_mul(p).wrapping_add(z) }",
                   [tops.OVERFLOW_DISCIPLINE])
        self.assertIn("(a * p)", replacements(recs))
        self.assertNotIn("a * p", replacements(recs))

    def test_bare_operator_parenthesises_a_compound_argument(self):
        """`a.wrapping_mul(x + 1)` must not become `a * x + 1`."""
        recs = gen("fn f(a: u32, x: u32) -> u32 { a.wrapping_mul(x + 1) }",
                   [tops.OVERFLOW_DISCIPLINE])
        self.assertIn("(a * (x + 1))", replacements(recs))

    def test_ignores_shifts(self):
        """Shifts belong to shift-discipline, so the two operators cannot double-count."""
        self.assertEqual(gen("fn f(a: u32) -> u32 { a << 2 }",
                             [tops.OVERFLOW_DISCIPLINE]), [])

    def test_ignores_division_and_remainder(self):
        """C defines no wraparound for `/` or `%`, and `wrapping_div`/`wrapping_rem`
        differ from `/`/`%` only at `iN::MIN / -1` -- never for unsigned types. Equivalent
        by construction, so never generated."""
        for src in ("fn f(a: u32, b: u32) -> u32 { a / b }",
                    "fn f(a: u32, b: u32) -> u32 { a % b }",
                    "fn f(a: u32, b: u32) -> u32 { a.wrapping_div(b) }",
                    "fn f(a: u32, b: u32) -> u32 { a.wrapping_rem(b) }",
                    "fn f(a: u32, b: u32) -> u32 { a.saturating_div(b) }"):
            with self.subTest(src=src):
                self.assertEqual(gen(src, [tops.OVERFLOW_DISCIPLINE]), [])

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

    def test_ignores_non_shift_operators(self):
        """The negative case: only `<<`/`>>` are shifts. `a + b` belongs to
        overflow-discipline, and `a + b << 2` must parenthesise rather than reassociate."""
        self.assertEqual(gen("fn f(a: u32, b: u32) -> u32 { a + b }",
                             [tops.SHIFT_DISCIPLINE]), [])

    def test_parenthesises_a_non_atomic_left_operand(self):
        recs = gen("fn f(a: u32, b: u32) -> u32 { a + b << 2 }", [tops.SHIFT_DISCIPLINE])
        self.assertEqual(replacements(recs), ["(a + b).wrapping_shl(2)"])


class TestGenrePrefix(unittest.TestCase):
    """Every genre must be namespaced, so a transpilation mutant is never mistaken for a
    cargo-mutants one downstream. (Operator *disjointness* is a corpus-scale property and
    is asserted in tests/test_disjointness.py, not here.)"""

    def test_every_genre_is_prefixed(self):
        recs = gen("fn f(a: u32, b: u8) -> u32 { a.wrapping_add(b as u32) << 1 }")
        self.assertTrue(recs)
        self.assertTrue(all(r["genre"].startswith("Transpile") for r in recs))


class TestCfgTestExclusion(unittest.TestCase):
    """Inline `#[cfg(test)]` items are the crate's own unit tests, not scored code.

    The exclusion lives in ``generate`` so every caller shares it -- duplicating it in the
    workspace walker and in the disjointness guard is exactly the drift the `interfaces`
    exclusion already cost once.
    """

    def test_skips_a_cfg_test_module(self):
        text = ("fn real(a: u32, b: u32) -> u32 { a + b }\n"
                "#[cfg(test)]\nmod tests {\n"
                "    fn t(c: u32, d: u32) -> u32 { c + d }\n}\n")
        recs = gen(text, [tops.OVERFLOW_DISCIPLINE])
        self.assertEqual(replacements(recs), ["a.wrapping_add(b)"])

    def test_skips_a_cfg_test_function(self):
        text = ("fn real(a: u32, b: u32) -> u32 { a + b }\n"
                "#[cfg(test)]\nfn helper(c: u32, d: u32) -> u32 { c + d }\n")
        self.assertEqual(replacements(gen(text, [tops.OVERFLOW_DISCIPLINE])),
                         ["a.wrapping_add(b)"])

    def test_skips_a_compound_cfg_predicate(self):
        text = ("#[cfg(all(test, feature = \"x\"))]\n"
                "mod tests { fn t(c: u32, d: u32) -> u32 { c + d } }\n")
        self.assertEqual(gen(text, [tops.OVERFLOW_DISCIPLINE]), [])

    def test_keeps_a_module_gated_on_something_else(self):
        text = ("#[cfg(feature = \"x\")]\n"
                "mod extra { fn t(c: u32, d: u32) -> u32 { c + d } }\n")
        self.assertEqual(replacements(gen(text, [tops.OVERFLOW_DISCIPLINE])),
                         ["c.wrapping_add(d)"])

    def test_keeps_a_plain_module_named_tests(self):
        """The skip is structural, not a name match on `mod tests`."""
        text = "mod tests { fn t(c: u32, d: u32) -> u32 { c + d } }\n"
        self.assertEqual(replacements(gen(text, [tops.OVERFLOW_DISCIPLINE])),
                         ["c.wrapping_add(d)"])


class TestTextUnit(unittest.TestCase):
    def test_swaps_chars_for_bytes(self):
        recs = gen("fn f(s: &str) -> usize { s.chars().count() }", [tops.TEXT_UNIT])
        self.assertIn("bytes", replacements(recs))

    def test_rewrites_len_as_char_count(self):
        recs = gen("fn f(s: &str) -> usize { s.len() }", [tops.TEXT_UNIT])
        self.assertIn("s.chars().count()", replacements(recs))

    def test_ignores_unrelated_calls(self):
        self.assertEqual(gen("fn f(v: Vec<u8>) -> Option<&u8> { v.first() }",
                             [tops.TEXT_UNIT]), [])


class TestFloatWidth(unittest.TestCase):
    def test_narrows_f64(self):
        recs = gen("fn f(x: i32) -> f64 { x as f64 }", [tops.FLOAT_WIDTH])
        self.assertEqual(replacements(recs), ["f32"])

    def test_widens_f32(self):
        recs = gen("fn f(x: i32) -> f32 { x as f32 }", [tops.FLOAT_WIDTH])
        self.assertEqual(replacements(recs), ["f64"])

    def test_ignores_integer_casts(self):
        self.assertEqual(gen("fn f(x: u8) -> u32 { x as u32 }", [tops.FLOAT_WIDTH]), [])


class TestLostInPlaceUpdate(unittest.TestCase):
    def test_clones_a_mutable_reference_argument(self):
        recs = gen("fn f(acc: &mut u32) { g(&mut *acc); }", [tops.LOST_IN_PLACE_UPDATE])
        self.assertIn("&mut (*acc).clone()", replacements(recs))

    def test_clones_a_mutable_reference_to_a_binding(self):
        recs = gen("fn f() { let mut a = 0u32; g(&mut a); }",
                   [tops.LOST_IN_PLACE_UPDATE])
        self.assertIn("&mut a.clone()", replacements(recs))

    def test_ignores_shared_references(self):
        self.assertEqual(gen("fn f() { let a = 0u32; g(&a); }",
                             [tops.LOST_IN_PLACE_UPDATE]), [])


class TestErrorPathErasure(unittest.TestCase):
    def test_replaces_try_with_unwrap_or_default(self):
        recs = gen("fn f(v: Result<u32, ()>) -> u32 { v? }", [tops.ERROR_PATH_ERASURE])
        self.assertEqual(replacements(recs), ["v.unwrap_or_default()"])

    def test_ignores_plain_expressions(self):
        self.assertEqual(gen("fn f(v: u32) -> u32 { v }", [tops.ERROR_PATH_ERASURE]), [])


class TestRangeBound(unittest.TestCase):
    def test_makes_an_exclusive_range_inclusive(self):
        recs = gen("fn f(b: &[u8]) -> &[u8] { &b[1..4] }", [tops.RANGE_BOUND])
        self.assertIn("1..=4", replacements(recs))

    def test_shortens_an_exclusive_range(self):
        recs = gen("fn f(b: &[u8]) -> &[u8] { &b[1..4] }", [tops.RANGE_BOUND])
        self.assertIn("1..4 - 1", replacements(recs))

    def test_makes_an_inclusive_range_exclusive(self):
        recs = gen("fn f(b: &[u8]) -> &[u8] { &b[1..=4] }", [tops.RANGE_BOUND])
        self.assertIn("1..4", replacements(recs))

    def test_ignores_open_ranges(self):
        """`&b[..]` has no endpoints to be wrong about."""
        self.assertEqual(gen("fn f(b: &[u8]) -> &[u8] { &b[..] }", [tops.RANGE_BOUND]), [])


class TestArgTransposition(unittest.TestCase):
    def test_swaps_two_adjacent_arguments(self):
        recs = gen("fn f() { g(p, q); }", [tops.ARG_TRANSPOSITION])
        self.assertEqual(replacements(recs), ["(q, p)"])

    def test_swaps_each_adjacent_pair_of_three(self):
        recs = gen("fn f() { g(a, b, c); }", [tops.ARG_TRANSPOSITION])
        self.assertEqual(replacements(recs), ["(a, c, b)", "(b, a, c)"])

    def test_ignores_single_argument_calls(self):
        self.assertEqual(gen("fn f() { g(a); }", [tops.ARG_TRANSPOSITION]), [])

    def test_ignores_empty_calls(self):
        self.assertEqual(gen("fn f() { g(); }", [tops.ARG_TRANSPOSITION]), [])

    def test_ignores_identical_adjacent_arguments(self):
        self.assertEqual(gen("fn f() { g(x, x); }", [tops.ARG_TRANSPOSITION]), [])

    def test_still_swaps_the_distinct_neighbouring_pair(self):
        recs = gen("fn f() { g(x, x, y); }", [tops.ARG_TRANSPOSITION])
        self.assertEqual(replacements(recs), ["(x, y, x)"])


class TestListerContract(unittest.TestCase):
    """The lister must be substitutable for mutate_shared.list_mutants."""

    def test_signature_matches_list_mutants(self):
        import inspect
        from list_transpile_mutants import list_transpile_mutants
        from mutate_shared import list_mutants
        self.assertEqual(list(inspect.signature(list_transpile_mutants).parameters),
                         list(inspect.signature(list_mutants).parameters))

    def test_run_fast_mutation_accepts_a_lister(self):
        import inspect
        from mutate_fast import run_fast_mutation
        self.assertIn("mutant_lister",
                      inspect.signature(run_fast_mutation).parameters)


if __name__ == "__main__":
    unittest.main()
