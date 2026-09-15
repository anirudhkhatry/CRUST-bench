from additional_datasets.adapters.js_split import find_leaf_cases, split_suite

SUITE = """import { describe, test, expect } from 'vitest';
import { search } from './app/index.js';

describe('Binary Search', () => {
  const sorted = [1, 3, 5];

  describe('search', () => {
    test('finds an element', () => {
      expect(search(sorted, 3)).toBe(1);
    });

    test('misses an absent element', () => {
      expect(search(sorted, 4)).toBe(-1);
    });

    it('handles the empty array', () => {
      expect(search([], 1)).toBe(-1);
    });
  });
});
"""


def test_find_leaf_cases_finds_test_and_it_but_not_describe():
    cases = find_leaf_cases(SUITE)
    assert [c["title"] for c in cases] == [
        "finds an element", "misses an absent element", "handles the empty array"]
    assert [c["kind"] for c in cases] == ["test", "test", "it"]


def test_split_masks_each_half_with_skip_and_keeps_everything_else():
    # The count-sum invariant this test used to assert (live calls + `.skip(` calls >= 3)
    # is constant at exactly 3 regardless of what `_mask` does, because masking only ever
    # rewrites `test(`/`it(` into `test.skip(`/`it.skip(` -- it can never change the sum.
    # The real property -- that each half skips exactly the cases assigned to the OTHER
    # half -- is covered by `test_every_case_lands_in_exactly_one_half` below (extended to
    # check both halves), so it is not duplicated here.
    a, b, assign = split_suite(SUITE, seed="binary_search")
    for half in (a, b):
        # Shared setup survives in both halves -- this is why we mask instead of extract.
        assert "import { search } from './app/index.js';" in half
        assert "const sorted = [1, 3, 5];" in half
        assert "describe('Binary Search'" in half
    assert set(assign.values()) == {"A", "B"}


def test_every_case_lands_in_exactly_one_half():
    # `f"'{title}'" in live and f"'{title}'" in masked` used to stand in for "the case is
    # live in one half and masked in the other", but it is true by construction: `_mask`
    # only ever INSERTS `.skip`, never deletes text, so every title's quoted string is
    # present in both halves regardless of which side it was assigned to. Checking the
    # actual call shape -- `kind('title'` unmasked vs. `kind.skip('title'` masked -- is the
    # assertion that can actually fail, e.g. if `_mask` skipped the wrong case's cases.
    cases = find_leaf_cases(SUITE)
    kind_by_title = {c["title"]: c["kind"] for c in cases}
    a, b, assign = split_suite(SUITE, seed="binary_search")
    for title, side in assign.items():
        live, masked = (a, b) if side == "A" else (b, a)
        kind = kind_by_title[title]
        assert f"{kind}('{title}'" in live
        assert f"{kind}.skip('{title}'" in masked
    # Counting skips: each half skips exactly the cases assigned to the other. Checked for
    # both halves (extended from the original, which only checked `a`) so it also covers
    # the count-sum invariant removed from the previous test.
    n_a = sum(1 for v in assign.values() if v == "A")
    n_b = sum(1 for v in assign.values() if v == "B")
    assert a.count(".skip(") == n_b
    assert b.count(".skip(") == n_a


def test_the_split_is_deterministic_for_a_given_seed():
    assert split_suite(SUITE, seed="x")[2] == split_suite(SUITE, seed="x")[2]


def test_different_seeds_can_produce_different_assignments():
    seeds = [split_suite(SUITE, seed=s)[2] for s in ("a", "b", "c", "d", "e", "f")]
    assert any(s != seeds[0] for s in seeds)


def test_a_single_case_suite_is_reported_as_unsplittable():
    one = "describe('x', () => { test('only', () => { expect(1).toBe(1); }); });"
    a, b, assign = split_suite(one, seed="x")
    assert assign == {}      # caller marks ARM A n/a rather than scoring a corrupt half
    assert a == b == one


def test_duplicate_titles_are_split_independently():
    suite = ("describe('d', () => {\n"
             "  test('dup', () => { expect(1).toBe(1); });\n"
             "  test('other', () => { expect(2).toBe(2); });\n"
             "  test('dup', () => { expect(3).toBe(3); });\n"
             "});\n")
    cases = find_leaf_cases(suite)
    assert len(cases) == 3            # three physical cases, two distinct titles
    a, b, assign = split_suite(suite, seed="x")
    # Every physical case is masked in exactly one half.
    assert a.count(".skip(") + b.count(".skip(") == 3


def test_a_suite_using_test_each_is_unsplittable():
    suite = ("describe('d', () => {\n"
             "  test('plain', () => { expect(1).toBe(1); });\n"
             "  test.each([1, 2])('param %i', (n) => { expect(n).toBeTruthy(); });\n"
             "});\n")
    assert split_suite(suite, seed="x")[2] == {}


def test_a_describe_title_that_looks_like_a_case_is_not_one():
    suite = ("describe(\"outer test('sub') case\", () => {\n"
             "  test('real one', () => { expect(1).toBe(1); });\n"
             "  test('real two', () => { expect(2).toBe(2); });\n"
             "});\n")
    assert [c["title"] for c in find_leaf_cases(suite)] == ['real one', 'real two']


def test_a_title_containing_a_call_shape_does_not_block_the_split():
    # The title's prose contains ", test(" -- a phantom call site for any guard that
    # cannot tell code from string content. The suite is perfectly parseable and must
    # still split.
    suite = ("describe('d', () => {\n"
             "  test('should reject calls like foo(1, test(2))', () => {\n"
             "    expect(1).toBe(1);\n"
             "  });\n"
             "  test('second case', () => {\n"
             "    expect(2).toBe(2);\n"
             "  });\n"
             "});\n")
    assert len(find_leaf_cases(suite)) == 2
    a, b, assign = split_suite(suite, seed="x")
    assert assign, "a parseable two-case suite must not be reported unsplittable"
    assert a.count(".skip(") + b.count(".skip(") == 2


def test_a_commented_out_case_does_not_block_the_split():
    suite = ("describe('d', () => {\n"
             "  // test('disabled', () => { expect(0).toBe(0); });\n"
             "  test('one', () => { expect(1).toBe(1); });\n"
             "  test('two', () => { expect(2).toBe(2); });\n"
             "});\n")
    assert len(find_leaf_cases(suite)) == 2
    assert split_suite(suite, seed="x")[2], "a commented-out case must not block the split"
