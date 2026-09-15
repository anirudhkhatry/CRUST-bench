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
    a, b, assign = split_suite(SUITE, seed="binary_search")
    for half in (a, b):
        # Shared setup survives in both halves -- this is why we mask instead of extract.
        assert "import { search } from './app/index.js';" in half
        assert "const sorted = [1, 3, 5];" in half
        assert "describe('Binary Search'" in half
        assert half.count("test(") + half.count("it(") + half.count(".skip(") >= 3
    assert set(assign.values()) == {"A", "B"}


def test_every_case_lands_in_exactly_one_half():
    a, b, assign = split_suite(SUITE, seed="binary_search")
    for title, side in assign.items():
        live, masked = (a, b) if side == "A" else (b, a)
        assert f"'{title}'" in live and f"'{title}'" in masked
    # Counting skips: each half skips exactly the cases assigned to the other.
    n_b = sum(1 for v in assign.values() if v == "B")
    assert a.count(".skip(") == n_b


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
