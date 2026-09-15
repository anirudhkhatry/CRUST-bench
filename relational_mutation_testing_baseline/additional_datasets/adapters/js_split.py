"""Split a vitest suite into two disjoint halves without cutting anything out.

Baseline A and the hidden oracle would otherwise be the same file, which makes Baseline A
a tautology (spec §4.4). Measuring the pilot corpus killed the obvious split: 16 of the 18
pure-logic fixtures have exactly one top-level `describe`, so the only granularity that
exists in quantity is the leaf `test`/`it` case.

So both halves are the *whole* original file, with the other half's leaf calls rewritten
`test(` -> `test.skip(`. Nothing is extracted, so imports, nested describes, `beforeEach`
and shared fixture constants -- `const sorted = [1, 3, 5]` in the outer describe body --
survive in both halves rather than having to be re-derived, which is where a text-cutting
split would quietly go wrong.
"""
import hashlib
import re

# A leaf case opens with `test(` or `it(` followed by a quoted title. `describe(` is
# deliberately absent, and `.skip`/`.only`/`.each` are excluded by the `(` right after
# the token, so an already-masked file is never double-masked.
CASE_RE = re.compile(r"""\b(?P<kind>test|it)\(\s*(?P<q>['"`])(?P<title>(?:\\.|(?!(?P=q)).)*)(?P=q)""")


def find_leaf_cases(text):
    """Every leaf `test`/`it` call, in source order."""
    return [{"title": m.group("title"), "kind": m.group("kind"),
             "start": m.start("kind"), "end": m.start("kind") + len(m.group("kind"))}
            for m in CASE_RE.finditer(text)]


def _side(seed, index):
    """Alternate, with the starting parity seeded so neither half is always the earlier
    tests across the corpus."""
    off = int(hashlib.sha256(seed.encode()).hexdigest(), 16) & 1
    return "A" if (index + off) % 2 == 0 else "B"


def _mask(text, cases, drop):
    """Rewrite the named cases' callee token to `<kind>.skip`, right to left."""
    out = text
    for c in sorted(cases, key=lambda c: -c["start"]):
        if c["title"] in drop:
            out = out[:c["end"]] + ".skip" + out[c["end"]:]
    return out


def split_suite(text, seed):
    """(half_a, half_b, {title: "A"|"B"}). An unsplittable suite returns ({}, text, text)."""
    cases = find_leaf_cases(text)
    if len(cases) < 2:
        return text, text, {}
    assignment = {}
    for i, c in enumerate(cases):
        assignment.setdefault(c["title"], _side(seed, i))
    a_titles = {t for t, s in assignment.items() if s == "A"}
    b_titles = {t for t, s in assignment.items() if s == "B"}
    if not a_titles or not b_titles:          # degenerate after de-duplicating titles
        return text, text, {}
    return _mask(text, cases, b_titles), _mask(text, cases, a_titles), assignment
