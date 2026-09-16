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

# A leaf case opens with `test(` or `it(` followed by a quoted title, at a statement position
# (after newline, semicolon, brace, paren, or start of string). Anchoring prevents matching
# inside string literals like describe("test('title')"). The pattern cannot parse `test.each(`
# (next char is `.`) or multi-line template-literal titles (no DOTALL). If loose and strict
# patterns disagree, we have an unsupported form and refuse to split.
CASE_RE = re.compile(
    r"""(?:^|[;{}()\n,])\s*(?P<kind>test|it)\(\s*(?P<q>['"`])(?P<title>(?:\\.|(?!(?P=q)).)*)(?P=q)""",
    re.M)

# A looser pattern that matches every leaf-call SHAPE, including chained forms
# (`test.each(`, `it.only(`) and titles the strict pattern cannot parse. If the loose
# and strict patterns disagree on how many call sites exist, some form is present that
# we cannot mask correctly -- and a case we fail to mask stays live in BOTH halves,
# silently breaking disjointness. Refuse to split rather than guess.
LOOSE_RE = re.compile(r"(?:^|[;{}()\n,])\s*(?:test|it)\s*(?:\.\s*\w+\s*)?\(", re.M)


def find_leaf_cases(text):
    """Every leaf `test`/`it` call at a statement position, in source order.

    Uses the named group positions for start/end, which correctly identify the callee token
    regardless of what anchor precedes it.
    """
    return [{"title": m.group("title"), "kind": m.group("kind"),
             "start": m.start("kind"), "end": m.start("kind") + len(m.group("kind"))}
            for m in CASE_RE.finditer(text)]


def _side(seed, index):
    """Alternate, with the starting parity seeded so neither half is always the earlier
    tests across the corpus."""
    off = int(hashlib.sha256(seed.encode()).hexdigest(), 16) & 1
    return "A" if (index + off) % 2 == 0 else "B"


def _code_only(text):
    """`text` with the CONTENTS of strings, template literals and comments blanked.

    Length and newlines are preserved so anchors and offsets behave identically. Only
    LOOSE_RE runs over this: it has no way to tell code from prose, and a title containing
    `, test(` would otherwise manufacture a phantom call site and declare a perfectly
    parseable suite unsplittable. CASE_RE still runs over the original text -- it has to
    read the titles, and its statement anchor already rejects in-string matches.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"`":
            quote, j = ch, i + 1
            while j < n and text[j] != quote:
                if text[j] == "\\":
                    j += 1
                if j < n and text[j] != "\n":
                    out[j] = " "
                j += 1
            i = j + 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = i
            while j < n and text[j] != "\n":
                out[j] = " "
                j += 1
            i = j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = i
            while j < n and not (text[j] == "*" and j + 1 < n and text[j + 1] == "/"):
                if text[j] != "\n":
                    out[j] = " "
                j += 1
            i = min(j + 2, n)
            continue
        i += 1
    return "".join(out)


def _confidently_complete(text, cases):
    """True iff the strict pattern found all the call sites the loose pattern sees.

    If they disagree, the strict pattern cannot parse some form (test.each, multi-line
    template literals, etc.) and masking would fail silently, leaving some cases live
    in both halves. We run LOOSE_RE over code-only text (strings/comments blanked) to
    avoid phantom matches from prose containing anchor chars followed by "test(", while
    CASE_RE runs over the original text so it can read actual titles.
    """
    return len(LOOSE_RE.findall(_code_only(text))) == len(cases)


def _mask(text, drop):
    """Rewrite the given cases' callee token to `<kind>.skip`, right to left.

    Right-to-left so inserting `.skip` never invalidates the offsets of cases still to
    be processed.
    """
    out = text
    for c in sorted(drop, key=lambda c: -c["start"]):
        out = out[:c["end"]] + ".skip" + out[c["end"]:]
    return out


def cap_cases(text, k):
    """(capped_text, n_kept) -- keep the first `k` leaf cases, skip-mask the surplus.

    `(None, 0)` means REFUSED, and the caller must use `text` uncapped: either the suite is
    not confidently parseable (`test.each`, a multi-line template title, an existing
    `.skip` -- masking would then miss cases and the cap would be a lie about how many
    tests ran), or `k` is not a positive target. Refusal is deliberately distinguishable
    from a successful no-op cap, which returns the text itself.

    Capping MASKS rather than cuts, for the reason in this module's docstring: the surplus
    cases' `test(`/`it(` becomes `test.skip(`/`it.skip(`, so imports, `beforeEach` and
    shared fixture constants still exist for the cases that remain live. A masked case can
    nonetheless have been load-bearing for a neighbour (it populated a shared mutable
    fixture), so a caller matching two suites by case count must RE-RUN the capped text and
    check it still admits `k` cases rather than trusting `n_kept`.

    `k >= len(cases)` is a no-op returning the original text unchanged -- capping to at
    least what is present must not rewrite anything.
    """
    cases = find_leaf_cases(text)
    if k <= 0 or not _confidently_complete(text, cases):
        return None, 0
    if k >= len(cases):
        return text, len(cases)
    return _mask(text, cases[k:]), k


def cap_cases_files(files, k):
    """(capped_files, n_kept) -- keep the first `k` leaf cases ACROSS an ordered file list.

    The multi-file analogue of `cap_cases`, and the unit is still the leaf case, not the
    file: the budget is spent walking the files in order, so a first file holding 30 cases
    consumes the whole cap and the remaining files are fully masked. That matches how
    `run_suites` numbers cases (file index, then source order), so "the first k cases" means
    the same thing here and there.

    Files are kept in place even when every one of their cases is masked -- dropping them
    would renumber the survivors and invalidate the admissible set. `(None, 0)` means
    REFUSED, on the same terms as `cap_cases`: if any single file is not confidently
    parseable the whole cap is a lie about how many tests ran, so none of it is applied.
    """
    if k <= 0:
        return None, 0
    out, budget = [], k
    for text in files:
        cases = find_leaf_cases(text)
        if not _confidently_complete(text, cases):
            return None, 0
        keep = min(budget, len(cases))
        out.append(text if keep == len(cases) else _mask(text, cases[keep:]))
        budget -= keep
    return out, k - budget


def split_suite(text, seed):
    """(half_a, half_b, {title: "A"|"B"}). An unsplittable suite returns (text, text, {})."""
    cases = find_leaf_cases(text)
    if len(cases) < 2 or not _confidently_complete(text, cases):
        return text, text, {}
    sides = [_side(seed, i) for i in range(len(cases))]
    if not ({"A", "B"} <= set(sides)):
        return text, text, {}
    a_cases = [c for c, s in zip(cases, sides) if s == "A"]
    b_cases = [c for c, s in zip(cases, sides) if s == "B"]
    assignment = {c["title"]: s for c, s in zip(cases, sides)}
    return _mask(text, b_cases), _mask(text, a_cases), assignment
