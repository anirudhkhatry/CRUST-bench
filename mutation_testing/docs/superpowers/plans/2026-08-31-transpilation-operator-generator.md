# Transpilation Operator Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `transpilation_operators.py`, a standalone generator that emits
cargo-mutants-shaped mutant records for eleven C-to-Rust transpilation bug classes, and
wire it into the existing mutation harness behind one parameter.

**Architecture:** A tree-sitter query plus a rewrite callable defines each operator; a
registry of frozen `Operator` dataclasses makes adding a twelfth operator a one-object
change. `generate()` returns dicts matching `_MUTANT_KEYS` from `mutate_shared.py:52`, so
every downstream stage — build, viability, verdict, `outcomes.json`, analysis — works
unchanged. Every emitted record is round-trip-verified against `mutate_shared.splice`
before it leaves the generator.

**Tech Stack:** Python 3.12 (`/nas/miniconda3/envs/repair_rust/bin/python`), `tree_sitter`
0.25.2, `tree_sitter_rust` 0.24.0, stdlib `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-transpilation-mutation-operators-design.md`

## Scope note: this is plan 1 of 3

The spec covers three independently testable subsystems. This plan implements only the
first; each later one gets its own plan, written after its predecessor lands.

1. **Generator** (this plan) — operators, records, harness integration. Ships working,
   testable software on its own: a run that scores transpilation mutants across the four
   settings.
2. **Witness pipeline** — instrumenting `ground_truth_mutation.py` to keep the failing-test
   corpus instead of deleting it, then the inverse-repair search that decides which
   operators stay in the catalog. Depends on the `inverse` callables this plan defines but
   leaves unimplemented.
3. **Study** — sampling, equivalence quarantine, the pilot run, the analysis.

Splitting here matters: the witness gate can *delete* operators, so building the study on
top of an unwitnessed catalog would mean rerunning it.

## Global Constraints

- Python interpreter is `/nas/miniconda3/envs/repair_rust/bin/python`. Plain `python` is
  `/opt/conda/bin/python` and lacks `tree_sitter`.
- `pytest` is **not** installed and this plan does not install it. Tests use stdlib
  `unittest`, run from `mutation_testing/` with
  `python -m unittest discover -s tests -t . -v`.
- Never import `run_mutation` from `transpilation_operators.py` or from any test. It pulls
  in the whole LLM pipeline and requires `OPENAI_API_KEY`. `mutate_shared` imports cleanly
  and is the only harness import allowed.
- Mutant records must contain exactly these keys, matching `mutate_shared.py:52`:
  `name`, `package`, `file`, `function`, `span`, `replacement`, `genre`.
- `span` is `{"start": {"line", "column"}, "end": {"line", "column"}}`, both 1-based, and
  `column` is a **character** offset into a Python `str` line, not a byte offset — see
  `_offset` at `mutate_shared.py:55`.
- Every `genre` starts with `Transpile`.
- Operator names are kebab-case and match the spec's catalog exactly: `cast-width`,
  `cast-sign`, `overflow-discipline`, `shift-discipline`, `endianness`, `text-unit`,
  `float-width`, `lost-in-place-update`, `arg-transposition`, `error-path-erasure`,
  `range-bound`.
- Do not modify `mutate_shared.py`. The only harness edit in this plan is one added
  parameter in `mutate_fast.py`.
- Commit after every task.

## File Structure

| file | responsibility |
|---|---|
| `transpilation_operators.py` (create) | Parser plumbing, `Operator` dataclass, span conversion, round-trip verification, `generate()`, and the `OPERATORS` registry. No CRUST-bench paths. |
| `tests/__init__.py` (create) | Makes `tests` a package so `unittest discover -t .` can import `transpilation_operators`. |
| `tests/test_transpilation_operators.py` (create) | Unit tests per operator over string fixtures, plus the round-trip property test. |
| `tests/test_disjointness.py` (create) | Asserts no generated record collides with the 1101 stored cargo-mutants records. |
| `mutate_fast.py` (modify) | One new `mutant_lister` parameter and one CLI flag. |

`transpilation_operators.py` is one file on purpose: the operators are small, they share
the span and record machinery, and splitting eleven ~15-line callables across modules would
cost more in indirection than it buys. If it passes ~600 lines, split the registry into
`transpilation_operators_registry.py` and keep the machinery in place.

---

### Task 1: Generator machinery and the first operator

**Files:**
- Create: `transpilation_operators.py`
- Create: `tests/__init__.py`
- Test: `tests/test_transpilation_operators.py`

**Interfaces:**
- Consumes: `mutate_shared.splice(text, span, replacement)`.
- Produces: `Operator` dataclass; `generate(text, rel_path, package, operators=None)
  -> list[dict]` (a `None` default resolves to `OPERATORS`, so the registry can grow after
  this task without stale default bindings); `OPERATORS: list[Operator]`; helpers
  `_span(text, src, node)`,
  `_enclosing_fn(node, src)`, `_genre(name)`, `_record(...)`; constant
  `_METHOD_OR_PATH` (a tree-sitter query string capturing `@m` on both method names and
  path-qualified function names); operator `ENDIANNESS`.

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as an empty file, then `tests/test_transpilation_operators.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `mutation_testing/`:

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transpilation_operators'`.

- [ ] **Step 3: Write the implementation**

Create `transpilation_operators.py`:

```python
"""Mutation operators that model C-to-Rust transpilation bugs.

cargo-mutants' operators are language-agnostic bug shapes -- swap an operator, blank a
body, delete a match arm. They say nothing about *translation*. The operators here model
the specific places where C and Rust disagree, and where a model translating between them
can therefore go wrong: integer width and overflow discipline, byte order, text units,
and in-place update lost to a borrow-checker workaround.

The module deliberately knows nothing about CRUST-bench. It takes source text and returns
records shaped like cargo-mutants' own (``_MUTANT_KEYS`` in mutate_shared), which is what
lets every downstream stage of the harness consume them unchanged.

Design note on precision: tree-sitter gives syntax, not types, so some generated mutants
will not compile. That is acceptable rather than sloppy -- the harness already treats a
failed build as ``Unviable`` and drops it from the denominator, so imprecision costs build
time, not correctness. It is still worth tuning queries, because build time is ~16-19 s per
mutant.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import tree_sitter as ts
import tree_sitter_rust as tsr

from mutate_shared import splice

_LANG = ts.Language(tsr.language())
_PARSER = ts.Parser(_LANG)

# Captures @m on both `x.method()` and `Type::function()`. Several operators mutate a
# callee's *identity* rather than its arguments, and Rust spells those two forms
# differently, so both patterns live in one query.
_METHOD_OR_PATH = (
    "(call_expression function: (field_expression field: (field_identifier) @m))\n"
    "(scoped_identifier name: (identifier) @m)"
)


@dataclass(frozen=True)
class Operator:
    """One mutation operator.

    ``rewrite`` receives a captured node and the source bytes, and yields
    ``(target_node, replacement_text, label)`` triples. Yielding nothing means the capture
    did not actually match this operator -- queries are deliberately broad and the rewrite
    does the precise filtering.

    ``inverse`` is unused by the generator. It exists for the witness pipeline (plan 2),
    which applies an operator backwards to a broken translation to prove the operator
    models a real bug.
    """

    name: str
    family: str
    query: str
    rewrite: Callable
    inverse: Optional[Callable] = None


def _text(node, src):
    return src[node.start_byte:node.end_byte].decode("utf-8")


def _char_index(src, byte_offset):
    """Byte offset -> character index. These differ the moment the file is not ASCII."""
    return len(src[:byte_offset].decode("utf-8"))


def _position(text, char_idx):
    line = text.count("\n", 0, char_idx) + 1
    column = char_idx - (text.rfind("\n", 0, char_idx) + 1) + 1
    return {"line": line, "column": column}


def _span(text, src, node):
    """(span, start_char, end_char) for a node, in the harness's line/column encoding."""
    start = _char_index(src, node.start_byte)
    end = _char_index(src, node.end_byte)
    return {"start": _position(text, start), "end": _position(text, end)}, start, end


def _enclosing_fn(node, src):
    """The function a node sits in. Mirrors cargo-mutants' `function` field."""
    n = node
    while n is not None and n.type != "function_item":
        n = n.parent
    if n is None:
        return {"function_name": "<unknown>", "return_type": ""}
    name = n.child_by_field_name("name")
    ret = n.child_by_field_name("return_type")
    return {"function_name": _text(name, src) if name else "<unknown>",
            "return_type": _text(ret, src) if ret else ""}


def _genre(op_name):
    return "Transpile" + "".join(part.capitalize() for part in op_name.split("-"))


def _record(text, src, rel_path, package, op, node, replacement, label):
    """Build one record, or None if it cannot be verified.

    The verification is the point: tree-sitter counts bytes and ``_offset`` counts
    characters, so a non-ASCII file could otherwise produce a span that splices somewhere
    other than where the operator looked. A record that cannot reproduce its own edit is
    dropped rather than emitted.
    """
    span, start, end = _span(text, src, node)
    if splice(text, span, replacement) != text[:start] + replacement + text[end:]:
        return None
    pos = span["start"]
    return {
        "name": f"{rel_path}:{pos['line']}:{pos['column']}: {label}",
        "package": package,
        "file": rel_path,
        "function": _enclosing_fn(node, src),
        "span": span,
        "replacement": replacement,
        "genre": _genre(op.name),
    }


def generate(text, rel_path, package, operators=None):
    """Every transpilation-aware mutant for one Rust source file.

    Returns cargo-mutants-shaped records -- name, package, file, function, span,
    replacement, genre -- so everything downstream of list_mutants works unchanged.
    """
    operators = OPERATORS if operators is None else operators
    src = text.encode("utf-8")
    tree = _PARSER.parse(src)

    seen, records = set(), []
    for op in operators:
        cursor = ts.QueryCursor(ts.Query(_LANG, op.query))
        for nodes in cursor.captures(tree.root_node).values():
            for node in nodes:
                for target, replacement, label in op.rewrite(node, src):
                    rec = _record(text, src, rel_path, package, op, target,
                                  replacement, label)
                    if rec is None:
                        continue
                    # A node can be captured by more than one pattern in a query; identical
                    # edits are the same mutant and must not be scored twice.
                    key = (rec["file"], rec["span"]["start"]["line"],
                           rec["span"]["start"]["column"], rec["span"]["end"]["line"],
                           rec["span"]["end"]["column"], rec["replacement"])
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(rec)

    records.sort(key=lambda r: (r["file"], r["span"]["start"]["line"],
                                r["span"]["start"]["column"], r["replacement"]))
    return records


# --------------------------------------------------------------------------------------
# F2. Data representation
# --------------------------------------------------------------------------------------

# C hand-rolls byte packing with shifts and masks; Rust names the byte order, so the name
# can simply be the wrong one. Grouped so a swap always stays within a compatible signature.
_ENDIAN_GROUPS = [
    ("to_be_bytes", "to_le_bytes", "to_ne_bytes"),
    ("from_be_bytes", "from_le_bytes", "from_ne_bytes"),
    ("to_be", "to_le"),
    ("from_be", "from_le"),
]


def _rewrite_endianness(node, src):
    name = _text(node, src)
    for group in _ENDIAN_GROUPS:
        if name in group:
            for other in group:
                if other != name:
                    yield node, other, f"replace {name} with {other}"
            return


ENDIANNESS = Operator(
    name="endianness",
    family="F2-representation",
    query=_METHOD_OR_PATH,
    rewrite=_rewrite_endianness,
)


OPERATORS = [ENDIANNESS]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench
git add mutation_testing/transpilation_operators.py mutation_testing/tests/
git commit -m "feat(mutation): generator machinery and the endianness operator"
```

---

### Task 2: Family F1 — integer semantics

**Files:**
- Modify: `transpilation_operators.py` (append operators, extend `OPERATORS`)
- Test: `tests/test_transpilation_operators.py` (append test classes)

**Interfaces:**
- Consumes: `Operator`, `_METHOD_OR_PATH`, `_text`, `generate` from Task 1.
- Produces: operators `CAST_WIDTH`, `CAST_SIGN`, `OVERFLOW_DISCIPLINE`,
  `SHIFT_DISCIPLINE`; constants `_INT_LADDER: dict[str, list[str]]`,
  `_WRAP_FAMILIES: dict[str, str]`, `_SHIFT_OPS: dict[str, str]`.

This is the highest-yield family: `as`-casts alone have 2228 sites across 84 of 101
projects, and `wrapping_*` appears in 51.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transpilation_operators.py`, above the `if __name__` block:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: FAIL with `AttributeError: module 'transpilation_operators' has no attribute
'CAST_WIDTH'`.

- [ ] **Step 3: Write the implementation**

Append to `transpilation_operators.py`, immediately before the `OPERATORS = [...]` line:

```python
# --------------------------------------------------------------------------------------
# F1. Integer semantics
# --------------------------------------------------------------------------------------

# C's int/long/size_t map onto Rust's fixed widths by a judgement call the translator has
# to make, and it can pick the neighbour. Mutating one rung along the ladder models that
# directly; jumping further would be a less plausible bug.
_INT_LADDER = {
    "u8": ["u16"], "u16": ["u8", "u32"], "u32": ["u16", "u64"],
    "u64": ["u32", "u128"], "u128": ["u64"], "usize": ["u32", "u64"],
    "i8": ["i16"], "i16": ["i8", "i32"], "i32": ["i16", "i64"],
    "i64": ["i32", "i128"], "i128": ["i64"], "isize": ["i32", "i64"],
}

# C `char` signedness is platform-defined, and sign decides whether `>>` is arithmetic or
# logical. usize/isize are excluded: they are chosen for indexing, not for sign, so a flip
# there models nothing real.
_SIGN_FLIP = {
    "u8": "i8", "u16": "i16", "u32": "i32", "u64": "i64", "u128": "i128",
    "i8": "u8", "i16": "u16", "i32": "u32", "i64": "u64", "i128": "u128",
}

_CAST_QUERY = "(type_cast_expression type: (primitive_type) @ty)"

# Arithmetic only. Shifts are handled by shift-discipline so the two cannot double-count
# the same site.
_WRAP_FAMILIES = {"add": "+", "sub": "-", "mul": "*", "div": "/", "rem": "%"}
_SHIFT_OPS = {"<<": "shl", ">>": "shr"}


def _rewrite_cast_width(node, src):
    name = _text(node, src)
    for other in _INT_LADDER.get(name, []):
        yield node, other, f"replace cast {name} with {other}"


def _rewrite_cast_sign(node, src):
    name = _text(node, src)
    if name in _SIGN_FLIP:
        other = _SIGN_FLIP[name]
        yield node, other, f"replace cast {name} with {other}"


CAST_WIDTH = Operator(name="cast-width", family="F1-integer",
                      query=_CAST_QUERY, rewrite=_rewrite_cast_width)

CAST_SIGN = Operator(name="cast-sign", family="F1-integer",
                     query=_CAST_QUERY, rewrite=_rewrite_cast_sign)


def _binary_parts(node, src):
    """(left, op_token, right) for a binary_expression, or None if it is not one.

    The grammar gives the operator as an unnamed child, so it is found positionally
    rather than by field name.
    """
    if node.type != "binary_expression" or len(node.children) != 3:
        return None
    left, op, right = node.children
    if op.is_named:
        return None
    return _text(left, src), _text(op, src), _text(right, src)


def _rewrite_overflow(node, src):
    # Direction 1: a named wrapping/saturating call. Swap the discipline, or drop it for
    # the bare operator, which is what a translator does when it forgets C wraps.
    if node.type == "field_identifier" or node.type == "identifier":
        name = _text(node, src)
        for prefix, other in (("wrapping_", "saturating_"), ("saturating_", "wrapping_")):
            if name.startswith(prefix) and name[len(prefix):] in _WRAP_FAMILIES:
                stem = name[len(prefix):]
                yield node, other + stem, f"replace {name} with {other}{stem}"
                call = node.parent.parent  # field_identifier -> field_expression -> call
                if call is not None and call.type == "call_expression":
                    recv = call.child_by_field_name("function").child_by_field_name("value")
                    args = call.child_by_field_name("arguments")
                    inner = [c for c in args.children if c.is_named] if args else []
                    if recv is not None and len(inner) == 1:
                        expr = (f"{_text(recv, src)} {_WRAP_FAMILIES[stem]} "
                                f"{_text(inner[0], src)}")
                        yield call, expr, f"replace {name} with bare {_WRAP_FAMILIES[stem]}"
                return
        return

    # Direction 2: a bare arithmetic operator that could have needed wrapping.
    parts = _binary_parts(node, src)
    if parts is None:
        return
    left, op, right = parts
    for stem, symbol in _WRAP_FAMILIES.items():
        if op == symbol:
            yield node, f"{left}.wrapping_{stem}({right})", \
                f"replace {op} with wrapping_{stem}"
            return


OVERFLOW_DISCIPLINE = Operator(
    name="overflow-discipline",
    family="F1-integer",
    query=_METHOD_OR_PATH + "\n(binary_expression) @b",
    rewrite=_rewrite_overflow,
)


def _rewrite_shift(node, src):
    parts = _binary_parts(node, src)
    if parts is None:
        return
    left, op, right = parts
    if op in _SHIFT_OPS:
        stem = _SHIFT_OPS[op]
        yield node, f"{left}.wrapping_{stem}({right})", \
            f"replace {op} with wrapping_{stem}"


SHIFT_DISCIPLINE = Operator(name="shift-discipline", family="F1-integer",
                            query="(binary_expression) @b", rewrite=_rewrite_shift)
```

Then replace the registry line with:

```python
OPERATORS = [
    ENDIANNESS,
    CAST_WIDTH, CAST_SIGN, OVERFLOW_DISCIPLINE, SHIFT_DISCIPLINE,
]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: PASS. If `test_drops_wrapping_for_the_bare_operator` fails, the cause is almost
certainly the `node.parent.parent` walk — print `call.type` to confirm the capture is a
`field_identifier` inside a `call_expression` and adjust the walk, do not weaken the test.

- [ ] **Step 5: Commit**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench
git add mutation_testing/transpilation_operators.py mutation_testing/tests/
git commit -m "feat(mutation): F1 integer-semantics operators (cast width/sign, overflow, shift)"
```

---

### Task 3: Family F2 — remaining representation operators

**Files:**
- Modify: `transpilation_operators.py`
- Test: `tests/test_transpilation_operators.py`

**Interfaces:**
- Consumes: `Operator`, `_METHOD_OR_PATH`, `_CAST_QUERY`, `_text`.
- Produces: operators `TEXT_UNIT`, `FLOAT_WIDTH`; constant `_FLOAT_FLIP: dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transpilation_operators.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: FAIL with `AttributeError: ... has no attribute 'TEXT_UNIT'`.

- [ ] **Step 3: Write the implementation**

Append before the registry line in `transpilation_operators.py`:

```python
# C's strlen counts bytes. A translator reaching for `chars()` is right on ASCII and wrong
# on everything else -- exactly the bug a weak suite cannot see.
def _rewrite_text_unit(node, src):
    name = _text(node, src)
    if name in ("chars", "bytes"):
        other = "bytes" if name == "chars" else "chars"
        yield node, other, f"replace {name} with {other}"
        return
    if name == "len":
        call = node.parent.parent
        if call is not None and call.type == "call_expression":
            recv = call.child_by_field_name("function").child_by_field_name("value")
            if recv is not None:
                yield call, f"{_text(recv, src)}.chars().count()", \
                    "replace len with chars().count()"


TEXT_UNIT = Operator(name="text-unit", family="F2-representation",
                     query=_METHOD_OR_PATH, rewrite=_rewrite_text_unit)


_FLOAT_FLIP = {"f32": "f64", "f64": "f32"}


def _rewrite_float_width(node, src):
    name = _text(node, src)
    if name in _FLOAT_FLIP:
        yield node, _FLOAT_FLIP[name], f"replace cast {name} with {_FLOAT_FLIP[name]}"


FLOAT_WIDTH = Operator(name="float-width", family="F2-representation",
                       query=_CAST_QUERY, rewrite=_rewrite_float_width)
```

Update the registry:

```python
OPERATORS = [
    ENDIANNESS, TEXT_UNIT, FLOAT_WIDTH,
    CAST_WIDTH, CAST_SIGN, OVERFLOW_DISCIPLINE, SHIFT_DISCIPLINE,
]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench
git add mutation_testing/transpilation_operators.py mutation_testing/tests/
git commit -m "feat(mutation): F2 representation operators (text unit, float width)"
```

---

### Task 4: Family F3 — ownership and interface

**Files:**
- Modify: `transpilation_operators.py`
- Test: `tests/test_transpilation_operators.py`

**Interfaces:**
- Consumes: `Operator`, `_text`.
- Produces: operators `LOST_IN_PLACE_UPDATE`, `ERROR_PATH_ERASURE`, `RANGE_BOUND`,
  `ARG_TRANSPOSITION`.

These are the operators with no analogue in any existing mutation tool, because they only
exist as bugs when a borrow checker is in the loop.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transpilation_operators.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: FAIL with `AttributeError: ... has no attribute 'LOST_IN_PLACE_UPDATE'`.

- [ ] **Step 3: Write the implementation**

Append before the registry line:

```python
# --------------------------------------------------------------------------------------
# F3. Ownership and interface
# --------------------------------------------------------------------------------------

def _rewrite_lost_update(node, src):
    """`&mut x` -> `&mut x.clone()`.

    When the borrow checker fights a translation, the standard escape is to clone, and the
    caller's update silently vanishes. `&mut x.clone()` borrows a temporary, which Rust
    permits in argument position, so the mutant compiles and does nothing.
    """
    kids = node.children
    if len(kids) != 3 or kids[1].type != "mutable_specifier":
        return
    inner = kids[2]
    text = _text(inner, src)
    # Parenthesise anything that is not a plain path, so `&mut *p` does not become
    # `&mut *p.clone()`, which clones the pointer rather than the pointee.
    if inner.type not in ("identifier", "field_expression", "scoped_identifier"):
        text = f"({text})"
    yield node, f"&mut {text}.clone()", "clone a &mut argument (lost in-place update)"


LOST_IN_PLACE_UPDATE = Operator(name="lost-in-place-update", family="F3-ownership",
                                query="(reference_expression) @r",
                                rewrite=_rewrite_lost_update)


def _rewrite_error_path(node, src):
    """`v?` -> `v.unwrap_or_default()`: C's -1/NULL sentinel becomes a swallowed error."""
    inner = node.children[0] if node.children else None
    if inner is None:
        return
    yield node, f"{_text(inner, src)}.unwrap_or_default()", \
        "replace ? with unwrap_or_default()"


ERROR_PATH_ERASURE = Operator(name="error-path-erasure", family="F3-ownership",
                              query="(try_expression) @t", rewrite=_rewrite_error_path)


def _rewrite_range_bound(node, src):
    """C pointer arithmetic becomes Rust slicing, and the bound slips by one."""
    kids = node.children
    named = [c for c in kids if c.is_named]
    ops = [c for c in kids if not c.is_named and c.type in ("..", "..=")]
    if len(named) != 2 or len(ops) != 1:
        return  # open ranges such as `..` or `a..` have no endpoint to get wrong
    left, right = _text(named[0], src), _text(named[1], src)
    if ops[0].type == "..":
        yield node, f"{left}..={right}", "replace .. with ..="
        yield node, f"{left}..{right} - 1", "shorten range end by one"
    else:
        yield node, f"{left}..{right}", "replace ..= with .."


RANGE_BOUND = Operator(name="range-bound", family="F3-ownership",
                       query="(range_expression) @r", rewrite=_rewrite_range_bound)


def _rewrite_arg_transposition(node, src):
    """Swap adjacent arguments: the C signature order lost in translation.

    tree-sitter has no types, so this cannot check that the two arguments are
    interchangeable. Mis-typed swaps fail to compile and the harness records them
    Unviable, which costs build time but not correctness.
    """
    args = [c for c in node.children if c.is_named]
    if len(args) < 2:
        return
    texts = [_text(a, src) for a in args]
    for i in range(len(texts) - 1):
        swapped = list(texts)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        yield node, "(" + ", ".join(swapped) + ")", \
            f"transpose arguments {i + 1} and {i + 2}"


ARG_TRANSPOSITION = Operator(name="arg-transposition", family="F3-ownership",
                             query="(call_expression arguments: (arguments) @a)",
                             rewrite=_rewrite_arg_transposition)
```

Update the registry to the full catalog:

```python
OPERATORS = [
    ENDIANNESS, TEXT_UNIT, FLOAT_WIDTH,
    CAST_WIDTH, CAST_SIGN, OVERFLOW_DISCIPLINE, SHIFT_DISCIPLINE,
    LOST_IN_PLACE_UPDATE, ERROR_PATH_ERASURE, RANGE_BOUND, ARG_TRANSPOSITION,
]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: PASS, all operator tests plus the round-trip property test.

- [ ] **Step 5: Size `arg-transposition` against the corpus**

The spec left this operator's site count open because it needs the AST. Measure it now:

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench/mutation_testing
/nas/miniconda3/envs/repair_rust/bin/python - <<'PY'
from collections import Counter
from pathlib import Path
import transpilation_operators as tops

root = Path("results/claude/results_opus5_4settings/_frozen_source")
counts, projects = Counter(), {}
for f in root.rglob("*.rs"):
    if "/bin/" in str(f) or "/target/" in str(f):
        continue
    try:
        recs = tops.generate(f.read_text(encoding="utf-8", errors="replace"),
                             str(f), "demo")
    except Exception as e:
        print("skip", f, e)
        continue
    for r in recs:
        counts[r["genre"]] += 1
        projects.setdefault(r["genre"], set()).add(f.relative_to(root).parts[0])
for genre, n in counts.most_common():
    print(f"{genre:34} {n:7} hits  {len(projects[genre]):3} proj")
PY
```

Record the output in the commit message. If `TranspileArgTransposition` appears in fewer
than 10 projects, note it as a drop candidate for the witness gate; do not remove it here.

- [ ] **Step 6: Commit**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench
git add mutation_testing/transpilation_operators.py mutation_testing/tests/
git commit -m "feat(mutation): F3 ownership operators (lost update, error path, range, arg order)"
```

---

### Task 5: Disjointness from cargo-mutants

**Files:**
- Test: `tests/test_disjointness.py` (create)

**Interfaces:**
- Consumes: `generate`, `OPERATORS`.
- Produces: nothing importable; this task is a guarantee, not a component.

The catalog's central claim is that these operators are not the ones cargo-mutants already
has. That claim is checkable against the 1101 mutants already on disk from the completed
five-project run, so it should be a test rather than a paragraph.

**This task is not red-green.** It is a guard on a property the previous tasks were
designed to have, so the honest expectation is that it passes on first run. What it is
protecting against is a *future* operator drifting into cargo-mutants' territory. If it
fails now, that is a real finding about Tasks 2-4, and Step 3 says what to do about it.

- [ ] **Step 1: Write the test**

Create `tests/test_disjointness.py`:

```python
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
```

- [ ] **Step 2: Run the test**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest tests.test_disjointness -v
```

Expected: PASS, 2 tests.

If `test_corpus_is_present` fails, the paths are wrong, not the operators — confirm
`results/claude/opus5_sample/_mutants_out` holds 19 `outcomes.json` files and
`_workspaces/<project>/core/src` exists.

- [ ] **Step 3: If a real overlap is reported, narrow the operator**

Do not relax the assertion. The likely culprit is `overflow-discipline` direction 2 if a
rewrite emits a replacement over the operator token's own span rather than over the whole
`binary_expression` — cargo-mutants owns that exact span, and matching it is the one way
these catalogs can collide. Fix the rewrite to span the full expression and rerun.

Skip this step if Step 2 passed.

- [ ] **Step 4: Run the full suite**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: PASS, all suites.

- [ ] **Step 5: Commit**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench
git add mutation_testing/tests/test_disjointness.py
git commit -m "test(mutation): assert transpilation operators are disjoint from cargo-mutants"
```

---

### Task 6: Harness integration and end-to-end smoke

**Files:**
- Modify: `mutate_fast.py` (`run_fast_mutation` signature and mutant listing; `main`)
- Create: `list_transpile_mutants.py`

**Interfaces:**
- Consumes: `generate`, `OPERATORS`; `mutate_fast.run_fast_mutation`.
- Produces: `list_transpile_mutants(ws, package, env=None, timeout=900) -> list[dict]`,
  signature-compatible with `mutate_shared.list_mutants`; `run_fast_mutation(...,
  mutant_lister=list_mutants)`; CLI flag `--mutants {cargo,transpile,both}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transpilation_operators.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'list_transpile_mutants'`.

- [ ] **Step 3: Write the lister**

Create `list_transpile_mutants.py`:

```python
"""Adapt the transpilation operators to mutate_shared.list_mutants' signature.

Keeping this separate from transpilation_operators.py preserves that module's independence
from CRUST-bench layout: this file knows a workspace has a `core/` package, the generator
does not.
"""

from pathlib import Path

from transpilation_operators import OPERATORS, generate


def list_transpile_mutants(ws, package, env=None, timeout=900):
    """Every transpilation mutant for ``package``'s sources under ``ws``.

    ``env`` and ``timeout`` are accepted and ignored: no subprocess runs here. They exist
    so this is a drop-in for mutate_shared.list_mutants.
    """
    core = Path(ws) / "core"
    records = []
    for src in sorted((core / "src").rglob("*.rs")):
        rel_to_src = src.relative_to(core / "src")
        if "bin" in rel_to_src.parts:
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        records.extend(generate(text, "core/" + str(src.relative_to(core)),
                                package, operators=OPERATORS))
    return records
```

- [ ] **Step 4: Wire the lister into the harness**

In `mutate_fast.py`, add `mutant_lister=None` to the end of the `run_fast_mutation`
keyword arguments (alongside `verbose=True`), and replace the listing line at
`mutate_fast.py:199`:

```python
    mutants = (mutant_lister or list_mutants)(ws, package, env=env0)
```

Add to the module docstring, under the existing notes:

```
``mutant_lister`` selects where mutants come from. The default is cargo-mutants, which is
the validated path and is byte-identical to previous runs. Passing
``list_transpile_mutants.list_transpile_mutants`` scores the transpilation-aware catalog
instead; passing a lambda that concatenates both scores them together, in which case the
``Transpile*`` genre prefix keeps the two families attributable in the analysis.
```

In `main()`, add the CLI flag and resolve it:

```python
    p.add_argument("--mutants", choices=["cargo", "transpile", "both"], default="cargo",
                   help="which operator set to score (default: cargo-mutants, the "
                        "validated path)")
```

and, where `run_fast_mutation` is called, pass:

```python
    from list_transpile_mutants import list_transpile_mutants
    listers = {
        "cargo": None,
        "transpile": list_transpile_mutants,
        "both": lambda ws, package, env=None, timeout=900: (
            list_mutants(ws, package, env=env, timeout=timeout)
            + list_transpile_mutants(ws, package, env=env, timeout=timeout)),
    }
    ...
    mutant_lister=listers[args.mutants],
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/nas/miniconda3/envs/repair_rust/bin/python -m unittest discover -s tests -t . -v
```

Expected: PASS.

- [ ] **Step 6: Verify the default path is unchanged**

The whole point of the parameter is that the validated run does not move. Confirm the
signature default resolves to cargo-mutants:

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench/mutation_testing
/nas/miniconda3/envs/repair_rust/bin/python -c "
import inspect, mutate_fast
p = inspect.signature(mutate_fast.run_fast_mutation).parameters['mutant_lister']
print('default is None (cargo-mutants):', p.default is None)
"
```

Expected: `default is None (cargo-mutants): True`.

- [ ] **Step 7: End-to-end smoke on one shard**

Verify a transpilation mutant survives build, verdict, and `outcomes.json` unchanged. This
builds real Rust and takes a few minutes; run it in the foreground and watch the tqdm bar.

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench/mutation_testing
OPENAI_API_KEY=dummy /nas/miniconda3/envs/repair_rust/bin/python mutate_fast.py ulidgen \
  --mutants transpile --shard 1/40 --workers 4 --jobs 4 --min-test-timeout 20
```

Expected: the run completes; `outcomes.json` is written; every `genre` in it starts with
`Transpile`. Check:

```bash
/nas/miniconda3/envs/repair_rust/bin/python -c "
import json, glob
from collections import Counter
for f in glob.glob('results/claude/opus5_sample/_mutants_out/*/ulidgen/mutants.out/outcomes.json'):
    o = json.load(open(f))['outcomes']
    print(f.split('/')[-4], Counter(x['scenario']['Mutant']['genre'] for x in o))
"
```

- [ ] **Step 8: Verify the workspace is clean**

A run that dies without unwinding leaves a mutant spliced into `core/`, and this has
happened twice in this project. Always check:

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench/mutation_testing/results/claude/opus5_sample
diff -r --exclude=bin _frozen_source/ulidgen/src _workspaces/ulidgen/core/src && echo CLEAN
```

Expected: `CLEAN`. If not, restore with
`python -c "import run_opus5_sample as r; print(r.restore('ulidgen'))"`.

- [ ] **Step 9: Commit**

```bash
cd /nas/CRUST-bench-COLM/CRUST-bench
git add mutation_testing/list_transpile_mutants.py mutation_testing/mutate_fast.py \
        mutation_testing/tests/
git commit -m "feat(mutation): score transpilation operators via mutate_fast --mutants"
```

---

## Definition of done

- All suites pass: `python -m unittest discover -s tests -t . -v`.
- Eleven operators in `OPERATORS`, each with at least one positive and one negative test.
- Every generated record round-trips through `mutate_shared.splice`.
- Zero edit collisions with the 1101 stored cargo-mutants records.
- `mutate_fast.py --mutants cargo` is byte-identical in behaviour to before this plan.
- A shard of `ulidgen` scored end to end with `--mutants transpile`, and the workspace
  verified clean afterwards.
- `arg-transposition`'s corpus site count measured and recorded.

## Not in this plan

- The witness pipeline (plan 2). The `inverse` field exists on `Operator` and stays `None`
  throughout; no operator is added to or removed from the catalog on evidence yet.
- Sampling caps, the equivalence quarantine, and the pilot study (plan 3).
- Any change to `mutate_shared.py`.
