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

    Two independent checks guard against a silently wrong span, which matters because
    tree-sitter counts bytes and ``mutate_shared._offset`` counts characters, so a
    non-ASCII file could otherwise produce a span that splices somewhere other than where
    the operator looked:

    * The first check catches a mis-derived byte->char conversion: it compares the text
      sliced by the computed character offsets against the node's own text, independently
      of ``splice``. ``_position``/``_offset`` are exact inverses for *any* integer, so a
      wrong ``start``/``end`` (e.g. a raw byte offset that skipped ``.decode()``) would
      still satisfy a splice-only comparison -- this check is what actually catches that.
    * The second check catches a splice/record disagreement: it confirms
      ``mutate_shared.splice`` -- the harness's own application logic, driven by the
      ``span`` dict as emitted -- reproduces the exact same edit the generator intended.

    A record that fails either check is dropped rather than emitted.
    """
    span, start, end = _span(text, src, node)
    if text[start:end] != _text(node, src):
        return None
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


def _is_cfg_test(attr_item, src):
    """True for `#[cfg(test)]`, and for any `cfg` predicate mentioning `test`.

    Structural rather than textual: the attribute's path must be `cfg`, and a `test`
    identifier must appear inside its token tree. That accepts `#[cfg(test)]` and
    `#[cfg(all(test, feature = "x"))]` while rejecting a `#[doc = "cfg(test)"]` or a
    comment that merely contains the string.
    """
    attr = next((c for c in attr_item.children if c.type == "attribute"), None)
    if attr is None or not attr.children:
        return False
    if _text(attr.children[0], src) != "cfg":
        return False
    stack = [c for c in attr.children[1:]]
    while stack:
        n = stack.pop()
        if n.type == "identifier" and _text(n, src) == "test":
            return True
        stack.extend(n.children)
    return False


def _cfg_test_ranges(root, src):
    """Byte ranges of every item gated behind a `#[cfg(test)]` attribute.

    Inline `#[cfg(test)] mod tests` blocks are the crate's *own* unit tests, not the code
    the generated suite is scored on: a mutant there can only ever report a fact about the
    crate's inline tests. cargo-mutants skips this region by construction, so skipping it
    here is also what keeps the two tools scoring the same population.

    The exclusion lives in ``generate`` rather than in the workspace walker because it is a
    fact about Rust, not about CRUST-bench's layout -- and because every caller
    (list_transpile_mutants, the disjointness guard, future ones) then gets it for free
    instead of re-deriving it and drifting.

    In tree-sitter-rust an attribute is a *sibling* preceding the item it decorates, so the
    scan pairs each gating attribute with the next non-attribute sibling.
    """
    ranges, stack = [], [root]
    while stack:
        node = stack.pop()
        gated = False
        for child in node.children:
            if child.type == "attribute_item":
                gated = gated or _is_cfg_test(child, src)
                continue
            if gated:
                ranges.append((child.start_byte, child.end_byte))
                gated = False
                continue  # no need to descend: the whole item is excluded
            stack.append(child)
    return ranges


def generate(text, rel_path, package, operators=None):
    """Every transpilation-aware mutant for one Rust source file.

    Returns cargo-mutants-shaped records -- name, package, file, function, span,
    replacement, genre -- so everything downstream of list_mutants works unchanged.
    """
    operators = OPERATORS if operators is None else operators
    src = text.encode("utf-8")
    tree = _PARSER.parse(src)
    skip = _cfg_test_ranges(tree.root_node, src)

    seen, records = set(), []
    for op in operators:
        cursor = ts.QueryCursor(ts.Query(_LANG, op.query))
        for nodes in cursor.captures(tree.root_node).values():
            for node in nodes:
                if any(lo <= node.start_byte < hi for lo, hi in skip):
                    continue
                for target, replacement, label in op.rewrite(node, src):
                    if any(lo <= target.start_byte < hi for lo, hi in skip):
                        continue
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


# An expression is "atomic" here if a postfix `.method(..)` binds to the whole of it
# rather than to some sub-part -- i.e. it is a primary or postfix expression. Anything
# outside this set (a binary or unary expression, a cast, a reference, a block-like
# expression) binds *looser* than `.`, so splicing it as a receiver would reassociate:
# `a + b` spliced into `{}.wrapping_shl(2)` yields `a + b.wrapping_shl(2)`.
# Parenthesising an expression is semantically neutral in Rust, so the set is kept
# deliberately conservative: anything not listed here gets parentheses.
_ATOMIC_EXPR = frozenset({
    "identifier", "self", "scoped_identifier", "generic_function", "metavariable",
    "integer_literal", "float_literal", "string_literal", "raw_string_literal",
    "char_literal", "boolean_literal", "unit_expression",
    "call_expression", "field_expression", "index_expression", "macro_invocation",
    "parenthesized_expression", "tuple_expression", "array_expression",
    "try_expression", "await_expression",
})


def _operand(node, src):
    """A node's text, parenthesised unless it is already atomic (see ``_ATOMIC_EXPR``)."""
    text = _text(node, src)
    return text if node.type in _ATOMIC_EXPR else f"({text})"


def _binary_parts(node, src):
    """(left_node, op_token_text, right_node) for a binary_expression, else None.

    The grammar gives the operator as an unnamed child, so it is found positionally
    rather than by field name. Operands are returned as *nodes*, not text, because the
    callers must know each operand's shape to decide whether it needs parenthesising.
    """
    if node.type != "binary_expression" or len(node.children) != 3:
        return None
    left, op, right = node.children
    if op.is_named:
        return None
    return left, _text(op, src), right


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
                        # The replacement is a *binary* expression standing where a
                        # postfix call stood, so it binds looser than what it replaces:
                        # without the outer parentheses `acc.wrapping_mul(P).wrapping_add(z)`
                        # would become `acc * P.wrapping_add(z)`. The operands are
                        # parenthesised on the same rule, so `a.wrapping_mul(x + 1)` does
                        # not become `a * x + 1`.
                        expr = (f"({_operand(recv, src)} {_WRAP_FAMILIES[stem]} "
                                f"{_operand(inner[0], src)})")
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
            # `_operand` on the left is what keeps `365 * yoe + yoe / 4` from becoming
            # `365 * yoe.wrapping_add(yoe / 4)`. The right operand sits inside the call's
            # own parentheses and so cannot reassociate.
            yield node, f"{_operand(left, src)}.wrapping_{stem}({_text(right, src)})", \
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
        # Same precedence guard as overflow-discipline: `a + b << 2` must become
        # `(a + b).wrapping_shl(2)`, never `a + b.wrapping_shl(2)`.
        yield node, f"{_operand(left, src)}.wrapping_{stem}({_text(right, src)})", \
            f"replace {op} with wrapping_{stem}"


SHIFT_DISCIPLINE = Operator(name="shift-discipline", family="F1-integer",
                            query="(binary_expression) @b", rewrite=_rewrite_shift)


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
        if texts[i] == texts[i + 1]:
            continue  # identical adjacent arguments swap to the same text: no-op mutant
        swapped = list(texts)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        yield node, "(" + ", ".join(swapped) + ")", \
            f"transpose arguments {i + 1} and {i + 2}"


ARG_TRANSPOSITION = Operator(name="arg-transposition", family="F3-ownership",
                             query="(call_expression arguments: (arguments) @a)",
                             rewrite=_rewrite_arg_transposition)


OPERATORS = [
    ENDIANNESS, TEXT_UNIT, FLOAT_WIDTH,
    CAST_WIDTH, CAST_SIGN, OVERFLOW_DISCIPLINE, SHIFT_DISCIPLINE,
    LOST_IN_PLACE_UPDATE, ERROR_PATH_ERASURE, RANGE_BOUND, ARG_TRANSPOSITION,
]
