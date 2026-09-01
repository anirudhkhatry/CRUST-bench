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
