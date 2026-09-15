"""Find functions whose shipped code no longer matches the plan that specified them.

The plan is the durable artifact: if someone regenerates a brief from it, they get what it
says, not what is in git. Two fixes have already been applied to code and silently left out
of the plan, so this checks all of them rather than the ones I happen to remember.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench/.claude/worktrees/additional-datasets")
PKG = ROOT / "relational_mutation_testing_baseline"
PLANS = [ROOT / "docs/superpowers/plans/2026-09-14-additional-datasets-harness-and-js.md"]


def plan_python_blocks(path):
    lines = path.read_text().split("\n")
    out, cur, lang = [], None, None
    for ln in lines:
        m = re.match(r"^```(\w*)\s*$", ln)
        if m and cur is None:
            lang, cur = m.group(1), []
        elif ln.strip() == "```" and cur is not None:
            if lang == "python":
                out.append("\n".join(cur))
            cur = None
        elif cur is not None:
            cur.append(ln)
    return out


def funcs_of(src):
    """{name: normalised body} for every top-level def in a source string."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # Compare structure, not comments or docstrings: drift that matters is a
            # changed guard or a changed expression, not a reworded docstring.
            stripped = ast.dump(ast.parse(ast.unparse(node)), annotate_fields=False)
            out[node.name] = stripped
    return out


plan_funcs = {}
for p in PLANS:
    for block in plan_python_blocks(p):
        for name, body in funcs_of(block).items():
            plan_funcs.setdefault(name, body)

code_funcs = {}
for f in sorted(PKG.glob("additional_datasets/**/*.py")):
    for name, body in funcs_of(f.read_text()).items():
        code_funcs[name] = (body, str(f.relative_to(PKG)))

drift, missing, ok = [], [], 0
for name, planned in sorted(plan_funcs.items()):
    if name.startswith("test_") or name.startswith("_task") or name in {"_script", "_row_"}:
        continue
    if name not in code_funcs:
        missing.append(name)
        continue
    shipped, where = code_funcs[name]
    if shipped != planned:
        drift.append((name, where))
    else:
        ok += 1

print(f"{ok} functions match the plan exactly")
if drift:
    print(f"\nDRIFT -- shipped code differs from the plan ({len(drift)}):")
    for name, where in drift:
        print(f"  {name:28} {where}")
if missing:
    print(f"\nin the plan but not shipped ({len(missing)}): {', '.join(missing)}")
sys.exit(1 if drift else 0)
