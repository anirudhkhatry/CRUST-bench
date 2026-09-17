"""Survey every RefactorBench-JS fixture's oracle. No LLM calls: vitest only.

Sizing gate for a full-corpus run. A fixture can produce labelled candidates only if its
holdout suite RUNS against the fixture's own untouched sources and leaves a non-empty
admissible set (the cases green on P0). One that does not can label nothing -- every
candidate would come back GOOD by default -- so it must be excluded BEFORE any paid call
is spent on it, not after.

`pure_logic_fixtures` is not used here. It excludes every fixture whose holdout imports
@testing-library/react, which was right only while staged workspaces could not resolve
bare specifiers; since `stage` symlinks node_modules they run fine, and all three fixtures
the experiment currently rests on are React ones.
"""
import json
import sys
import time
from pathlib import Path

from .adapters import js_oracle
from .adapters import js_vitest as JS


def survey(names=None, out=None, log=print):
    names = names or [p.name for p in sorted(JS.FIXTURES.iterdir()) if p.is_dir()]
    rows = []
    for i, name in enumerate(names, 1):
        t0 = time.time()
        try:
            o = js_oracle.prepare_oracle(name, Path("/tmp/survey_ws") / name)
            row = {"fixture": name, "n_admissible": len(o["admissible_oracle"]),
                   "reason": o["reason"], "error": None,
                   "n_cases_total": None, "seconds": round(time.time() - t0, 1)}
        except Exception as e:                        # a fixture that cannot even stage
            row = {"fixture": name, "n_admissible": 0, "reason": "",
                   "error": f"{type(e).__name__}: {e}"[:200], "n_cases_total": None,
                   "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        log(f"[{i}/{len(names)}] {name}: admissible={row['n_admissible']} "
            f"{row['reason'] or row['error'] or 'green'} ({row['seconds']}s)", flush=True)
        if out:
            Path(out).write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    survey(out=sys.argv[1] if len(sys.argv) > 1 else None)
