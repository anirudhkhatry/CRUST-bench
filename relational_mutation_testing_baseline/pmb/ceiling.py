"""How much of PyMigBench can each oracle actually reach?

Phase 0 measured one thing: can this migration carry the *mutation funnel*? That funnel
mutates P, so phase 0 asks P to import first and never looks at the post side when it
does not -- `post_ok` is 165 but the post side was only ever attempted 176 times. That
short-circuit is correct for the funnel and wrong for everything else we now run, because
judging a candidate migration needs M and M_llm and never needs P at all.

So this module re-measures the ceiling as a ladder of independent tiers:

    tier 0   the changed file parses, on both sides. Needs nothing -- no venv, no
             network, no interpreter of the right era. This is the floor under the
             static developer-diff oracle, and the check that says whether a *candidate*
             is even syntactically Python.
    tier 2   the changed file imports on the post side, asked directly rather than
             behind a pre-side gate. This is the real bar for `validate`: run a test
             against M, run it against M_llm.

Tier 1 -- resolving names against the target library alone, without the repo -- is not
built yet; its value depends on how much of the gap tier 2 closes on its own.

Nothing here spends a token. Tier 0 is arithmetic on the AST; tier 2 is the same venv
machinery phase 0 uses, pointed at one side.
"""
import ast
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import pybench as PB
from .dataset import all_migrations
from .phase0 import PY2_SYNTAX, looks_python2

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = RESULTS / "pmb_ceiling"


def compiles(text):
    """Does this source parse? Returns (ok, diagnostic).

    `ast.parse` rather than `compile`, so that a file which parses but emits a
    SyntaxWarning -- an unescaped regex, which a great many of these files contain --
    counts as the pass it is.
    """
    try:
        ast.parse(text)
        return True, ""
    except SyntaxError as e:
        return False, f"{type(e).__name__}: {e}"[:200]
    except (ValueError, RecursionError, MemoryError) as e:
        return False, f"{type(e).__name__}: {e}"[:200]


def tier0(mig, trees):
    """Tier 0 over every candidate file, pre and post.

    Reported per candidate because the later tiers are free to study the migration
    through a quieter sibling, exactly as phase 0 is: a record where the busiest file is
    Python 2 but a sibling is not still has somewhere to stand.
    """
    per, any_ok = {}, False
    for path in mig.candidates[:4]:
        rec = {}
        for side in ("pre", "post"):
            f = trees[side] / path
            if not f.is_file():
                rec[side] = None
                continue
            ok, why = compiles(f.read_text(errors="ignore"))
            rec[side] = ok
            if not ok:
                rec[f"{side}_why"] = why
        if rec.get("pre") and rec.get("post"):
            any_ok = True
        per[path] = rec
    return {"tier0_ok": any_ok, "tier0_main": bool(per.get(mig.main_path, {}).get("post")),
            "tier0_per_file": per}


def _post_only(mig, tree, path, pyver, newer):
    """Ask the post side to import, with no pre-side gate in front of it."""
    env, note = PB.build_env(tree, mig.target, pyver, "ceil_post", path, newer=newer)
    if env is None and ("SyntaxError" in note or "invalid syntax" in note) \
            and not any(s in note for s in PY2_SYNTAX):
        # Wrong era, not wrong code -- the same single retry phase 0 allows.
        alt = PB.FALLBACK_PY.get(pyver, "3.8")
        env, note = PB.build_env(tree, mig.target, alt, f"ceil_post_{alt}", path,
                                 newer=newer)
        note = f"[retry {alt}] {note}"
    return env, note


def probe(mig):
    t0 = time.time()
    rec = {"id": mig.id, "source": mig.source, "target": mig.target}

    ok, note = PB.fetch(mig)
    rec["fetch"] = note
    if not ok:
        return {**rec, "tier0_ok": False, "tier2_ok": False, "level": "fetch_failed",
                "seconds": round(time.time() - t0, 1)}

    trees = {}
    for side in ("pre", "post"):
        t = PB.materialise(mig, side)
        if t is None or not (t / mig.main_path).is_file():
            return {**rec, "tier0_ok": False, "tier2_ok": False, "level": "tree_missing",
                    "side": side, "seconds": round(time.time() - t0, 1)}
        trees[side] = t

    rec.update(tier0(mig, trees))

    pyver = PB.python_for(mig)
    newer = PB.commit_date(mig)
    rec["tier2_ok"], rec["tier2_path"] = False, None
    for path in mig.candidates[:4]:
        if not (trees["post"] / path).is_file():
            continue
        text = (trees["post"] / path).read_text(errors="ignore")
        if not compiles(text)[0] or looks_python2(text):
            # Tier 0 already settles this file; building a venv for it would only spend
            # a minute to be told the same thing.
            rec.setdefault("tier2_notes", {})[path] = "tier0 failed or python2"
            continue
        env, note = _post_only(mig, trees["post"], path, pyver, newer)
        rec.setdefault("tier2_notes", {})[path] = note[-300:]
        if env is not None:
            rec["tier2_ok"], rec["tier2_path"] = True, path
            break

    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def one(mig, redo=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{mig.id}.json"
    if out.exists() and not redo:
        return json.loads(out.read_text())
    try:
        rec = probe(mig)
    except Exception as e:  # noqa: BLE001 - one record must not sink the sweep
        rec = {"id": mig.id, "tier0_ok": False, "tier2_ok": False, "level": "error",
               "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()[-1500:]}
    out.write_text(json.dumps(rec, indent=2))
    return rec


def run(workers=3, ids=None, redo=False):
    migs = [m for m in all_migrations() if not ids or m.id in ids]
    print(f"ceiling sweep: {len(migs)} migrations, {workers} workers", flush=True)
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, m, redo): m for m in migs}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            done += 1
            print(f"[{done}/{len(migs)}] {r['id']}: tier0={r.get('tier0_ok')} "
                  f"tier2={r.get('tier2_ok')}", flush=True)
    return report(rows)


def report(rows=None):
    rows = rows or [json.loads(p.read_text()) for p in sorted(CACHE.glob("*.json"))]
    phase0 = {}
    d = RESULTS / "pmb_phase0"
    if d.is_dir():
        phase0 = {json.loads(p.read_text())["id"]: json.loads(p.read_text()).get("level")
                  for p in d.glob("*.json")}

    t0 = [r for r in rows if r.get("tier0_ok")]
    t2 = [r for r in rows if r.get("tier2_ok")]
    # The number the sweep exists to produce: post-side imports that the pre-side gate
    # was hiding.
    gained = sorted(r["id"] for r in t2 if phase0.get(r["id"]) not in (None, "runnable"))
    lost = sorted(r["id"] for r in rows
                  if phase0.get(r["id"]) == "runnable" and not r.get("tier2_ok"))

    summary = {
        "n": len(rows),
        "tier0_compiles": len(t0),
        "tier2_post_imports": len(t2),
        "phase0_runnable": sum(1 for v in phase0.values() if v == "runnable"),
        "gained_over_phase0": len(gained),
        "lost_vs_phase0": len(lost),
        "by_phase0_level": {},
    }
    for r in rows:
        lvl = phase0.get(r["id"], "unknown")
        b = summary["by_phase0_level"].setdefault(lvl, {"n": 0, "tier0": 0, "tier2": 0})
        b["n"] += 1
        b["tier0"] += bool(r.get("tier0_ok"))
        b["tier2"] += bool(r.get("tier2_ok"))

    (RESULTS / "CEILING.json").write_text(json.dumps(
        {**summary, "gained": gained, "lost": lost}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
