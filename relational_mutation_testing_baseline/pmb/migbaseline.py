"""The migration baseline -- ask a model to do the migration, and keep what it produces.

Everything else in this package treats the post-migration file M as given: the repository
shipped it, so it is correct, and the study mutates it to make something wrong. That gives
a *synthetic* incorrect version, which is fine for asking whether a test transferred but
not for asking whether a test can tell a good migration from a bad one. A ported fault is
not the kind of mistake a migration makes.

So this produces the real thing: P handed to a model with nothing but the two library
names -- no API mapping, no sight of M -- and whatever comes back is a candidate migration
M_llm. Sometimes right, sometimes subtly wrong, occasionally not a migration at all. Those
candidates are the population the migrated tests are then asked to judge.

The prompt is not ours. It is the one published in "An Empirical Study of Python Library
Migration Using Large Language Models" (arXiv:2504.13272), which ran exactly this task
over exactly this benchmark, reproduced verbatim in `prompts/migrate_code.md`. Using it
unchanged is the point: whole file, both library names, no API mapping and no examples, so
our candidates are drawn from the same distribution as the published ones and our numbers
sit beside theirs rather than needing a translation.

What that study could *not* do is the reason this module exists. Its static evaluation
covers every migration, but its runtime evaluation -- actually running tests against the
migrated code -- covers 25 of 314, because only 25 client repos ship a suite that installs
and runs. That is the gap the migrated tests close: T' is generated per migration and
needs nothing from the repository, so runtime evidence becomes available where there was
none.

One arm per invocation, and the arm *is* the model (`RMT_MODEL`), following the study's own
design of comparing models of differing capability -- it reports 26%/49%/57% fully-correct
migrations for Llama 3.1 / GPT-4o mini / GPT-4o. A spread of model strength is what makes
the candidate population contain both correct and incorrect migrations, which a single
strong model would not; handicapping one model's context instead would produce failures of
our own manufacture rather than the ones migration actually suffers.
"""
import ast
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import llm
from . import pybench as PB
from .llm import MODEL, ask, fenced
from .prompts import load as load_prompt

RESULTS = Path(__file__).resolve().parent.parent / "results"

# The published prompt carries all the task instructions, so the system message says only
# what the API needs said. Anything more would be an addition to a borrowed instrument.
SYSTEM = "You are a helpful assistant that migrates Python code between libraries."


def references(src, lib):
    """Does this source actually import the named library? AST first, text as fallback.

    A candidate that never mentions the target library did not migrate anything, and no
    verdict a test returns about it would mean what we want it to mean. Checked
    mechanically, because it is the one property of a migration needing no judgement.
    """
    root = lib.replace("-", "_").split(".")[0].lower()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return root in src.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0].lower() == root for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0].lower() == root:
                return True
    return root in src.lower()


def propose(mig, pre_src, rel, tag, retries=2):
    """One candidate migration of P. Returns (text|None, record).

    The reply is prose then code, as the published prompt asks for, so the code is taken
    from the fenced block and the explanation is kept beside it -- the study keeps it for
    human review and it is the only account we get of what the model thought it was doing.
    """
    user = load_prompt("migrate_code", source=mig.source, target=mig.target,
                       impl=pre_src)
    trail = []
    for i in range(retries + 1):
        reply = ask(SYSTEM, user, effort="high", max_tokens=64000,
                    tag=f"migc_{tag}_{i}")
        code = fenced(reply, "python")
        if not code or not code.strip():
            trail.append({"attempt": i, "outcome": "no_code_block"})
            user += ("\n\n---\n\nThat reply contained no fenced Python code block. "
                     "Provide the complete modified file in one ```python block.")
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            trail.append({"attempt": i, "outcome": "syntax_error", "detail": str(e)[:200]})
            user += (f"\n\n---\n\nThat code does not parse: {e}. It may have been cut "
                     "off. Provide the complete modified file in one ```python block.")
            continue
        # Truncation that still parses is the failure mode the study calls a response
        # failure, and it is silent: a file that lost its last 200 lines is valid Python.
        # A candidate much shorter than P is therefore treated as truncated, not migrated.
        ratio = len(code.splitlines()) / max(1, len(pre_src.splitlines()))
        if ratio < 0.6:
            trail.append({"attempt": i, "outcome": "truncated", "ratio": round(ratio, 2)})
            user += (f"\n\n---\n\nThat reply contained only {round(100 * ratio)}% of the "
                     "file's lines, so it was cut off. Provide the *complete* modified "
                     "file, every line of it, in one ```python block.")
            continue
        return code, {"model": MODEL, "applied": True, "loc": len(code.splitlines()),
                      "explanation": reply.split("```")[0].strip()[:2000],
                      "trail": trail}
    return None, {"model": MODEL, "applied": False,
                  "reason": trail[-1]["outcome"] if trail else "failed", "trail": trail}


def candidate(mig, post_env, pre_src, outdir, redo=False):
    """Build, gate and store one candidate migration. Returns its record.

    The candidate is probed in the *post-migration* environment: it is P's code rewritten
    against the target library, so the target library has to be installed, and the post
    venv is the only one where it is. Running it there also means M and M_llm are judged
    under identical conditions -- same tree, same interpreter, same installed versions --
    so a difference between them is a difference in the code and not in the environment.

    The gates run in the order a defect gets harder to spot:
      no_patch      the model produced nothing usable
      no_import     the candidate does not load             -- anyone detects this
      no_reference  it loads but never imports the target   -- anyone detects this
      admissible    it loads and uses the target library     -- the tests' business
    """
    d = outdir / "candidates" / MODEL
    d.mkdir(parents=True, exist_ok=True)
    done = d / "candidate.json"
    if done.exists() and not redo:
        return json.loads(done.read_text())

    text, rec = propose(mig, pre_src, post_env.rel, tag=mig.id)
    rec["migration"] = mig.id
    rec["file"] = post_env.rel
    if text is None:
        rec["gate"] = "no_patch"
    else:
        (d / "candidate.py").write_text(text)
        patch = {post_env.rel: text}
        probe = PB.run_test(post_env, None, patch=patch)
        rec["imports"] = probe["status"] == "green"
        rec["references_target"] = references(text, mig.target)
        rec["still_imports_source"] = references(text, mig.source)
        rec["gate"] = ("admissible" if rec["imports"] and rec["references_target"]
                       else "no_import" if not rec["imports"] else "no_reference")
        if not rec["imports"]:
            rec["import_error"] = probe["out"][-1200:]
    done.write_text(json.dumps(rec, indent=2))
    return rec


def candidate_text_only(mig, outdir, redo=False):
    """A candidate for a migration whose environment will not build.

    Most of PyMigBench cannot be executed -- 165 of 321 migrations import on both sides,
    and the rest fail on a sibling dependency, a dead transitive pin or Python 2, none of
    which is a property of the migration itself. Those records are still perfectly
    scorable by the two static oracles: does the candidate parse, and did it make the
    edits the developer made. Refusing to generate a candidate for them would throw away
    the larger half of the dataset to protect a gate that was never the point.

    So this runs the same prompt against the same file read straight from the tree, and
    substitutes `compiles` for the import probe. The record is tagged `static_only` so no
    downstream table silently mixes a parsed candidate with an imported one.
    """
    from . import changed_dev as CD

    d = outdir / "candidates" / MODEL
    d.mkdir(parents=True, exist_ok=True)
    done = d / "candidate.json"
    if done.exists() and not redo:
        return json.loads(done.read_text())

    rel = mig.main_path
    pre_src = CD.read_side(mig, "pre", rel)
    if pre_src is None:
        rec = {"migration": mig.id, "model": MODEL, "gate": "no_tree", "file": rel}
        done.write_text(json.dumps(rec, indent=2))
        return rec

    text, rec = propose(mig, pre_src, rel, tag=mig.id)
    rec.update({"migration": mig.id, "file": rel, "static_only": True})
    if text is None:
        rec["gate"] = "no_patch"
    else:
        (d / "candidate.py").write_text(text)
        try:
            ast.parse(text)
            rec["compiles"] = True
        except SyntaxError as e:
            rec["compiles"] = False
            rec["syntax_error"] = str(e)[:300]
        rec["references_target"] = references(text, mig.target)
        rec["still_imports_source"] = references(text, mig.source)
        rec["gate"] = ("admissible_static" if rec["compiles"] and rec["references_target"]
                       else "no_compile" if not rec["compiles"] else "no_reference")
    done.write_text(json.dumps(rec, indent=2))
    return rec


def run(run_name, migs, workers=3, redo=False, allow_static=False):
    """Produce one candidate migration per migration in the panel, for the current model.

    Candidates live beside the run's tests, under `<run>/<migration>/candidates/<model>/`,
    because a candidate is only meaningful next to the T' that will judge it: both were
    built from the same P and the same file phase 0 pinned.
    """
    root = RESULTS / run_name
    root.mkdir(parents=True, exist_ok=True)
    print(f"migbaseline: run={run_name} migrations={len(migs)} model={MODEL} "
          f"workers={workers}", flush=True)

    def one(mig):
        outdir = root / mig.id
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            pre, post = PB.open_migration(mig)
            return candidate(mig, post, pre.read(), outdir, redo=redo)
        except Exception as e:  # noqa: BLE001 - one repo must not sink the panel
            (outdir / "CANDIDATE_ERROR.txt").write_text(traceback.format_exc())
            if allow_static:
                try:
                    return candidate_text_only(mig, outdir, redo=redo)
                except Exception:  # noqa: BLE001
                    (outdir / "CANDIDATE_ERROR.txt").write_text(traceback.format_exc())
            return {"migration": mig.id, "model": MODEL, "gate": "env_failed",
                    "reason": f"{type(e).__name__}: {e}"}

    t0, rows = time.time(), []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, m): m for m in migs}
        for n, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            rows.append(rec)
            print(f"[{n}/{len(migs)}] {rec['migration']}: {rec['gate']}", flush=True)

    (root / f"candidates_{MODEL}.json").write_text(json.dumps(rows, indent=2))
    gates = {}
    for r in rows:
        gates[r["gate"]] = gates.get(r["gate"], 0) + 1
    cost = llm.cost_ledger()
    print(f"\nmigbaseline done in {round(time.time() - t0)}s; {gates}; "
          f"${cost['total']['paid_usd']:.2f} paid cumulative", flush=True)
    return rows
