"""Stage 2 -- generate a pytest test that kills the mutant.

The gate is *executed*, never predicted: we run T against P and against P', and accept T
only when it is green on P and red on P'. Failures go back to the model verbatim, so
repair sees the real pytest output.

One asymmetry with the C side is worth naming. A C test signals failure by exit status,
so "did not compile" and "failed" are the same non-zero number and have to be told apart
by the harness. pytest already distinguishes them -- a collection error is `build_error`
here -- and a test that never loaded must not be scored as a kill, on either side.
"""
import re

from . import pybench as PB
from .llm import ask, fenced
from .prompts import load as load_prompt

SYSTEM = ("You are a precise Python programmer writing mutation-killing tests. "
          "You reason about concrete values and follow output formats exactly.")

HAS_ASSERT = re.compile(r"^\s*assert\b|pytest\.raises|self\.assert", re.M)
HAS_TEST = re.compile(r"^\s*(async\s+)?def\s+test\w*\s*\(", re.M)
# A test that reads the module's own source and re-derives the answer from it passes
# both versions by construction.
CHEATS = re.compile(r"inspect\.getsource|__file__.*read|linecache")


def static_checks(test_py):
    if not HAS_TEST.search(test_py):
        return "file contains no `def test_...` function, so pytest collects nothing"
    if not HAS_ASSERT.search(test_py):
        return "test contains no assertion"
    if CHEATS.search(test_py):
        return "test inspects the module's source instead of calling it"
    if len(test_py.splitlines()) < 5:
        return "test is trivially short"
    return None


def _describe(label, r):
    if r["status"] == "build_error":
        return (f"{label}: the test did not even LOAD (collection error).\n```\n"
                f"{r['out'][-2500:]}\n```")
    if r["status"] == "timeout":
        return f"{label}: the test TIMED OUT (blocking call, sleep, or real network?)."
    if r["status"] == "no_tests":
        return f"{label}: pytest collected no tests from the file."
    return (f"{label}: {r['passed']} passed, {r['failed']} failed."
            + (f"\n```\n{r['out'][-1800:]}\n```" if r["out"].strip() else ""))


def gate(env, test_py, patch):
    """Run T on P and on P'. Returns (ok, feedback, {orig, mutant} results)."""
    orig = PB.run_test(env, test_py)
    res = {"orig": orig}
    if orig["status"] == "build_error":
        return False, _describe("On the ORIGINAL module", orig), res
    if orig["status"] in ("no_tests", "timeout"):
        return False, _describe("On the ORIGINAL module", orig), res
    if orig["status"] != "green":
        return False, (_describe("On the ORIGINAL module", orig) +
                       "\nIt must pass on the original -- your expected values are "
                       "wrong, you patched a name the module does not call, or the "
                       "module attempted real I/O."), res

    mut = PB.run_test(env, test_py, patch=patch)
    res["mutant"] = mut
    if mut["status"] == "build_error":
        # The mutant no longer imports *with this test*: a broken file, not a kill.
        return False, (_describe("On the MUTANT", mut) +
                       "\nA collection error is not a kill. The test must load against "
                       "both versions and differ only in its assertion result."), res
    if mut["status"] == "green":
        return False, (_describe("On the MUTANT", mut) +
                       "\nIt must FAIL on the mutant. Your assertions are not sensitive "
                       "to the mutated behaviour -- find an input that exercises the "
                       "changed code and assert the exact value the ORIGINAL produces "
                       "for it."), res
    return True, None, res


def generate(mig, env, mutant, diff, retries=3, tag="t", lib=None):
    """Produce a killing pytest test for one gated mutant.

    `lib` is the library the module in `env` actually uses. It defaults to the source
    library because that is the side Stage 2 normally runs on, but `target_arm` points
    this same machinery at the post-migration module, where saying `source` would invite
    the model to mock a library the file no longer imports.
    """
    rec = mutant["record"]
    # On the migration-anchored arm the mutant is not an arbitrary fault: it is the
    # image, in this file, of a specific way the migration could go wrong. Saying which
    # one keeps the test aimed at behaviour the migration has to preserve, rather than at
    # whatever incidental difference is easiest to assert on.
    hazard = rec.get("migration_hazard", "")
    hazard_block = (f"\nWhy this fault matters: it is what this module would do if the "
                    f"migration to `{mig.target}` mishandled the following difference "
                    f"between the two libraries.\nMigration hazard: {hazard}\n"
                    f"The behaviour your test pins down must be the behaviour that "
                    f"difference threatens." if hazard else "")
    base = load_prompt(
        "testgen", repo=mig.repo, file=env.rel, source=lib or mig.source,
        source_code=env.read(), fault_class=rec.get("fault_class", "?"),
        function=rec.get("function", "?"), rationale=rec.get("rationale", ""),
        hazard_block=hazard_block, diff=diff)

    attempts, user = [], base
    for i in range(retries + 1):
        reply = ask(SYSTEM, user, effort="high", max_tokens=16000, tag=f"tst_{tag}_{i}")
        if reply.strip().startswith("EQUIVALENT:") or "\nEQUIVALENT:" in reply:
            attempts.append({"attempt": i, "outcome": "declared_equivalent",
                             "feedback": reply.strip()[:400]})
            return {"test_py": None, "killed": False, "reason": "declared_equivalent",
                    "attempts": attempts}
        test_py = fenced(reply, "python")
        fb = "You returned no fenced Python code block." if not test_py \
            else static_checks(test_py)
        if fb is None:
            ok, fb, res = gate(env, test_py, mutant["patch"])
            attempts.append({"attempt": i, "outcome": "killed" if ok else "gate_failed",
                             "orig": res["orig"]["status"],
                             "mutant": res.get("mutant", {}).get("status"),
                             "feedback": (fb or "")[:1200]})
            if ok:
                return {"test_py": test_py, "killed": True, "reason": "ok",
                        "attempts": attempts}
        else:
            attempts.append({"attempt": i, "outcome": "static_reject", "feedback": fb})

        user = base + "\n\n---\n\n" + load_prompt(
            "testgen_repair", feedback=fb, prev=test_py or "(no code returned)")

    return {"test_py": None, "killed": False, "reason": "retries_exhausted",
            "attempts": attempts}


def collateral(env, test_py, mutants, skip_id):
    """Which *other* mutants of this module does T also kill? (mutant-redundancy check)"""
    out = []
    for m in mutants:
        mid = m["verdict"]["id"]
        if mid == skip_id:
            continue
        r = PB.run_test(env, test_py, patch=m["patch"])
        out.append({"id": mid, "killed": r["status"] in ("test_failed", "timeout"),
                    "status": r["status"]})
    return out
