"""Stage 2 -- generate a C test that kills the mutant (design spec S4).

The gate is *executed*, never predicted: we compile and run T against P and against P',
and only accept T when P exits 0 and P' exits non-zero. Failures are fed back to the
model verbatim, up to `retries` times, so repair sees the real compiler and runtime.

This also value-grounds the test for free: an assertion survives only if the original
library actually produces that value.
"""
import re

from .prompts import load as load_prompt
from .llm import ask, fenced

SYSTEM = ("You are a precise C programmer writing mutation-killing tests. "
          "You reason about concrete values and follow output formats exactly.")

INCLUDES_C = re.compile(r'#\s*include\s*"[^"]*\.c"')
HAS_CHECK = re.compile(r"\bassert\s*\(|\breturn\s+[1-9]|\bexit\s*\(\s*[1-9]")


def static_checks(test_c):
    """Cheap rejections that the executed gate would not catch as clearly."""
    if INCLUDES_C.search(test_c):
        return "test #includes a .c file; it must link against the library, not inline it"
    if not HAS_CHECK.search(test_c):
        return "test contains no assertion or non-zero exit path"
    if len(test_c.splitlines()) < 5:
        return "test is trivially short"
    return None


def _describe(label, r):
    if r.stage == "compile":
        return f"{label}: the test did not COMPILE.\n```\n{r.detail[-2500:]}\n```"
    if r.stage == "link":
        return f"{label}: the test did not LINK.\n```\n{r.detail[-2000:]}\n```"
    if r.returncode == 124:
        return f"{label}: the test TIMED OUT (infinite loop or blocking read)."
    tail = (r.stdout + r.stderr)[-1200:]
    return (f"{label}: the test ran and exited {r.returncode}."
            + (f"\n```\n{tail}\n```" if tail.strip() else ""))


def gate(project, test_c, patch):
    """Run T on P and on P'. Returns (ok, feedback, {orig, mutant} results)."""
    orig = project.run(test_c)
    mut = project.run(test_c, patch=patch)
    res = {"orig": orig, "mutant": mut}

    if not orig.ok:
        return False, _describe("On the ORIGINAL library", orig), res
    if orig.returncode != 0:
        return False, (_describe("On the ORIGINAL library", orig) +
                       "\nIt must exit 0 on the original -- your expected values are "
                       "wrong, or you called the API incorrectly."), res
    if not mut.ok:
        # The mutant no longer builds *with this test*: a signature/ODR clash, not a kill.
        return False, (_describe("On the MUTANT", mut) +
                       "\nA build failure is not a kill. The test must build against "
                       "both versions and differ only in its runtime result."), res
    if mut.returncode == 0:
        return False, (_describe("On the MUTANT", mut) +
                       "\nIt must exit NON-ZERO on the mutant. Your assertions are not "
                       "sensitive to the mutated behaviour -- find an input that "
                       "exercises the changed line and assert the exact value the "
                       "ORIGINAL produces for it."), res
    return True, None, res


def generate(project, mutant, iface_text, source_bundle, diff, retries=3, tag="t"):
    """Produce a killing C test for one gated mutant.

    Returns a record with the test source (or None) and the full attempt trail.
    """
    rec = mutant["record"]
    base = load_prompt(
        "testgen", project=project.name, sources=source_bundle, iface=iface_text,
        fault_class=rec.get("fault_class", "?"), function=rec.get("function", "?"),
        file=mutant["verdict"]["file"], rationale=rec.get("rationale", ""), diff=diff)

    attempts, user, prev = [], base, None
    for i in range(retries + 1):
        reply = ask(SYSTEM, user, effort="high", max_tokens=16000, tag=f"tst_{tag}_{i}")
        if reply.strip().startswith("EQUIVALENT:") or "\nEQUIVALENT:" in reply:
            attempts.append({"attempt": i, "outcome": "declared_equivalent",
                             "feedback": reply.strip()[:400]})
            return {"test_c": None, "killed": False, "reason": "declared_equivalent",
                    "attempts": attempts}
        test_c = fenced(reply, "c")
        if not test_c:
            fb = "You returned no fenced C code block."
        else:
            fb = static_checks(test_c)
        if fb is None:
            ok, fb, res = gate(project, test_c, mutant["patch"])
            attempts.append({"attempt": i, "outcome": "killed" if ok else "gate_failed",
                             "orig_rc": res["orig"].returncode if res["orig"].ok else None,
                             "orig_stage": res["orig"].stage,
                             "mutant_rc": res["mutant"].returncode if res["mutant"].ok else None,
                             "mutant_stage": res["mutant"].stage,
                             "feedback": (fb or "")[:1200]})
            if ok:
                return {"test_c": test_c, "killed": True, "reason": "ok",
                        "attempts": attempts}
        else:
            attempts.append({"attempt": i, "outcome": "static_reject", "feedback": fb})

        prev = test_c or "(no code returned)"
        user = base + "\n\n---\n\n" + load_prompt(
            "testgen_repair", feedback=fb, prev=prev)

    return {"test_c": None, "killed": False, "reason": "retries_exhausted",
            "attempts": attempts}


def collateral(project, test_c, mutants, skip_id):
    """Which *other* mutants of this project does T also kill? (spec S7 diagnostics)"""
    out = []
    for m in mutants:
        mid = m["verdict"]["id"]
        if mid == skip_id:
            continue
        r = project.run(test_c, patch=m["patch"])
        out.append({"id": mid, "killed": bool(r.ok and r.returncode != 0),
                    "stage": r.stage, "rc": r.returncode})
    return out
