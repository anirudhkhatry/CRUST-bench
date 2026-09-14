"""Stage 3 -- transpile the killing C test into a Rust test (design spec S4).

The model sees the C test and the RBench *interfaces* only, never an implementation, so
it cannot copy an answer out of a candidate. Repair is compile-errors-only: we feed back
rustc diagnostics and forbid weakening assertions, because a test that builds by
checking less has silently become a different experiment.
"""
import re

from .prompts import load as load_prompt
from . import rbench as R
from .llm import ask, fenced

SYSTEM = ("You are a precise Rust programmer transpiling tests. You preserve every "
          "assertion's discriminating power and use only APIs you have been shown.")

UNTRANSLATABLE = re.compile(r"//\s*UNTRANSLATABLE:", re.I)


def transpile(project, test_c, retries=3, tag="r"):
    """Returns {test_rs, compiles, attempts, n_untranslatable}."""
    mods = R.lib_modules(project) or ["<module>"]
    base = load_prompt("transpile", test_c=test_c, iface=R.iface_source(project),
                        crate=R.crate_name(project), modules=", ".join(mods),
                        module0=mods[0])

    attempts, user = [], base
    for i in range(retries + 1):
        reply = ask(SYSTEM, user, effort="high", max_tokens=16000, tag=f"tr_{tag}_{i}")
        test_rs = fenced(reply, "rust")
        if not test_rs:
            attempts.append({"attempt": i, "outcome": "no_code"})
            user = base + "\n\nYou returned no fenced Rust block. Return the file."
            continue

        res = R.compile_check(project, test_rs, tag=f"tc{i}")
        attempts.append({"attempt": i, "outcome": res["status"],
                         "passed": res["passed"], "failed": res["failed"]})
        if res["status"] != "build_error":
            # It compiles. Whether it passes or fails is Stage 4's question, not ours.
            return {"test_rs": test_rs, "compiles": True, "attempts": attempts,
                    "n_untranslatable": len(UNTRANSLATABLE.findall(test_rs))}

        errs = "\n".join(l for l in res["out"].splitlines()
                         if l.startswith(("error", "  -->", "warning: unused")))[-4000:]
        user = base + "\n\n---\n\n" + load_prompt(
            "transpile_repair", errors=errs or res["out"][-3000:], prev=test_rs)

    return {"test_rs": test_rs, "compiles": False, "attempts": attempts,
            "n_untranslatable": len(UNTRANSLATABLE.findall(test_rs or ""))}


def repair_unsound(project, test_c, test_rs, failure, retries=2, tag="u"):
    """Stage 3.5 -- the transpiled test fails *correct* Rust; re-derive it.

    Compile repair (above) is not allowed to touch assertions. This is the one place
    that may, and it is fenced two ways. The model still never sees an implementation:
    it gets the C test, the interfaces and the failing assertion's output, so a fix has
    to come from re-reading the contract, not from copying what the reference does. And
    the result is only *accepted* by Stage 4's kill check -- a test loosened until it
    passes everything is an F4, not a success.

    Returns {test_rs, compiles, attempts}; `test_rs` is the best candidate, which the
    caller re-trials on M.
    """
    mods = R.lib_modules(project) or ["<module>"]
    base = load_prompt("transpile", test_c=test_c, iface=R.iface_source(project),
                        crate=R.crate_name(project), modules=", ".join(mods),
                        module0=mods[0])
    attempts, prev, out = [], test_rs, failure
    for i in range(retries + 1):
        user = base + "\n\n---\n\n" + load_prompt("transpile_unsound", prev=prev,
                                                       failure=out[-3500:])
        reply = ask(SYSTEM, user, effort="high", max_tokens=16000, tag=f"tu_{tag}_{i}")
        cand = fenced(reply, "rust")
        if not cand:
            attempts.append({"attempt": i, "outcome": "no_code"})
            continue
        if "REFERENCE_DISAGREES" in cand.split("#[test]")[0]:
            # The model stands by the C-derived value; the caller decides what that means.
            attempts.append({"attempt": i, "outcome": "reference_disagrees"})
            return {"test_rs": cand, "compiles": None, "attempts": attempts,
                    "stands_by_c": True}
        res = R.compile_check(project, cand, tag=f"tu{i}")
        attempts.append({"attempt": i, "outcome": res["status"],
                         "passed": res["passed"], "failed": res["failed"]})
        prev = cand
        if res["status"] == "green":
            return {"test_rs": cand, "compiles": True, "attempts": attempts,
                    "stands_by_c": False}
        out = res["out"] or "(no output)"
    return {"test_rs": prev, "compiles": attempts[-1]["outcome"] != "build_error"
            if attempts else False, "attempts": attempts, "stands_by_c": False}
