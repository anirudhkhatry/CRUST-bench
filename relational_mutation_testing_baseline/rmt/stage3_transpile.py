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
