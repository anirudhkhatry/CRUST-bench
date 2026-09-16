"""ARM N -- the naive baseline: tests written from P0 and the migration spec, no mutant.

This is the arm the pipeline's claim is measured against, so everything it is denied must
be denied for a reason. It gets the same model, the same effort, the same repair budget
and the same output format as ARM M.

ARM N IS A MATCHED CONTROL (decision 2026-09-15, user-directed; spec's "ARM N is a
matched control" section). It draws `N_DRAWS` suites -- the same count ARM M draws
mutants -- gives each the same `_generate_with_repair` budget, filters them the same way
except for the kill gate, and assembles them through the same `assemble.assemble` helper.
The single surviving difference is that ARM M additionally requires each draw to be RED on
its mutant, which is the variable under test. Before this, ARM N got one suite and no
selection while ARM M got six and best-of-six, so an ARM M win could not distinguish
"mutation pressure works" from "six filtered samples beat one unfiltered sample".

TWO RESIDUAL ASYMMETRIES ARE INTENTIONAL AND ARE DISCLOSED RATHER THAN REMOVED:

1. ARM M spends ~3 extra calls proposing its faults (`mutation.propose_mutants`). That is
   the cost of knowing a fault at all, and no version of ARM N needs it.
2. ARM M's draws are diverse because each prompt carries a *different* fault description;
   ARM N's draws differ only by sampling noise, because `rmt.llm.ask` does not plumb
   `temperature` (and that module is shared with two other pipelines, so it is not changed
   from here). Each draw therefore varies only in its cache tag -- which is load-bearing:
   `ask` names its cache entry `<tag>_<hash>.json`, so reusing one tag would return the
   same suite six times for free and silently collapse ARM N back to a single draw.
   Giving ARM N fault-driven diversity would mean giving it faults, which is the thing
   being withheld.

Both residuals favour ARM M, i.e. they bias against this project's own thesis, which is
the honest direction for a residual to run.
"""
from rmt.llm import ask, fenced

from ..prompts import load
from ..runner import assert_no_leak
from .assemble import N_DRAWS, assemble

SYSTEM = ("You are a precise programmer writing tests. You assert exact values, you use "
          "only APIs you have been shown, and you never write a test that could fail a "
          "correct implementation.")


def generate(task, sources, adapter, workspace, import_line, framework="vitest",
             n_draws=N_DRAWS, retries=3, tag="naive", forbidden=()):
    """(suite_files, record). Repair is build-errors-only; assertions may not weaken.

    The first element is a LIST of test-file texts -- one per surviving draw -- not one
    concatenated file. See `assemble`.

    `n_draws` defaults to the shared `assemble.N_DRAWS`, which is also
    `mutation.generate`'s `n_mutants` default: the two arms draw the same number of suites
    by construction, not by two independently-maintained literals.

    `import_line` is the exact import statement the suite must use to reach the code under
    test. It is passed in rather than derived, because only the adapter knows where the
    file lands in a staged workspace -- handing a model the target's path relative to the
    fixture makes it write an import that does not resolve, and every suite build-errors.

    `forbidden` arms spec §7.2: the caller passes the text that must NOT reach this prompt,
    and the composed prompt is checked before a single token is paid for. What belongs in
    it is decided by what the prompt legitimately contains -- P0's own sources, which is
    the whole point of the arm -- so the list must hold only text that is NOT in P0: the
    hidden oracle's suite -- which since spec §4.4 is the whole developer suite, and is
    therefore also ARM A's, so one entry covers both. Candidate sources cannot be passed
    and must not be: a candidate is mostly P0 verbatim, so any unchanged file would match
    itself and fire the gate on nothing, and at this point in the protocol no candidate
    exists yet (the freeze in `runner.freeze` enforces that ordering separately).
    """
    base = load("naive_testgen", sources=_bundle(sources),
                transformation=task.transformation, import_line=import_line,
                language=task.language.capitalize(), framework=framework)
    assert_no_leak(base, forbidden)

    kept, rejected = [], []
    for k in range(n_draws):
        # A distinct workspace per draw, mirroring ARM M's `p0_{k}`: a suite left behind
        # by the previous draw would otherwise be run again with the next one.
        p0_ws = adapter.stage(task.task_id, workspace / f"p0_{k}")
        # A distinct `ask` tag per draw. Same prompt, same model, same everything else --
        # the tag is the ONLY thing making this a fresh sample rather than a cache hit.
        suite, srec = _generate_with_repair(base, adapter, p0_ws, retries, f"{tag}_{k}")
        if not srec["compiles"]:
            rejected.append({"index": k, "why": "test never built"})
            continue
        # ARM M's filter is `kills(on_p0, on_mutant)`: green on P0 AND red on the mutant.
        # ARM N's is the same minus the second clause -- green on P0 -- so a draw that
        # builds but contradicts the trusted program is dropped by both arms alike. Its
        # cases would be inadmissible anyway, so keeping it could only add build risk to
        # the composite.
        if (srec["on_p0"] or {}).get("status") != "green":
            rejected.append({"index": k, "why": "test is not green on P0",
                             "on_p0": (srec["on_p0"] or {}).get("status")})
            continue
        kept.append({"index": k, "suite": suite})

    files, kept, res, meta = assemble(kept, task, adapter, workspace)
    return files, {"suite": files, "n_draws": n_draws, "n_kept": len(kept),
                        "kept": [k["index"] for k in kept],
                        # See `mutation.generate`: the surviving contributors WITH their
                        # text, so the caller can truncate both arms to a matched count
                        # and re-assemble through the same shared helper.
                        "contributors": [{"index": k["index"], "suite": k["suite"]}
                                         for k in kept],
                        "rejected": rejected,
                        "compiles": bool(kept),
                        "on_p0": res.as_row() if res else None,
                        "admissible": res.passed if res else [], **meta}


def _bundle(sources, max_chars=180_000):
    parts, used = [], 0
    for rel, text in sorted(sources.items()):
        if used + len(text) > max_chars:
            parts.append(f"### `{rel}` (omitted, prompt budget)\n")
            continue
        used += len(text)
        parts.append(f"### `{rel}`\n\n```\n{text}\n```\n")
    return "\n".join(parts)


def _generate_with_repair(base, adapter, workspace, retries, tag):
    attempts, user, suite = [], base, None
    for i in range(retries + 1):
        reply = ask(SYSTEM, user, effort="high",
                    max_tokens=32000 if i == 0 else 64000, tag=f"{tag}_{i}")
        suite = fenced(reply)
        if not suite:
            attempts.append({"attempt": i, "outcome": "no_code"})
            user = base + ("\n\nYour reply ran out of room before the code block. Return "
                           "only the test file, and if the suite would be long write the "
                           "12 highest-value tests rather than 25.")
            continue
        res = adapter.run_suite(workspace, suite, f"{tag}_{i}")
        attempts.append({"attempt": i, "outcome": res.status,
                         "n_passed": len(res.passed), "n_failed": len(res.failed)})
        if res.status != "build_error":
            return suite, {"suite": suite, "compiles": True, "attempts": attempts,
                           "on_p0": res.as_row(), "admissible": res.passed}
        user = base + "\n\n---\n\n" + load("repair", errors=res.out[-4000:], prev=suite)
    return suite, {"suite": suite, "compiles": False, "attempts": attempts,
                   "on_p0": None, "admissible": []}
