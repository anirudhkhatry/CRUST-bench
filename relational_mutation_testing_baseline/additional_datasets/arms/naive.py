"""ARM N -- the naive baseline: tests written from P0 and the migration spec, no mutant.

This is the arm the pipeline's claim is measured against, so everything it is denied must
be denied for a reason. It gets the same model, the same effort, the same repair budget
and the same output format as ARM M. The single difference is that no mutant exists to
force its assertions to prove they discriminate.
"""
from rmt.llm import ask, fenced

from ..prompts import load
from ..runner import assert_no_leak

SYSTEM = ("You are a precise programmer writing tests. You assert exact values, you use "
          "only APIs you have been shown, and you never write a test that could fail a "
          "correct implementation.")


def generate(task, sources, adapter, workspace, import_line, framework="vitest",
             retries=3, tag="naive", forbidden=()):
    """(suite_text, record). Repair is build-errors-only; assertions may not weaken.

    `import_line` is the exact import statement the suite must use to reach the code under
    test. It is passed in rather than derived, because only the adapter knows where the
    file lands in a staged workspace -- handing a model the target's path relative to the
    fixture makes it write an import that does not resolve, and every suite build-errors.

    `forbidden` arms spec §7.2: the caller passes the text that must NOT reach this prompt,
    and the composed prompt is checked before a single token is paid for. What belongs in
    it is decided by what the prompt legitimately contains -- P0's own sources, which is
    the whole point of the arm -- so the list must hold only text that is NOT in P0: the
    hidden oracle's suite and Baseline A's half of it. Candidate sources cannot be passed
    and must not be: a candidate is mostly P0 verbatim, so any unchanged file would match
    itself and fire the gate on nothing, and at this point in the protocol no candidate
    exists yet (the freeze in `runner.freeze` enforces that ordering separately).
    """
    base = load("naive_testgen", sources=_bundle(sources),
                transformation=task.transformation, import_line=import_line,
                language=task.language.capitalize(), framework=framework)
    assert_no_leak(base, forbidden)
    return _generate_with_repair(base, adapter, workspace, retries, tag)


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
