"""Composite assembly, shared by ARM M and ARM N so "identical assembly" is structural.

Both arms draw several independently-generated suites and concatenate the survivors. Each
contributor built on its own; the CONCATENATION is a different program -- two files
declaring the same top-level identifier produce a composite that will not build -- so the
composite is validated against P0 and, if it collides, the most recent contributor is
dropped and the build retried rather than forfeiting the whole suite.

This lives in its own module, not in `naive.py`, because `mutation.py` already imports
from `naive.py`; a shared helper in either arm would make the dependency asymmetric and,
if ARM N later needed anything of ARM M's, circular. Both arms import it.

`N_DRAWS` is the draw count for BOTH arms (spec: "ARM N is a matched control"). It is
defined here, once, so the two arms cannot drift apart by someone editing one default.
"""
N_DRAWS = 6


def assemble(kept, task, adapter, workspace, tag="composite"):
    """(suite_text, kept, result, meta) -- concatenate, validate on P0, drop collisions.

    `kept` is a list of {"index": int, "suite": str, ...}; it is not mutated. Contributors
    are dropped from the END, newest first, which is the only ordering that keeps the
    earliest-proved tests: an arm that lost its first contributor to a collision caused by
    its last would be penalised for the order the model happened to emit.

    `result` is the SuiteResult of the composite that finally built (None if nothing did),
    so the caller can take its admissible set from the suite it will actually score.
    """
    kept, dropped, res = list(kept), [], None
    while kept:
        text = "\n\n".join(k["suite"] for k in kept)
        probe = adapter.stage(task.task_id, workspace / tag)
        res = adapter.run_suite(probe, text, tag)
        if res.status != "build_error":
            break
        dropped.append(kept.pop()["index"])
    else:
        res = None      # every contributor was dropped: there is no composite to report
    text = "\n\n".join(k["suite"] for k in kept) if kept else ""
    return text, kept, res, {"composite_status": res.status if kept else "empty",
                             "dropped_for_collision": dropped}
