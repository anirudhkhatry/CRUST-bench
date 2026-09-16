"""Suite assembly, shared by ARM M and ARM N so "identical assembly" is structural.

Both arms draw several independently-generated suites. An arm's suite is the LIST of
survivors -- each stays its own test file, run by one vitest invocation over all of them
(`adapter.run_suites`).

IT USED TO BE A CONCATENATION, and that was the single largest source of data loss in the
experiment. Each contributor is a standalone test file opening with its own
`import React from 'react'`, its own `import { describe, it, expect } from 'vitest'` and
its own top-level helpers, so joining two is a duplicate-declaration error. The old code
absorbed that by dropping the newest contributor and retrying until the composite built --
which, measured across every run and fixture, meant keeping exactly one:
`dropped_for_collision` was `[3, 2, 1]` every single time. ARM M proved four killing tests
per fixture, all four survived the translate re-gate, and three were thrown away for a
syntax reason; its scored suite then held 1 leaf case against ARM N's 32-99. Separate
files have separate module scopes, so the collision does not exist and nothing is dropped.

This lives in its own module, not in `naive.py`, because `mutation.py` already imports
from `naive.py`; a shared helper in either arm would make the dependency asymmetric and,
if ARM N later needed anything of ARM M's, circular. Both arms import it.

`N_DRAWS` is the draw count for BOTH arms (spec: "ARM N is a matched control"). It is
defined here, once, so the two arms cannot drift apart by someone editing one default.
"""
N_DRAWS = 6


def assemble(kept, task, adapter, workspace, tag="composite"):
    """(files, kept, result, meta) -- one test file per contributor, validated on P0.

    `kept` is a list of {"index": int, "suite": str, ...}; it is not mutated. `files` is
    the contributors' texts in order, and `result` is the SuiteResult of running them all
    together against P0, so the caller can take its admissible set from the suite it will
    actually score. Case names in that result are file-index prefixed by `run_suites`.

    A contributor that will not LOAD in the multi-file run is dropped and the rest re-run.
    Every contributor was already verified green on P0 in isolation, so this should not
    fire; it is here because a file that cannot load contributes no admissible case while
    still making the whole arm's status structural, which would silently convert one bad
    draw into a whole arm's worth of missing evidence.
    """
    kept = list(kept)
    dropped, res = [], None
    while kept:
        res = adapter.run_suites(adapter.stage(task.task_id, workspace / tag),
                                 [k["suite"] for k in kept], tag)
        if not res.load_failures:
            break
        # Drop by INDEX INTO `kept`, highest first, so the surviving entries keep their
        # positions -- and with them the file indices the case names were built from.
        for i in sorted(res.load_failures, reverse=True):
            if i < len(kept):
                dropped.append(kept.pop(i)["index"])
    else:
        res = None      # every contributor was dropped: there is no suite to report
    files = [k["suite"] for k in kept]
    return files, kept, res, {"composite_status": res.status if kept else "empty",
                              "dropped_for_load_failure": dropped}
