"""RefactorBench-JS access and execution.

The benchmark ships fixtures and a holdout test suite but no package.json -- its
HARNESS_INTERFACE.md hands package installation and test execution to the evaluator. So
there is one shared vitest install under `js_runtime/`, reused by every fixture, and each
run stages a throwaway copy of the fixture so the corpus is never mutated.
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..spec import SuiteResult

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench/datasets/refactor-bench")
FIXTURES = ROOT / "refactoring/data"
RUNTIME = Path(__file__).resolve().parent / "js_runtime"
VITEST = RUNTIME / "node_modules/.bin/vitest"
CONFIG = RUNTIME / "vitest.config.mjs"
TIMEOUT = 300


def fixture_config(name):
    return json.loads((FIXTURES / name / "refactoring_eval.config.json").read_text())


def holdout_text(name):
    return (FIXTURES / name / fixture_config(name)["testFile"]).read_text()


def pure_logic_fixtures():
    """Fixtures whose holdout suite needs no DOM -- the spec §4.3 pilot corpus."""
    out = []
    for d in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
        tests = sorted(d.glob("*.test.js")) + sorted(d.glob("*.test.jsx"))
        if not tests:
            continue
        text = tests[0].read_text()
        if "@testing-library/react" in text or "react-native" in text:
            continue
        out.append(d.name)
    return out


SRC_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*['"]\./src/""")
PREFIX_MARKER = ".ad_src_prefix"


def src_prefix(name):
    """"src" if this fixture's holdout reaches sources through src/, else "" (flat).

    The corpus is not uniform and there is no metadata field for this, so it is read off
    the holdout's own imports. The pattern deliberately matches DYNAMIC imports too
    (`await import('./src/auth.js')`) -- basic_web_with_auth has no static import of its
    target at all, and a `from`-only pattern silently classifies it as flat and then fails
    to resolve every import.
    """
    return "src" if SRC_IMPORT.search(holdout_text(name)) else ""


def import_path(name):
    """How a suite at the workspace root reaches the target file, prefix included.

    The arms must be told this verbatim. Handing a model `targetFile` alone makes it write
    `./app/page.jsx` for an src-prefixed fixture, which never resolves.
    """
    pre = src_prefix(name)
    rel = fixture_config(name)["targetFile"]
    return f"./{pre}/{rel}" if pre else f"./{rel}"


def stage(name, dest, sources=None):
    """Copy a fixture into a writable workspace. `sources` overrides files by relpath.

    Where `src/` lands depends on the fixture: flat fixtures get its CONTENTS at the
    workspace root (their holdout imports `./app/index.js`, so `app/` must be a sibling of
    the test file); src-prefixed fixtures keep it at `src/`. Override paths are always
    relative to `src/`, matching `targetFile` in `refactoring_eval.config.json`.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    pre = src_prefix(name)
    target = dest / pre if pre else dest
    shutil.copytree(FIXTURES / name / "src", target, dirs_exist_ok=True)
    for rel, text in (sources or {}).items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    # The prefix is a property of the staged workspace, so record it here rather than
    # threading the fixture name through run_suite. That keeps the adapter protocol
    # -- stage(name, dest, sources) / run_suite(workspace, text, suite_name) -- identical
    # for every benchmark, which the Java adapter also has to satisfy.
    (dest / PREFIX_MARKER).write_text(pre)
    # Bare specifiers -- `react`, `react/jsx-dev-runtime`, `@testing-library/react`,
    # `vitest` -- are resolved by Node walking UP from the importing file, so a workspace
    # staged outside the runtime directory resolves none of them. The automatic JSX
    # runtime needs `react/jsx-dev-runtime` in particular, so without this every React
    # fixture dies with "Failed to resolve import" and reports zero cases -- silently
    # excluding exactly the complex subjects the experiment needs. A symlink covers every
    # bare specifier at once rather than aliasing them one by one in the vitest config.
    link = dest / "node_modules"
    if not link.exists():
        link.symlink_to(RUNTIME / "node_modules", target_is_directory=True)
    return dest


def staged_prefix(workspace):
    """The source prefix recorded by `stage` for this workspace."""
    marker = Path(workspace) / PREFIX_MARKER
    return marker.read_text().strip() if marker.exists() else ""


def parse_vitest_json(blob):
    """(status, {case name: "ok"|"FAILED"}). Skipped cases are not scored either way.

    A file that fails to LOAD reports `status: "failed"` with zero assertionResults. That
    is a build error -- a structural rejection -- not "this suite has no tests", which is
    unscorable. Conflating them corrupts the structural-vs-behavioural split the design
    requires, and it is the most likely misreading of this report format.
    """
    cases = {}
    files = blob.get("testResults") or []
    for f in files:
        for a in f.get("assertionResults") or []:
            name = a.get("fullName") or a.get("title") or ""
            st = a.get("status")
            if st == "passed":
                cases[name] = "ok"
            elif st == "failed":
                cases[name] = "FAILED"
    if not cases:
        file_failed = any(f.get("status") == "failed" for f in files)
        return ("build_error" if file_failed else "no_tests"), cases
    return ("test_failed" if any(v == "FAILED" for v in cases.values()) else "green"), cases


FILE_INDEX = re.compile(r"__ad(\d+)\.test\.jsx$")


def parse_vitest_json_multi(blob, n_files):
    """(status, cases, load_failures) for a MULTI-FILE run. Case keys are `"<i>::<name>"`.

    Every case name is prefixed with its FILE INDEX, unconditionally. Two independently
    generated suites routinely choose the same `describe`/`it` titles, and vitest reports a
    case by title alone, so an unprefixed merge would silently collapse two distinct tests
    into one -- losing a failure whenever the same-named case passed in another file.
    Prefixing only on collision would be worse: the name a case gets would then depend on
    which OTHER files loaded, so a case's identity would change between the P0 run and the
    candidate run and the admissible set would stop lining up.

    A file that reports `status: "failed"` with zero assertionResults did not run: it failed
    to LOAD. Its index goes to `load_failures` and its cases are simply absent, which is why
    that fact has to be returned rather than inferred from `cases`.
    """
    cases, loaded, load_failures = {}, set(), []
    for f in blob.get("testResults") or []:
        m = FILE_INDEX.search(f.get("name") or "")
        i = int(m.group(1)) if m else 0
        ran = False
        for a in f.get("assertionResults") or []:
            name = a.get("fullName") or a.get("title") or ""
            st = a.get("status")
            if st in ("passed", "failed"):
                cases[f"{i}::{name}"] = "ok" if st == "passed" else "FAILED"
            # A SKIPPED case is not scored either way -- but it is proof the file LOADED.
            # `cap_cases_files` skip-masks whole files when the leaf-case budget is spent
            # before reaching them, and vitest reports such a file as `status: "passed"`
            # with every assertionResult `"skipped"`. Requiring a passed-or-failed case
            # here therefore recorded a deliberately masked file as a file that would not
            # load -- which `verdict` reads as a STRUCTURAL rejection of the candidate.
            # Measured on collaborative_doc_editor: it rejected all 4 GOOD candidates and
            # put ARM N's matched GAR at 0%, an artefact that looked exactly like a
            # finding about naive tests.
            ran = True
        if ran or f.get("status") != "failed":
            loaded.add(i)
        else:
            load_failures.append(i)
    # A file vitest never mentioned at all is also a file that did not run. Counting only
    # the ones it named "failed" would let a collection-time crash pass for "no tests".
    load_failures += [i for i in range(n_files)
                      if i not in loaded and i not in load_failures]
    load_failures = tuple(sorted(set(load_failures)))
    # A real assertion failure is reported even when a SIBLING file would not load, so
    # behavioural evidence takes precedence over the structural claim; `load_failures`
    # carries the structural fact alongside it either way.
    if any(v == "FAILED" for v in cases.values()):
        return "test_failed", cases, load_failures
    if load_failures:
        return "build_error", cases, load_failures
    return ("green" if cases else "no_tests"), cases, load_failures


def run_suites(workspace, suite_texts, suite_name):
    """Run several INDEPENDENT suite files in one vitest invocation.

    An arm's suite is a LIST of files, never one concatenated file. Concatenation is not
    sound: each contributor is a standalone test file with its own `import React from
    'react'` and its own top-level helpers, so joining two is a duplicate-declaration
    error. The composite-plus-drop-on-collision scheme that preceded this kept exactly ONE
    contributor in every arm of every fixture measured (`dropped_for_collision` was
    `[3, 2, 1]` every time), discarding three proven killing tests per fixture for a syntax
    reason. Separate files have separate module scopes, so nothing has to be dropped and no
    model output is rewritten.

    One vitest process runs them all, so this costs no more than the composite did.
    """
    workspace = Path(workspace)
    files = [workspace / f"{suite_name}__ad{i}.test.jsx"
             for i in range(len(suite_texts))]
    for p, text in zip(files, suite_texts):
        p.write_text(text)
    out_file = workspace / f"{suite_name}.results.json"
    out_file.unlink(missing_ok=True)
    env = {**os.environ, "AD_WORKSPACE": str(workspace), "CI": "1",
           "AD_SRC_PREFIX": staged_prefix(workspace)}
    t0 = time.time()
    try:
        if not suite_texts:
            return SuiteResult(status="no_tests", cases={}, rc=0, seconds=0.0)
        try:
            p = subprocess.run(
                [str(VITEST), "run", "--config", str(CONFIG), "--reporter=json",
                 f"--outputFile={out_file}", *(f.name for f in files)],
                cwd=workspace, env=env, capture_output=True, text=True, timeout=TIMEOUT)
            rc, out = p.returncode, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            return SuiteResult(status="timeout", cases={}, rc=124,
                               seconds=round(time.time() - t0, 1), out=str(e)[-3000:])
        secs = round(time.time() - t0, 1)
        if not out_file.exists():
            return SuiteResult(status="build_error", cases={}, rc=rc, seconds=secs,
                               out=out[-4000:],
                               load_failures=tuple(range(len(suite_texts))))
        status, cases, lf = parse_vitest_json_multi(json.loads(out_file.read_text()),
                                                    len(suite_texts))
        return SuiteResult(status=status, cases=cases, rc=rc, seconds=secs,
                           out=out[-4000:], load_failures=lf)
    finally:
        for f in files:
            f.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)


def run_suite(workspace, suite_text, suite_name):
    """Run one suite against a staged workspace. The suite is written in, then removed.

    Single-suite runs keep UNPREFIXED case names, so the arms that compare case names
    across separate invocations of this function -- ARM C's good/bad pair, the oracle's
    admissible set and a candidate's label -- keep working unchanged. `run_suites` is a
    different namespace on purpose; do not mix results from the two.
    """
    workspace = Path(workspace)
    test_file = workspace / f"{suite_name}.test.jsx"
    test_file.write_text(suite_text)
    out_file = workspace / f"{suite_name}.results.json"
    # A stale report from an earlier run of the same suite_name would be indistinguishable
    # from this run's output: the failure check below is `not out_file.exists()`, so a
    # crashed vitest would silently return the PREVIOUS run's result as if it were current.
    out_file.unlink(missing_ok=True)
    env = {**os.environ, "AD_WORKSPACE": str(workspace), "CI": "1",
           "AD_SRC_PREFIX": staged_prefix(workspace)}
    t0 = time.time()
    try:
        try:
            p = subprocess.run(
                [str(VITEST), "run", "--config", str(CONFIG), "--reporter=json",
                 f"--outputFile={out_file}", test_file.name],
                cwd=workspace, env=env, capture_output=True, text=True, timeout=TIMEOUT)
            rc, out = p.returncode, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            return SuiteResult(status="timeout", cases={}, rc=124,
                               seconds=round(time.time() - t0, 1), out=str(e)[-3000:])
        secs = round(time.time() - t0, 1)
        if not out_file.exists():
            # vitest never produced a report: the suite did not build or load.
            return SuiteResult(status="build_error", cases={}, rc=rc, seconds=secs,
                               out=out[-4000:])
        status, cases = parse_vitest_json(json.loads(out_file.read_text()))
        return SuiteResult(status=status, cases=cases, rc=rc, seconds=secs, out=out[-4000:])
    finally:
        # The suite is written in, then removed -- both the test file and its report --
        # so a later run_suite call against the same workspace never sees this call's
        # leftovers, and so repeated calls with the same suite_name (Task 7 stages once
        # and runs "whole"/"oracle"/"existing" against it) do not collide.
        test_file.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)
