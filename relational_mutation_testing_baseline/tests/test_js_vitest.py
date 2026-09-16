import pytest

from additional_datasets.adapters import js_vitest as JS

pytestmark = pytest.mark.skipif(not JS.FIXTURES.is_dir(),
                                reason="RefactorBench-JS corpus not cloned")


def test_parse_vitest_json_reads_case_names_and_statuses():
    # Real vitest `fullName` values are the describe/test titles joined by a SPACE, e.g.
    # "Binary Search binarySearchRecursive should find existing element" -- verified
    # against a real run. The parser passes the string through untouched, so the exact
    # separator does not matter to it; these names are shaped like the real ones.
    blob = {"testResults": [{"assertionResults": [
        {"fullName": "Binary Search search finds an element", "status": "passed"},
        {"fullName": "Binary Search search misses it", "status": "failed"},
        {"fullName": "Binary Search search skipped one", "status": "skipped"}]}]}
    status, cases = JS.parse_vitest_json(blob)
    assert status == "test_failed"
    assert cases["Binary Search search finds an element"] == "ok"
    assert cases["Binary Search search misses it"] == "FAILED"
    # A skipped case is masked out by the oracle split and must not be scored either way.
    assert "Binary Search search skipped one" not in cases


def test_parse_vitest_json_green_when_nothing_failed():
    blob = {"testResults": [{"assertionResults": [
        {"fullName": "a", "status": "passed"}]}]}
    assert JS.parse_vitest_json(blob)[0] == "green"


def test_parse_vitest_json_no_tests_when_everything_is_skipped():
    blob = {"testResults": [{"assertionResults": [
        {"fullName": "a", "status": "skipped"}]}]}
    assert JS.parse_vitest_json(blob)[0] == "no_tests"


def test_parse_vitest_json_reports_a_load_failure_as_build_error():
    # A file that will not load reports status "failed" with ZERO assertions. That is a
    # structural rejection, not "this suite has no tests" -- conflating them corrupts the
    # structural-vs-behavioural split the design depends on.
    blob = {"testResults": [{"status": "failed", "assertionResults": [],
                             "message": 'Failed to resolve import "./app/index.js"'}]}
    assert JS.parse_vitest_json(blob)[0] == "build_error"


def test_src_prefix_detects_both_corpus_layouts():
    # Measured: 7 of the 18 pure-logic fixtures are flat, 4 reach sources through src/.
    assert JS.src_prefix("binary_search") == ""
    assert JS.src_prefix("seo_content_data") == "src"


def test_src_prefix_matches_a_dynamic_only_import():
    # basic_web_with_auth reaches its target only through `await import('./src/auth.js')`.
    # A `from`-only pattern classifies it as flat and then fails to resolve every import.
    assert "import(" in JS.holdout_text("basic_web_with_auth")
    assert JS.src_prefix("basic_web_with_auth") == "src"


def test_import_path_carries_the_prefix():
    # Handing an arm `targetFile` alone yields './app/page.jsx' for an src-prefixed
    # fixture, which never resolves. The arms are told this string verbatim.
    assert JS.import_path("binary_search") == "./app/index.js"
    assert JS.import_path("seo_content_data").startswith("./src/")


def test_stage_records_the_prefix_for_run_suite(tmp_path):
    ws = JS.stage("seo_content_data", tmp_path / "ws")
    assert JS.staged_prefix(ws) == "src"
    assert (ws / "src").is_dir()
    flat = JS.stage("binary_search", tmp_path / "flat")
    assert JS.staged_prefix(flat) == ""
    assert (flat / "app").is_dir()


def test_pure_logic_fixtures_is_the_18_fixture_pilot_corpus():
    # 18 is the SYNTACTIC count -- fixtures whose holdout needs no DOM. Only 11 of them
    # actually verify end to end (holdout green on P0, split disjoint and complete); the
    # other 7 are corpus defects, and Task 7's verification is what discovers that.
    # Do not hardcode 11 here; this function does not run anything.
    names = JS.pure_logic_fixtures()
    assert len(names) == 18
    assert "binary_search" in names and "merge_sort" in names


def test_fixture_config_names_the_target_and_the_holdout_suite():
    cfg = JS.fixture_config("binary_search")
    assert cfg["targetFile"] == "app/index.js"
    assert cfg["testFile"].endswith(".test.js")


def test_the_holdout_suite_runs_green_against_the_untouched_fixture(tmp_path):
    """The toolchain smoke test from spec §8 step 1. If this fails, stop.

    Verified by hand before this plan was written: binary_search reports exactly 19
    cases, all green. A `build_error` here with "Failed to resolve import './app/index.js'"
    means `stage` nested the sources under `src/` instead of putting their contents at the
    workspace root — re-read `stage`'s docstring.
    """
    ws = JS.stage("binary_search", tmp_path / "ws")
    suite = (JS.FIXTURES / "binary_search" / JS.fixture_config("binary_search")["testFile"]).read_text()
    res = JS.run_suite(ws, suite, "oracle")
    assert res.status == "green", res.out[-2000:]
    assert len(res.cases) == 19


def test_two_suites_can_run_against_one_workspace_without_interference(tmp_path):
    """Task 7 stages once and runs several suites against that workspace."""
    ws = JS.stage("binary_search", tmp_path / "ws")
    whole = JS.holdout_text("binary_search")
    first = JS.run_suite(ws, whole, "whole")
    assert first.status == "green" and len(first.cases) == 19

    tiny = ("import { describe, test, expect } from 'vitest';\n"
            "import { binarySearchIterative } from './app/index.js';\n"
            "describe('tiny', () => {\n"
            "  test('finds it', () => { expect(binarySearchIterative([1, 3, 5], 3)).toBe(1); });\n"
            "});\n")
    second = JS.run_suite(ws, tiny, "tiny")
    # The second run must see ONLY its own case, not the 19 from the first.
    assert second.status == "green", second.out[-1500:]
    assert len(second.cases) == 1


def test_run_suite_leaves_no_files_behind(tmp_path):
    ws = JS.stage("binary_search", tmp_path / "ws")
    JS.run_suite(ws, JS.holdout_text("binary_search"), "whole")
    assert not list(ws.glob("*.test.jsx"))
    assert not list(ws.glob("*.results.json"))


def test_run_suites_runs_independent_files_that_could_never_be_concatenated(tmp_path):
    """The reason an arm's suite is a file LIST. Real vitest, not a fake.

    Both files declare a top-level `const H` -- with DIFFERENT values -- and both name a
    case `alpha`. Concatenated, that is a redeclaration error, which is exactly why the
    old composite kept only one contributor per arm (`dropped_for_collision == [3, 2, 1]`
    in every fixture measured). As separate files they both run, and the identical case
    titles stay distinct because `run_suites` prefixes every name with its file index.
    """
    ws = JS.stage("binary_search", tmp_path / "ws")
    imp = JS.import_path("binary_search")
    head = f"import {{ test, expect }} from 'vitest';\nimport * as mod from '{imp}';\n"
    a = head + "const H = 1;\ntest('alpha', () => { expect(H).toBe(1); });\n"
    b = (head + "const H = 2;\ntest('alpha', () => { expect(H).toBe(2); });\n"
         "test('beta', () => { expect(2).toBe(3); });\n")
    res = JS.run_suites(ws, [a, b], "demo")
    assert res.status == "test_failed"
    assert res.cases == {"0::alpha": "ok", "1::alpha": "ok", "1::beta": "FAILED"}
    assert res.load_failures == ()


def test_run_suites_reports_which_file_failed_to_load_and_keeps_the_others_cases(tmp_path):
    """A load failure is a STRUCTURAL claim and it is per FILE, so it cannot be read off
    `cases` -- the file's cases are simply absent. Fails on an implementation that reports
    only an overall status: the surviving file's passing case would be indistinguishable
    from a suite that ran clean, and `score.verdict` would call the missing cases a
    mid-run crash (behavioural) instead of a shape difference (structural).
    """
    ws = JS.stage("binary_search", tmp_path / "ws")
    imp = JS.import_path("binary_search")
    good = (f"import {{ test, expect }} from 'vitest';\nimport * as mod from '{imp}';\n"
            "test('alpha', () => { expect(1).toBe(1); });\n")
    unresolvable = "import * as nope from './does-not-exist.js';\ntest('x', () => {});\n"
    res = JS.run_suites(ws, [good, unresolvable], "demo2")
    assert res.load_failures == (1,)
    assert res.status == "build_error"
    assert res.cases == {"0::alpha": "ok"}
