from additional_datasets.arms import mutation, naive
from additional_datasets.prompts import load
from additional_datasets.spec import SuiteResult


class FakeAdapter:
    """Records what it was asked to run, and replays scripted results."""

    def __init__(self, results):
        self.results, self.calls = list(results), []

    def stage(self, name, dest, sources=None):
        return dest

    def run_suite(self, workspace, suite_text, suite_name):
        self.calls.append((suite_name, suite_text))
        return self.results.pop(0)


def test_naive_prompt_carries_the_program_the_transformation_and_the_import():
    text = load("naive_testgen", sources="SRC_HERE", transformation="MIGRATE_HERE",
                import_line="import * as mod from './src/app/page.jsx';",
                language="JavaScript", framework="vitest")
    assert "SRC_HERE" in text and "MIGRATE_HERE" in text
    assert "./src/app/page.jsx" in text
    assert "<<" not in text


def test_mutation_prompt_carries_the_fault_and_the_import():
    text = load("mutation_testgen", sources="SRC_HERE", semantic_change="FAULT_HERE",
                import_line="import * as mod from './app/index.js';",
                language="JavaScript", framework="vitest")
    assert "SRC_HERE" in text and "FAULT_HERE" in text
    assert "./app/index.js" in text
    assert "<<" not in text


def test_mutation_kill_gate_keeps_a_test_that_separates_p0_from_the_mutant():
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    assert mutation.kills(green, red) is True


def test_mutation_kill_gate_rejects_a_test_that_fails_p0():
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    assert mutation.kills(red, red) is False


def test_mutation_kill_gate_rejects_a_test_that_passes_both():
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    assert mutation.kills(green, green) is False


def test_apply_mutant_requires_a_unique_snippet():
    sources = {"app/index.js": "const a = 1;\nconst b = 2;\n"}
    out = mutation.apply_mutant(sources, {"file": "app/index.js",
                                          "original_snippet": "const a = 1;",
                                          "mutated_snippet": "const a = 2;"})
    assert out["app/index.js"].startswith("const a = 2;")

    assert mutation.apply_mutant(sources, {"file": "app/index.js",
                                           "original_snippet": "const",
                                           "mutated_snippet": "let"}) is None
    assert mutation.apply_mutant(sources, {"file": "nope.js",
                                           "original_snippet": "x",
                                           "mutated_snippet": "y"}) is None
