from additional_datasets.arms import mutation, naive
from additional_datasets.prompts import load
from additional_datasets.spec import SuiteResult, Task


class FakeAdapter:
    """Records what it was asked to run, and replays scripted results."""

    def __init__(self, results):
        self.results, self.calls = list(results), []

    def stage(self, name, dest, sources=None):
        return dest

    def run_suite(self, workspace, suite_text, suite_name):
        self.calls.append((suite_name, suite_text))
        return self.results.pop(0)


def _task(tmp_path):
    return Task(task_id="t0", benchmark="refactorbench_js", language="javascript",
                repository="repo", source_commit="abc123",
                transformation="rename foo to bar", results_root=tmp_path)


def _script(monkeypatch, module, replies):
    """Record every ask() kwarg and return canned replies in order."""
    calls = []

    def fake_ask(system, user, **kw):
        calls.append({"system": system, "user": user, **kw})
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(module, "ask", fake_ask)
    return calls


SUITE_REPLY = "```javascript\ntest('a', () => { expect(1).toBe(1); });\n```"


def test_naive_prompt_carries_the_program_the_transformation_and_the_import():
    text = load("naive_testgen", sources="SRC_HERE", transformation="MIGRATE_HERE",
                import_line="import * as mod from './src/app/page.jsx';",
                language="JavaScript", framework="vitest")
    assert "SRC_HERE" in text and "MIGRATE_HERE" in text
    assert "./src/app/page.jsx" in text
    assert "<<" not in text


def test_mutation_prompt_carries_the_fault_the_transformation_and_the_import():
    # The transformation is here because ARM N's prompt has always carried it and ARM M's
    # did not -- an uncontrolled variable in the direction of withholding information from
    # ARM M. Both arms are now told a transformation is coming; only ARM M is told a fault.
    text = load("mutation_testgen", sources="SRC_HERE", semantic_change="FAULT_HERE",
                transformation="MIGRATE_HERE",
                import_line="import * as mod from './app/index.js';",
                language="JavaScript", framework="vitest")
    assert "SRC_HERE" in text and "FAULT_HERE" in text
    assert "MIGRATE_HERE" in text
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


def test_a_mutant_that_does_not_build_is_not_a_kill():
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    broken = SuiteResult(status="build_error", cases={}, rc=1, seconds=0.1)
    assert mutation.kills(green, broken) is False


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


def test_naive_generate_returns_the_suite_and_its_admissible_cases(tmp_path, monkeypatch):
    calls = _script(monkeypatch, naive, [SUITE_REPLY])
    adapter = FakeAdapter([SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)])
    suite, rec = naive.generate(_task(tmp_path), {"app/index.js": "export const x = 1;"},
                                adapter, tmp_path / "ws", "import * as mod from './app/index.js';")
    assert "expect(1).toBe(1)" in suite
    assert rec["compiles"] is True
    assert rec["admissible"] == ["a"]
    assert len(calls) == 1


def test_both_arms_reach_ask_with_identical_generation_parameters(tmp_path, monkeypatch):
    """The experiment's central control: the arms must differ ONLY in the kill gate.

    If this test fails, someone has changed one arm's model, effort, token budget or
    retry count without changing the other, and the benchmark is now measuring that
    change rather than the effect of mutation pressure.
    """
    sources = {"app/index.js": "export const x = 1;"}
    imp = "import * as mod from './app/index.js';"
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)

    # Both arms' test-writing step runs through `_generate_with_repair`, which is defined
    # in naive.py and so always resolves `ask` via naive.py's own globals -- mutation.py
    # only imports the function, it does not get its own binding of that call. So one
    # script on `naive`, given two replies, covers BOTH the naive-arm call below and the
    # test-writing call inside the later mutation.generate call. (Patching only
    # `mutation.ask` does not intercept that call at all: verified by instrumenting
    # naive.ask to raise and observing it still fires from inside mutation.generate.)
    # `mutation.ask` is patched separately, below, for the fault-proposal call only --
    # that step has no analogue in ARM N.
    gen_calls = _script(monkeypatch, naive, [SUITE_REPLY, SUITE_REPLY])
    naive.generate(_task(tmp_path), sources, FakeAdapter([green]), tmp_path / "n", imp)

    mutate_reply = ('```json\n[{"file": "app/index.js", "original_snippet": "export const x = 1;",'
                    ' "mutated_snippet": "export const x = 2;",'
                    ' "semantic_change": "x is 2", "fault_class": "return_value"}]\n```')
    _script(monkeypatch, mutation, [mutate_reply])
    # Adapter call order observed (empirically) in mutation.generate for one kept mutant:
    #   stage(p0) -> run_suite(p0, testgen build check) -> stage(mut) -> run_suite(mut)
    #   -> run_suite(p0, on_p0) -> [kept] -> stage(composite) -> run_suite(composite)
    # -- one more run_suite than the kill gate alone would need, because generate() also
    # build-validates the composite suite (Finding 2).
    mutation.generate(_task(tmp_path), sources,
                      FakeAdapter([green, red, green, green]),
                      tmp_path / "m", imp, n_mutants=1)

    # The test-writing call in each arm -- naive's only ask() call, and the second call
    # naive.ask recorded overall (the first was naive.generate's own).
    n_gen, m_gen = gen_calls[0], gen_calls[1]
    for key in ("effort", "max_tokens"):
        assert n_gen[key] == m_gen[key], f"arms diverged on {key}"
    assert n_gen.get("temperature") == m_gen.get("temperature")
    assert n_gen["system"] == m_gen["system"], "arms diverged on the system prompt"


def test_mutation_generate_keeps_only_tests_that_kill(tmp_path, monkeypatch):
    sources = {"app/index.js": "export const x = 1;"}
    mutate_reply = ('```json\n[{"file": "app/index.js", "original_snippet": "export const x = 1;",'
                    ' "mutated_snippet": "export const x = 2;",'
                    ' "semantic_change": "x is 2", "fault_class": "return_value"}]\n```')
    # `mutation.ask` covers the fault-proposal call; the test-writing call still resolves
    # via naive.ask (see the comment in test_both_arms_reach_ask_with_identical_...).
    _script(monkeypatch, mutation, [mutate_reply])
    _script(monkeypatch, naive, [SUITE_REPLY])
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    # green on P0 during generation, then GREEN on the mutant too -- no kill, so the
    # loop never reaches the composite-validation step (kept stays empty).
    adapter = FakeAdapter([green, green, green, green])
    suite, rec = mutation.generate(_task(tmp_path), sources, adapter, tmp_path / "m",
                                   "import * as mod from './app/index.js';", n_mutants=1)
    assert rec["n_kept"] == 0
    assert suite == ""
    assert rec["rejected"], "a surviving mutant must be recorded with a reason"
