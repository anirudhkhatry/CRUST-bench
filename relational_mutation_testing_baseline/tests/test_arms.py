import inspect

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
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    # One draw, then the composite validation -- the same two adapter calls ARM M makes
    # around one kept contributor.
    adapter = FakeAdapter([green, green])
    suite, rec = naive.generate(_task(tmp_path), {"app/index.js": "export const x = 1;"},
                                adapter, tmp_path / "ws",
                                "import * as mod from './app/index.js';", n_draws=1)
    assert "expect(1).toBe(1)" in suite
    assert rec["compiles"] is True
    # The admissible set now comes from the COMPOSITE run, not from the single draw.
    assert rec["admissible"] == ["a"]
    assert rec["n_draws"] == 1 and rec["n_kept"] == 1
    assert rec["composite_status"] == "green"
    assert rec["dropped_for_collision"] == []
    assert len(calls) == 1


def test_naive_draws_are_matched_to_arm_m_and_each_draw_gets_its_own_cache_tag(
        tmp_path, monkeypatch):
    """The matched control: `n_draws` really produces n draws, with n distinct ask tags.

    Fails on the pre-change single-draw ARM N (`len(calls) == 1`, one tag), and fails
    again if someone reuses one tag across draws -- `rmt.llm.ask` caches on
    `<tag>_<hash>.json`, so identical tags would return the same suite every time and
    collapse ARM N back to one effective draw while still charging for six.
    """
    calls = _script(monkeypatch, naive, [SUITE_REPLY])
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    adapter = FakeAdapter([green] * 4)      # 3 draws + 1 composite
    _, rec = naive.generate(_task(tmp_path), {"app/index.js": "export const x = 1;"},
                            adapter, tmp_path / "ws",
                            "import * as mod from './app/index.js';", n_draws=3)
    assert len(calls) == 3, "each draw must reach ask() once"
    tags = [c["tag"] for c in calls]
    assert len(set(tags)) == 3, f"draws shared a cache tag: {tags}"
    assert rec["n_kept"] == 3
    # Matched by construction: ARM N's default draw count IS ARM M's default mutant
    # count. Fails if either default is edited on its own (e.g. n_draws back to 1).
    assert (inspect.signature(naive.generate).parameters["n_draws"].default
            == inspect.signature(mutation.generate).parameters["n_mutants"].default)


def test_naive_drops_a_draw_that_is_not_green_on_p0(tmp_path, monkeypatch):
    """ARM N's filter is ARM M's minus the kill gate: green on P0, nothing else.

    Fails if the P0 gate is dropped, which would keep the `test_failed` draw: `n_kept`
    would be 2 and `rejected` empty.
    """
    _script(monkeypatch, naive, [SUITE_REPLY])
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    # draw 0 contradicts P0, draw 1 is green, then the composite.
    adapter = FakeAdapter([red, green, green])
    _, rec = naive.generate(_task(tmp_path), {"app/index.js": "export const x = 1;"},
                            adapter, tmp_path / "ws",
                            "import * as mod from './app/index.js';", n_draws=2)
    assert rec["n_kept"] == 1 and rec["kept"] == [1]
    assert [r["why"] for r in rec["rejected"]] == ["test is not green on P0"]


def test_naive_drops_the_newest_contributor_when_the_composite_collides(tmp_path,
                                                                       monkeypatch):
    """Identical assembly, exercised through ARM N: a composite that will not build loses
    its most recent contributor and is retried.

    Fails if the composite is not validated at all (`composite_status` would be the last
    draw's "green" and `dropped_for_collision` empty), or if a collision forfeited the
    whole suite (`n_kept` 0, empty text).
    """
    _script(monkeypatch, naive, [SUITE_REPLY])
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    broken = SuiteResult(status="build_error", cases={}, rc=1, seconds=0.1)
    # two green draws, a colliding composite, then a composite that builds
    adapter = FakeAdapter([green, green, broken, green])
    suite, rec = naive.generate(_task(tmp_path), {"app/index.js": "export const x = 1;"},
                               adapter, tmp_path / "ws",
                               "import * as mod from './app/index.js';", n_draws=2)
    assert rec["dropped_for_collision"] == [1]
    assert rec["n_kept"] == 1 and rec["composite_status"] == "green"
    assert suite.count("expect(1).toBe(1)") == 1


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
    # TWO draws per arm, not one: with n=1 the call-count assertion below could not fail
    # against the pre-change single-draw ARM N, which also made exactly one call.
    gen_calls = _script(monkeypatch, naive, [SUITE_REPLY] * 4)
    # 2 draws + 1 composite validation.
    naive.generate(_task(tmp_path), sources, FakeAdapter([green, green, green]),
                   tmp_path / "n", imp, n_draws=2)
    n_gen_calls = len(gen_calls)

    mutate_reply = (
        '```json\n[{"file": "app/index.js", "original_snippet": "export const x = 1;",'
        ' "mutated_snippet": "export const x = 2;",'
        ' "semantic_change": "x is 2", "fault_class": "return_value"},'
        ' {"file": "app/index.js", "original_snippet": "const x",'
        ' "mutated_snippet": "const y",'
        ' "semantic_change": "renamed", "fault_class": "naming"}]\n```')
    _script(monkeypatch, mutation, [mutate_reply])
    # Adapter call order in mutation.generate for one kept mutant, with the translate
    # stage in place:
    #   stage(p0) -> run_suite(p0, testgen build check) -> stage(mut) -> run_suite(mut)
    #   -> run_suite(p0, on_p0) -> run_suite(p0, translated) -> run_suite(mut, translated)
    #   -> [kept] -> stage(composite) -> run_suite(composite)
    # -- five per mutant, then one composite validation (generate() also build-validates
    # the composite suite, Finding 2).
    mutation.generate(_task(tmp_path), sources,
                      FakeAdapter([green, red, green, green, red] * 2 + [green]),
                      tmp_path / "m", imp, n_mutants=2, target="app/index.js")
    m_gen_calls = len(gen_calls) - n_gen_calls

    # The matched-control assertion (spec, "ARM N is a matched control"): for the same n,
    # both arms write the same number of SUITES, and ARM M writes exactly one extra call
    # per kept mutant for the translate stage (design decision 2026-09-15 -- see
    # mutation.translate). Fails on the pre-change ARM N, which drew once (1) against ARM
    # M's two; fails if either arm's per-draw retry budget is changed on its own; and
    # fails if the translate stage is silently dropped (m_gen_calls would fall to 2).
    assert n_gen_calls == 2, n_gen_calls
    assert m_gen_calls == 2 + 2, m_gen_calls   # 2 test-writing + 2 translate

    # The test-writing call in each arm -- naive's first draw, and the first call
    # mutation.generate made through naive.ask.
    n_gen, m_gen = gen_calls[0], gen_calls[n_gen_calls]
    for key in ("effort", "max_tokens"):
        assert n_gen[key] == m_gen[key], f"arms diverged on {key}"
    # `temperature` is deliberately not compared here: neither arm passes it, and
    # `rmt.llm.ask` discards it, so the comparison would be `None == None` no matter what
    # either arm's code does -- it can never fail and so proves nothing.
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


# --- the translate stage (design decision 2026-09-15, reversing spec §2) ---------------

TRANSLATED_REPLY = "```javascript\ntest('b', () => { expect(mod.f()).toBe(1); });\n```"


def test_public_surface_lists_exported_names_and_nothing_else():
    """Every export form the corpus uses, and no program text.

    Fails on an implementation that only handles `export function`/`export const` (it would
    miss `b` and `default`), on one that returns the LOCAL name of an aliased export (it
    would yield `a` instead of `b` -- a name no importer can reach), and on any
    implementation that returns source text rather than identifiers (the last assertion).
    """
    text = ("const a = 1;\n"
            "export function tick(n) { return n + 1; }\n"
            "export const SIZE = 12;\n"
            "export class Store {}\n"
            "export { a as b };\n"
            "export default function main() { return 'SECRET_BODY'; }\n")
    assert mutation.public_surface(text) == ["SIZE", "Store", "b", "default", "main",
                                            "tick"]
    # A name is an identifier, so nothing here can carry a literal out of the program.
    assert all(n.isidentifier() for n in mutation.public_surface(text))
    assert "SECRET_BODY" not in "".join(mutation.public_surface(text))


def test_a_file_with_no_exports_has_an_empty_surface():
    """Fails if the `export` regexes match a bare declaration -- which would manufacture a
    public surface for a module that exports nothing and let the translate step be handed
    an internal name."""
    assert mutation.public_surface("const a = 1;\nfunction f() {}\n") == []


def test_the_translate_prompt_carries_the_test_and_the_exports_but_no_program_text():
    """The leak argument, asserted: the translate prompt is built from the killing test and
    a list of identifiers, never from the sources.

    Fails the moment someone passes `_bundle(sources)` (or a candidate) into this prompt:
    `SECRET_PROGRAM_BODY` would appear in the text.
    """
    text = load("mutation_translate", test="KILLING_TEST_HERE", surface="- `tick`",
                transformation="MIGRATE_HERE",
                import_line="import * as mod from './app/index.js';",
                language="JavaScript", framework="vitest")
    assert "KILLING_TEST_HERE" in text and "tick" in text
    assert "MIGRATE_HERE" in text and "./app/index.js" in text
    assert "SECRET_PROGRAM_BODY" not in text
    assert "<<" not in text


def _mutation_with_translate(tmp_path, monkeypatch, adapter_results,
                             replies=(SUITE_REPLY, TRANSLATED_REPLY)):
    sources = {"app/index.js": "export const x = 1;\nexport function f() { return 1; }\n"}
    mutate_reply = ('```json\n[{"file": "app/index.js",'
                    ' "original_snippet": "export const x = 1;",'
                    ' "mutated_snippet": "export const x = 2;",'
                    ' "semantic_change": "x is 2", "fault_class": "return_value"}]\n```')
    _script(monkeypatch, mutation, [mutate_reply])
    _script(monkeypatch, naive, list(replies))
    return mutation.generate(_task(tmp_path), sources, FakeAdapter(adapter_results),
                             tmp_path / "m", "import * as mod from './app/index.js';",
                             n_mutants=1, target="app/index.js")


def test_a_translated_test_that_still_kills_joins_the_composite_in_translated_form(
        tmp_path, monkeypatch):
    """The contributor must be the TRANSLATED text, not the original killing test.

    Fails on an implementation that runs the translate call and then keeps `suite` anyway
    (the suite would contain `expect(1).toBe(1)` instead of `expect(mod.f()).toBe(1)`) --
    which is exactly how a translate stage gets silently neutered while still being paid
    for.
    """
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    # testgen-on-p0, on_mutant, on_p0, translated-on-p0, translated-on-mutant, composite
    suite, rec = _mutation_with_translate(tmp_path, monkeypatch,
                                          [green, red, green, green, red, green])
    assert rec["n_kept"] == 1
    assert "expect(mod.f()).toBe(1)" in suite
    assert "expect(1).toBe(1)" not in suite
    assert rec["translate_outcomes"] == {"ok": 1}
    assert rec["public_surface"] == ["f", "x"]


def test_a_translation_that_stops_killing_is_dropped_as_F4_not_discriminating(
        tmp_path, monkeypatch):
    """The CRUST-bench outcome this stage exists to expose.

    Fails on an implementation that skips the post-translate re-gate: `n_kept` would be 1
    and the arm would bank a kill for a test that no longer separates the two programs.
    """
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    # ...translated is green on P0 AND green on the mutant: it no longer discriminates.
    suite, rec = _mutation_with_translate(tmp_path, monkeypatch,
                                          [green, red, green, green, green])
    assert rec["n_kept"] == 0 and suite == ""
    assert rec["translate_outcomes"] == {"F4_not_discriminating": 1}
    assert rec["rejected"][0]["why"] == "translation: F4_not_discriminating"


def test_a_translation_that_stops_passing_p0_is_dropped_as_F3_unsound_on_p0(
        tmp_path, monkeypatch):
    """Fails on an implementation that only re-checks the mutant: a translated test that
    contradicts the trusted program would be kept, and would then reject every candidate
    including the correct ones."""
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    # ...translated is RED on P0.
    suite, rec = _mutation_with_translate(tmp_path, monkeypatch,
                                          [green, red, green, red])
    assert rec["n_kept"] == 0 and suite == ""
    assert rec["translate_outcomes"] == {"F3_unsound_on_p0": 1}


def test_a_translation_that_never_builds_is_dropped_as_F2(tmp_path, monkeypatch):
    """Repair is build-errors-only, so a translation that build-errors through the whole
    repair budget is dropped rather than repaired into something weaker.

    Fails if a build-erroring translation is treated as a pass (n_kept would be 1).
    """
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    broken = SuiteResult(status="build_error", cases={}, rc=1, seconds=0.1)
    suite, rec = _mutation_with_translate(
        tmp_path, monkeypatch, [green, red, green] + [broken] * 4)
    assert rec["n_kept"] == 0 and suite == ""
    assert rec["translate_outcomes"] == {"F2_translate_no_build": 1}


def test_both_arms_report_their_contributors_with_text_for_matched_truncation(
        tmp_path, monkeypatch):
    """The caller can only match test counts if both arms hand back their kept suites.

    Fails if either arm stops recording `contributors`, or records indices without the
    text -- the matched arms in `run_task` would then be assembled from nothing.
    """
    _script(monkeypatch, naive, [SUITE_REPLY])
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    _, rec = naive.generate(_task(tmp_path), {"app/index.js": "export const x = 1;"},
                            adapter := FakeAdapter([green] * 3), tmp_path / "ws",
                            "import * as mod from './app/index.js';", n_draws=2)
    assert [c["index"] for c in rec["contributors"]] == [0, 1]
    assert all("expect(1).toBe(1)" in c["suite"] for c in rec["contributors"])
    assert adapter.calls


def test_more_mutants_than_requested_are_discarded(tmp_path, monkeypatch):
    """The model decides how long its JSON array is; the matched control decides how many
    draws ARM M gets.

    With `n_mutants=1` and a reply carrying two faults, an unclamped implementation makes
    TWO test-writing calls -- one more draw than ARM N is given for the same n, and one
    more paid call per extra fault. Fails on that implementation: `len(gen)` would be 2.
    """
    sources = {"app/index.js": "export const x = 1;\nexport const y = 2;\n"}
    mutate_reply = ('```json\n[{"file": "app/index.js",'
                    ' "original_snippet": "export const x = 1;",'
                    ' "mutated_snippet": "export const x = 2;",'
                    ' "semantic_change": "x is 2", "fault_class": "a"},'
                    ' {"file": "app/index.js", "original_snippet": "export const y = 2;",'
                    ' "mutated_snippet": "export const y = 3;",'
                    ' "semantic_change": "y is 3", "fault_class": "b"}]\n```')
    _script(monkeypatch, mutation, [mutate_reply])
    gen = _script(monkeypatch, naive, [SUITE_REPLY, TRANSLATED_REPLY])
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    _, rec = mutation.generate(_task(tmp_path), sources,
                               FakeAdapter([green, red, green, green, red, green]),
                               tmp_path / "m", "import * as mod from './app/index.js';",
                               n_mutants=1, target="app/index.js")
    assert rec["n_mutants"] == 1
    assert rec["n_mutants_requested"] == 1
    assert len(gen) == 2, "one testgen call + one translate call, for ONE mutant"
