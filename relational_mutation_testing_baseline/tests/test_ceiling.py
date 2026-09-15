from additional_datasets.arms import ceiling
from additional_datasets.spec import SuiteResult


def test_diff_of_renders_a_unified_diff_of_the_changed_file():
    good = {"a.js": "const x = 1;\n", "b.js": "const y = 2;\n"}
    bad = {"a.js": "const x = 2;\n", "b.js": "const y = 2;\n"}
    d = ceiling.diff_of(good, bad)
    assert "a.js" in d
    assert "-const x = 1;" in d and "+const x = 2;" in d
    assert "b.js" not in d          # unchanged files are not in the diff


def test_diff_of_covers_a_file_the_fault_added():
    d = ceiling.diff_of({"a.js": "x\n"}, {"a.js": "x\n", "new.js": "y\n"})
    assert "new.js" in d


def test_unreachable_is_detected_from_the_first_line_marker():
    assert ceiling.is_unreachable("// UNREACHABLE: no public accessor\ntest('x', () => {});")
    assert not ceiling.is_unreachable("test('x', () => {});")


def test_outcome_names_why_a_ceiling_attempt_failed():
    green = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=0.1)
    red = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=0.1)
    assert ceiling.outcome(green, red) == "KILLED"
    assert ceiling.outcome(red, red) == "F_UNSOUND_ON_GOOD"
    assert ceiling.outcome(green, green) == "F_NOT_DISCRIMINATING"
