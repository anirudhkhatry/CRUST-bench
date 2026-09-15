import subprocess
import sys


def test_cli_lists_the_pilot_corpus():
    p = subprocess.run([sys.executable, "run_additional_datasets.py", "select",
                        "--benchmark", "refactorbench_js"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "binary_search" in p.stdout
    assert "18" in p.stdout


def test_cli_rejects_an_unknown_benchmark():
    p = subprocess.run([sys.executable, "run_additional_datasets.py", "select",
                        "--benchmark", "nope"], capture_output=True, text=True)
    assert p.returncode != 0
