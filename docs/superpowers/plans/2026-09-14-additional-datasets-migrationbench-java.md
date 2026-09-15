# additional_datasets: MigrationBench Java 8→17 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the MigrationBench Java 8→17 benchmark to the existing `additional_datasets` harness, producing GOOD and BAD compiling migrations scored by the same four arms.

**Architecture:** This plan adds **one adapter and one benchmark module**. `spec.py`, `score.py`, `runner.py`, `seed.py`, `harvest.py` and all four arms are already built and language-agnostic; nothing in them changes. The new code is a conda-provisioned two-JDK Maven adapter, a phase-0 executability ladder, and the Java wiring.

**Tech Stack:** conda env with `openjdk=8`, `openjdk=17`, `maven=3.9`; Python 3.12 (`.venv/bin/python`); JUnit via the repository's own Maven Surefire setup.

**Spec:** `docs/superpowers/specs/2026-09-14-additional-datasets-design.md` (§3 and §3.6 in particular)

**Prerequisite:** `docs/superpowers/plans/2026-09-14-additional-datasets-harness-and-js.md` must be complete. Tasks here import `additional_datasets.spec`, `.runner`, `.score`, `.seed`, `.harvest` and `.arms` and assume their tests pass.

## Global Constraints

- **Working dir** is `relational_mutation_testing_baseline/`. `docs/` and `datasets/` are relative to the repo root `/nas/CRUST-bench-COLM/CRUST-bench`.
- **Python is `.venv/bin/python`.** Java and Maven come from a conda env, never from `PATH` — this machine has no system `java`, `javac` or `mvn`.
- **Two JDKs coexist**, selected per invocation by `JAVA_HOME`. P0 builds under 8; every candidate builds under 17. The adapter owns this; nothing else learns about it.
- **A BAD candidate must compile under 17** and **the hidden oracle must confirm it behaves differently** (spec §3.5). A candidate that does not compile is `EXCLUDE`, never BAD (spec §20).
- **BAD candidates are never filtered on the developer suite.** `existing_tests_pass` is a recorded column so Baseline A scores honestly.
- **Baseline A's GAR is 1.0 by construction here** (spec §3.6) and must be reported as `definitional`, never as a measurement. `score.report` already does this.
- **`seed.py` must not import from `rmt/stage1_mutate.py`** (spec §7.4). Java hazards go in `seed.HAZARDS["java"]`, which already exists.
- Commit after every task. Prefix: `feat(ad):`, `test(ad):`, `fix(ad):`.

---

### Task 1: Provision the Java toolchain, and prove it before anything else

**Files:**
- Create: `scripts/setup_java_toolchain.sh`
- Create: `additional_datasets/adapters/java_env.py`
- Create: `tests/test_java_env.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `JAVA8`, `JAVA17`, `MVN` (Paths); `available() -> bool`; `env_for(version) -> dict` (a process env with `JAVA_HOME` and `PATH` set); `mvn(args, cwd, version, timeout) -> subprocess.CompletedProcess`.

This is spec §8 step 1. If the toolchain does not work, nothing downstream is meaningful, so it is Task 1 and it ends with a real Maven build.

- [ ] **Step 1: Write the provisioning script**

`scripts/setup_java_toolchain.sh`:

```bash
#!/usr/bin/env bash
# Two JDKs and Maven, from conda-forge, without sudo. The host has no system java.
set -euo pipefail
ENV_ROOT="${AD_JAVA_ROOT:-$HOME/.ad_java}"

conda create -y -p "$ENV_ROOT/jdk8"  -c conda-forge openjdk=8
conda create -y -p "$ENV_ROOT/jdk17" -c conda-forge openjdk=17 maven=3.9

"$ENV_ROOT/jdk8/bin/java"   -version
"$ENV_ROOT/jdk17/bin/java"  -version
"$ENV_ROOT/jdk17/bin/mvn"   -version
echo "toolchain ready at $ENV_ROOT"
```

- [ ] **Step 2: Run it**

```bash
chmod +x scripts/setup_java_toolchain.sh && ./scripts/setup_java_toolchain.sh
```
Expected: `openjdk version "1.8.0_..."`, `openjdk version "17..."`, and a Maven version. conda-forge carries `openjdk` 8.0.412/8.0.472, 17.0.17/17.0.18 and `maven` 3.9.13–3.9.16, all verified present. If the solve fails, stop and report — every later task depends on this.

- [ ] **Step 3: Write the failing test**

Create `tests/test_java_env.py`:

```python
import pytest

from additional_datasets.adapters import java_env as JE

pytestmark = pytest.mark.skipif(not JE.available(),
                                reason="java toolchain not provisioned")


def test_env_for_points_java_home_at_the_right_jdk():
    e8 = JE.env_for(8)
    e17 = JE.env_for(17)
    assert e8["JAVA_HOME"] != e17["JAVA_HOME"]
    assert e8["JAVA_HOME"].endswith("jdk8")
    assert e17["JAVA_HOME"].endswith("jdk17")
    assert e8["JAVA_HOME"] + "/bin" in e8["PATH"]


def test_each_jdk_reports_its_own_version():
    import subprocess
    for version, needle in ((8, "1.8."), (17, "17.")):
        p = subprocess.run([str(JE.java_home(version) / "bin" / "java"), "-version"],
                           capture_output=True, text=True)
        assert needle in (p.stdout + p.stderr), (version, p.stdout + p.stderr)


def test_maven_runs_under_jdk17():
    p = JE.mvn(["-version"], cwd=".", version=17, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "Apache Maven" in p.stdout
```

- [ ] **Step 4: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_java_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.adapters.java_env'`

- [ ] **Step 5: Implement `additional_datasets/adapters/java_env.py`**

```python
"""The two-JDK Maven toolchain, provisioned by scripts/setup_java_toolchain.sh.

This machine has no system java, javac or mvn, so both JDKs come from conda envs and are
selected per invocation by JAVA_HOME. P0 builds under 8 and every candidate builds under
17, which is the whole migration, so the version is an explicit argument everywhere rather
than ambient state.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("AD_JAVA_ROOT", Path.home() / ".ad_java"))
JAVA8 = ROOT / "jdk8"
JAVA17 = ROOT / "jdk17"
MVN = JAVA17 / "bin/mvn"


def java_home(version):
    return {8: JAVA8, 17: JAVA17}[version]


def available():
    return (JAVA8 / "bin/java").exists() and (JAVA17 / "bin/java").exists() and MVN.exists()


def env_for(version):
    home = java_home(version)
    return {**os.environ, "JAVA_HOME": str(home),
            "PATH": f"{home}/bin:{JAVA17}/bin:{os.environ.get('PATH', '')}",
            # Keep every build's downloads in one warm local repository.
            "MAVEN_OPTS": os.environ.get("MAVEN_OPTS", "-Xmx2g")}


def mvn(args, cwd, version=17, timeout=1800):
    """Run Maven under the named JDK. Batch mode, no colour, one shared local repo."""
    cmd = [str(MVN), "-B", "-ntp",
           f"-Dmaven.repo.local={ROOT / 'm2repo'}", *args]
    return subprocess.run(cmd, cwd=str(cwd), env=env_for(version),
                          capture_output=True, text=True, timeout=timeout)
```

- [ ] **Step 6: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_java_env.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 7: Commit**

```bash
git add scripts/setup_java_toolchain.sh additional_datasets/adapters/java_env.py tests/test_java_env.py
git commit -m "feat(ad): two-JDK maven toolchain from conda, selected per invocation"
```

---

### Task 2: The MigrationBench dataset and the task pool

**Files:**
- Create: `additional_datasets/adapters/migbench_data.py`
- Create: `tests/test_migbench_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CSV` (Path), `Repo` dataclass (`repo`, `base_commit`, `num_java_files`, `num_loc`, `num_pom_xml`, `num_src_test_java_files`, `num_test_cases`, `license`, `task_id`); `load_repos(which="selected") -> list[Repo]`; `pool(repos, min_tests=5, min_test_files=2, max_loc=None) -> list[Repo]`; `clone(repo, dest, timeout=900) -> Path`.

`task_id` is `repo.replace("/", "__") + "@" + base_commit[:8]`, matching the `owner@name__sha` shape the `pmb/` results already use.

- [ ] **Step 1: Download the dataset**

```bash
mkdir -p /nas/CRUST-bench-COLM/CRUST-bench/datasets/MigrationBench
cd /nas/CRUST-bench-COLM/CRUST-bench/datasets/MigrationBench
curl -sL "https://huggingface.co/datasets/AmazonScience/migration-bench-java-selected/resolve/main/test.csv" -o selected.csv
curl -sL "https://huggingface.co/datasets/AmazonScience/migration-bench-java-full/resolve/main/test.csv" -o full.csv
wc -l selected.csv full.csv
```
Expected: `301 selected.csv` (300 repos + header) and `5103 full.csv`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_migbench_data.py`:

```python
import pytest

from additional_datasets.adapters import migbench_data as MB

pytestmark = pytest.mark.skipif(not MB.CSV.exists(), reason="MigrationBench csv not downloaded")


def test_loads_the_300_selected_repos():
    repos = MB.load_repos("selected")
    assert len(repos) == 300
    assert all(r.base_commit and r.repo for r in repos)


def test_task_id_is_stable_and_filesystem_safe():
    r = MB.Repo(repo="JodaOrg/joda-beans", base_commit="abcdef1234567890",
                num_java_files=1, num_loc=2, num_pom_xml=1,
                num_src_test_java_files=3, num_test_cases=4, license="Apache-2.0")
    assert r.task_id == "JodaOrg__joda-beans@abcdef12"
    assert "/" not in r.task_id


def test_pool_keeps_only_repos_with_a_real_developer_suite():
    """Spec §3.2: 92 of the 300 have >=5 test cases across >=2 test files."""
    got = MB.pool(MB.load_repos("selected"), min_tests=5, min_test_files=2)
    assert len(got) == 92
    assert all(r.num_test_cases >= 5 and r.num_src_test_java_files >= 2 for r in got)


def test_pool_can_cap_loc_for_a_pilot():
    """`anba/es6draft` is 205k LOC -- not a pilot-sized `mvn test`."""
    got = MB.pool(MB.load_repos("selected"), min_tests=5, min_test_files=2, max_loc=20000)
    assert got and all(r.num_loc <= 20000 for r in got)
    assert not any(r.repo == "anba/es6draft" for r in got)


def test_unknown_test_case_counts_are_treated_as_zero_not_negative():
    """The CSV uses -1 and -2 as 'unknown'; they must never pass a min_tests filter."""
    repos = MB.load_repos("full")
    assert any(r.num_test_cases < 0 for r in repos)
    assert all(r.num_test_cases >= 5 for r in MB.pool(repos, min_tests=5, min_test_files=2))
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_migbench_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.adapters.migbench_data'`

- [ ] **Step 4: Implement `additional_datasets/adapters/migbench_data.py`**

```python
"""MigrationBench access.

Like PyMigBench, the benchmark ships metadata only -- one CSV naming a repository and a
base commit -- so every task has to be materialised from GitHub before it can be run. The
`num_test_cases` column is why this benchmark was chosen: it is the only one in the study
that supplies Baseline A without extra work.
"""
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench/datasets/MigrationBench")
CSV = ROOT / "selected.csv"


@dataclass(frozen=True)
class Repo:
    repo: str
    base_commit: str
    num_java_files: int
    num_loc: int
    num_pom_xml: int
    num_src_test_java_files: int
    num_test_cases: int
    license: str

    @property
    def task_id(self):
        return f"{self.repo.replace('/', '__')}@{self.base_commit[:8]}"

    @property
    def url(self):
        return f"https://github.com/{self.repo}.git"


def _int(v):
    """The CSV uses -1 and -2 for 'unknown'; a filter must never read those as counts."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def load_repos(which="selected"):
    path = ROOT / f"{which}.csv"
    with path.open() as fh:
        return [Repo(repo=r["repo"], base_commit=r["base_commit"],
                     num_java_files=_int(r["num_java_files"]),
                     num_loc=_int(r["num_loc"]), num_pom_xml=_int(r["num_pom_xml"]),
                     num_src_test_java_files=_int(r["num_src_test_java_files"]),
                     num_test_cases=_int(r["num_test_cases"]),
                     license=r.get("license", ""))
                for r in csv.DictReader(fh)]


def pool(repos, min_tests=5, min_test_files=2, max_loc=None):
    """Repositories with a developer suite worth calling Baseline A."""
    return [r for r in repos
            if r.num_test_cases >= min_tests
            and r.num_src_test_java_files >= min_test_files
            and (max_loc is None or r.num_loc <= max_loc)]


def clone(repo, dest, timeout=900):
    """Shallow-fetch the exact base commit; no history beyond it is needed."""
    dest = Path(dest)
    if (dest / ".git").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for cmd in (["git", "init", "-q"],
                ["git", "remote", "add", "origin", repo.url],
                ["git", "fetch", "-q", "--depth", "1", "origin", repo.base_commit],
                ["git", "checkout", "-q", "FETCH_HEAD"]):
        p = subprocess.run(cmd, cwd=dest, capture_output=True, text=True, timeout=timeout)
        if p.returncode:
            raise RuntimeError(f"{repo.repo}: {' '.join(cmd)} failed: {p.stderr[-500:]}")
    return dest
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_migbench_data.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/adapters/migbench_data.py tests/test_migbench_data.py
git commit -m "feat(ad): migrationbench dataset access and the 92-repo task pool"
```

---

### Task 3: The Maven adapter

**Files:**
- Create: `additional_datasets/adapters/java_mvn.py`
- Create: `tests/test_java_mvn.py`

**Interfaces:**
- Consumes: `java_env` (Task 1), `migbench_data` (Task 2), `SuiteResult` (harness Task 1).
- Produces: `stage(task_id, dest, sources=None) -> Path`; `build(workspace, version) -> SuiteResult`; `run_suite(workspace, suite_text, suite_name) -> SuiteResult`; `run_existing(workspace, version=17) -> SuiteResult`; `parse_surefire(workspace) -> (status, cases)`; `set_release(workspace, release) -> None`; `SOURCE_ROOTS`.

`run_suite` writes the generated suite as a JUnit class into `src/test/java/` and runs only that class with `-Dtest=<name>`, so one generated suite cannot be suppressed by an unrelated failing developer test — the same per-target isolation the 2026-09-08 write-up found was worth 35 projects of signal.

- [ ] **Step 1: Write the failing test**

Create `tests/test_java_mvn.py`:

```python
import textwrap

import pytest

from additional_datasets.adapters import java_env as JE
from additional_datasets.adapters import java_mvn as JM

pytestmark = pytest.mark.skipif(not JE.available(), reason="java toolchain not provisioned")


POM = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>ad</groupId><artifactId>smoke</artifactId><version>1.0</version>
  <properties>
    <maven.compiler.source>1.8</maven.compiler.source>
    <maven.compiler.target>1.8</maven.compiler.target>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>
  <dependencies>
    <dependency><groupId>junit</groupId><artifactId>junit</artifactId>
      <version>4.13.2</version><scope>test</scope></dependency>
  </dependencies>
</project>
"""

SRC = "package ad;\npublic class Calc { public static int add(int a, int b) { return a + b; } }\n"


def _project(tmp_path):
    (tmp_path / "pom.xml").write_text(POM)
    d = tmp_path / "src/main/java/ad"
    d.mkdir(parents=True)
    (d / "Calc.java").write_text(SRC)
    return tmp_path


def test_parse_surefire_reads_case_names_and_failures(tmp_path):
    rep = tmp_path / "target/surefire-reports"
    rep.mkdir(parents=True)
    (rep / "TEST-ad.CalcTest.xml").write_text(textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <testsuite name="ad.CalcTest" tests="2" failures="1">
          <testcase name="adds" classname="ad.CalcTest"/>
          <testcase name="subtracts" classname="ad.CalcTest">
            <failure message="expected 1 was 2">boom</failure>
          </testcase>
        </testsuite>"""))
    status, cases = JM.parse_surefire(tmp_path)
    assert status == "test_failed"
    assert cases["ad.CalcTest.adds"] == "ok"
    assert cases["ad.CalcTest.subtracts"] == "FAILED"


def test_parse_surefire_treats_an_error_like_a_failure(tmp_path):
    rep = tmp_path / "target/surefire-reports"
    rep.mkdir(parents=True)
    (rep / "TEST-ad.T.xml").write_text(
        '<testsuite name="ad.T"><testcase name="x" classname="ad.T">'
        '<error message="npe">boom</error></testcase></testsuite>')
    assert JM.parse_surefire(tmp_path)[1]["ad.T.x"] == "FAILED"


def test_set_release_rewrites_the_compiler_target(tmp_path):
    p = _project(tmp_path)
    JM.set_release(p, 17)
    pom = (p / "pom.xml").read_text()
    assert "<maven.compiler.release>17</maven.compiler.release>" in pom
    assert "1.8" not in pom


@pytest.mark.slow
def test_a_trivial_project_builds_under_8_and_under_17(tmp_path):
    """Spec §8 step 1: the toolchain smoke test. If this fails, stop."""
    p = _project(tmp_path)
    assert JM.build(p, version=8).status == "green"
    JM.set_release(p, 17)
    assert JM.build(p, version=17).status == "green"


@pytest.mark.slow
def test_run_suite_executes_only_the_generated_class(tmp_path):
    p = _project(tmp_path)
    JM.set_release(p, 17)
    suite = ("package ad;\nimport org.junit.Test;\nimport static org.junit.Assert.*;\n"
             "public class ADGenTest { @Test public void addsTwo() { "
             "assertEquals(3, Calc.add(1, 2)); } }\n")
    res = JM.run_suite(p, suite, "ADGenTest")
    assert res.status == "green", res.out[-2000:]
    assert any(c.endswith("addsTwo") for c in res.cases)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_java_mvn.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.adapters.java_mvn'`

- [ ] **Step 3: Register the `slow` marker**

Append to `pytest.ini`:

```ini
markers =
    slow: needs a real Maven build (minutes, network)
```

- [ ] **Step 4: Implement `additional_datasets/adapters/java_mvn.py`**

```python
"""Build and run Maven projects under a chosen JDK.

Two decisions carry the experiment:

- **The generated suite runs alone.** `-Dtest=<class>` restricts the run to the class we
  wrote, so a failing developer test elsewhere in the project cannot suppress our verdict.
  The C<->Rust study lost 35 projects' entire signal to exactly this before it built one
  target at a time.
- **A build failure is never a detection.** It is reported as `build_error` and the caller
  turns that into a structural rejection or an exclusion; spec §20 forbids counting a
  compilation failure as a caught fault.
"""
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from ..spec import SuiteResult
from . import java_env as JE

SOURCE_ROOTS = ("src/main/java", "src/main/resources")
TEST_ROOT = "src/test/java"
GEN_PACKAGE = "ad_generated"
BUILD_TIMEOUT = 1800
TEST_TIMEOUT = 1800

_PROP_RE = re.compile(
    r"<maven\.compiler\.(source|target|release)>[^<]*</maven\.compiler\.\1>")
_PROPS_OPEN = re.compile(r"<properties>")


def stage(task_id, dest, sources=None, origin=None):
    """Copy a checked-out project into a writable workspace, overlaying `sources`."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(origin, dest, ignore=shutil.ignore_patterns(".git", "target"))
    for rel, text in (sources or {}).items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return dest


def set_release(workspace, release):
    """Point the compiler at a Java release, replacing any source/target/release pair."""
    pom = Path(workspace) / "pom.xml"
    text = _PROP_RE.sub("", pom.read_text())
    tag = f"<maven.compiler.release>{release}</maven.compiler.release>"
    if "<properties>" in text:
        text = _PROPS_OPEN.sub(f"<properties>\n    {tag}", text, count=1)
    else:
        text = text.replace("</project>", f"  <properties>\n    {tag}\n  </properties>\n</project>")
    pom.write_text(text)


def _result(status, cases, p, t0, out=""):
    return SuiteResult(status=status, cases=cases,
                       rc=getattr(p, "returncode", 1) if p else 1,
                       seconds=round(time.time() - t0, 1), out=out[-6000:])


def build(workspace, version=17):
    """Compile only -- no tests. `green` means it compiles under that JDK."""
    t0 = time.time()
    try:
        p = JE.mvn(["-DskipTests", "test-compile"], cwd=workspace, version=version,
                   timeout=BUILD_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        return _result("timeout", {}, None, t0, str(e))
    out = p.stdout + p.stderr
    return _result("green" if p.returncode == 0 else "build_error", {}, p, t0, out)


def parse_surefire(workspace):
    """(status, {fqcn.method: "ok"|"FAILED"}) from target/surefire-reports/*.xml."""
    reports = sorted((Path(workspace) / "target/surefire-reports").glob("TEST-*.xml"))
    cases = {}
    for f in reports:
        try:
            root = ET.fromstring(f.read_text(errors="ignore"))
        except ET.ParseError:
            continue
        for tc in root.iter("testcase"):
            name = f"{tc.get('classname', '')}.{tc.get('name', '')}"
            if tc.find("skipped") is not None:
                continue
            bad = tc.find("failure") is not None or tc.find("error") is not None
            cases[name] = "FAILED" if bad else "ok"
    if not cases:
        return "no_tests", cases
    return ("test_failed" if any(v == "FAILED" for v in cases.values()) else "green"), cases


def _run_tests(workspace, selector, version, timeout):
    t0 = time.time()
    shutil.rmtree(Path(workspace) / "target/surefire-reports", ignore_errors=True)
    args = ["-Dsurefire.failIfNoSpecifiedTests=false", "test"]
    if selector:
        args.insert(0, f"-Dtest={selector}")
    try:
        p = JE.mvn(args, cwd=workspace, version=version, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return _result("timeout", {}, None, t0, str(e))
    out = p.stdout + p.stderr
    status, cases = parse_surefire(workspace)
    if not cases and ("COMPILATION ERROR" in out or "BUILD FAILURE" in out and not cases):
        return _result("build_error", {}, p, t0, out)
    return _result(status, cases, p, t0, out)


def run_suite(workspace, suite_text, suite_name, version=17):
    """Write the generated JUnit class and run only it."""
    cls = re.search(r"public\s+class\s+(\w+)", suite_text)
    name = cls.group(1) if cls else suite_name
    pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", suite_text, re.M)
    rel = Path(TEST_ROOT, *(pkg.group(1).split(".") if pkg else []), f"{name}.java")
    p = Path(workspace) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(suite_text)
    try:
        return _run_tests(workspace, name, version, TEST_TIMEOUT)
    finally:
        p.unlink(missing_ok=True)


def run_existing(workspace, version=17):
    """The repository's own suite -- Baseline A."""
    return _run_tests(workspace, None, version, TEST_TIMEOUT)
```

- [ ] **Step 5: Run the fast tests**

Run: `.venv/bin/python -m pytest tests/test_java_mvn.py -v -m "not slow"`
Expected: PASS, 3 tests.

- [ ] **Step 6: Run the slow toolchain smoke test**

Run: `.venv/bin/python -m pytest tests/test_java_mvn.py -v -m slow`
Expected: PASS, 2 tests. The first Maven run downloads plugins into `~/.ad_java/m2repo` and will take several minutes; later runs reuse it. **If this fails, stop and fix the toolchain** — it is spec §8 step 1.

- [ ] **Step 7: Commit**

```bash
git add additional_datasets/adapters/java_mvn.py tests/test_java_mvn.py pytest.ini
git commit -m "feat(ad): maven adapter, per-class test isolation, build failures never count as detections"
```

---

### Task 4: The phase-0 executability ladder

**Files:**
- Create: `additional_datasets/benchmarks/java_phase0.py`
- Create: `tests/test_java_phase0.py`

**Interfaces:**
- Consumes: `migbench_data`, `java_mvn`, `java_env`.
- Produces: `LEVELS`; `probe(repo, workdir) -> dict` with keys `task_id`, `level`, `detail`, `n_cases`, `seconds`; `run(repos, workers=4, redo=False) -> list[dict]`; `load() -> list[dict]`; `summarise(rows) -> str`.

Spec §3.3. This is the plan's largest risk: 92 repositories of median 7.3k LOC, no Docker, no warm Maven cache, and a real fraction will simply not build. The ladder exists to find that out cheaply and on the record.

- [ ] **Step 1: Write the failing test**

Create `tests/test_java_phase0.py`:

```python
from additional_datasets.benchmarks import java_phase0 as P0


def test_levels_are_ordered_from_cheapest_to_strongest():
    assert P0.LEVELS == ("failed", "cloned", "builds_8", "suite_runs", "suite_green")


def test_summarise_counts_each_level_and_names_the_usable_pool():
    rows = [{"task_id": "a", "level": "suite_green", "n_cases": 12},
            {"task_id": "b", "level": "suite_green", "n_cases": 8},
            {"task_id": "c", "level": "builds_8", "n_cases": 0},
            {"task_id": "d", "level": "failed", "n_cases": 0}]
    text = P0.summarise(rows)
    assert "suite_green" in text and "2" in text
    assert "usable pool: 2" in text


def test_usable_selects_only_suite_green_rows():
    rows = [{"task_id": "a", "level": "suite_green", "n_cases": 12},
            {"task_id": "c", "level": "suite_runs", "n_cases": 4}]
    assert [r["task_id"] for r in P0.usable(rows)] == ["a"]


def test_usable_can_require_a_minimum_case_count():
    rows = [{"task_id": "a", "level": "suite_green", "n_cases": 12},
            {"task_id": "b", "level": "suite_green", "n_cases": 2}]
    assert [r["task_id"] for r in P0.usable(rows, min_cases=5)] == ["a"]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_java_phase0.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.benchmarks.java_phase0'`

- [ ] **Step 3: Implement `additional_datasets/benchmarks/java_phase0.py`**

```python
"""How many MigrationBench repositories can actually be run here?

Cloning and `mvn test`-ing 92 repositories of median 7.3k LOC, with no Docker and a cold
Maven cache, is the largest cost and the largest risk in this benchmark. The ladder
measures it once, cheaply, and records the answer, so task selection is a fact rather than
a hope. Only `suite_green` repositories become tasks: the developer suite is Baseline A
and the labelling instrument, and a suite that does not pass on the trusted P0 cannot play
either role.
"""
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..adapters import java_mvn as JM
from ..adapters import migbench_data as MB

RESULTS = Path(__file__).resolve().parent.parent.parent / "results/additional_datasets/java_phase0"
WORK = Path("/tmp/ad_java_phase0")
LEVELS = ("failed", "cloned", "builds_8", "suite_runs", "suite_green")


def probe(repo, workdir=None):
    """Climb the ladder as far as this repository goes."""
    t0 = time.time()
    rec = {"task_id": repo.task_id, "repo": repo.repo, "commit": repo.base_commit,
           "loc": repo.num_loc, "level": "failed", "detail": "", "n_cases": 0}
    src = Path(workdir or WORK) / repo.task_id
    try:
        MB.clone(repo, src)
        rec["level"] = "cloned"
    except Exception as e:  # noqa: BLE001 - a clone failure is a datum, not a crash
        rec["detail"] = f"clone: {type(e).__name__}: {e}"[:400]
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    ws = JM.stage(repo.task_id, src.parent / f"{repo.task_id}__ws", origin=src)
    b = JM.build(ws, version=8)
    if b.status != "green":
        rec["detail"] = f"build_8: {b.status}: {b.out[-300:]}"
        rec["seconds"] = round(time.time() - t0, 1)
        return rec
    rec["level"] = "builds_8"

    t = JM.run_existing(ws, version=8)
    rec["n_cases"] = len(t.cases)
    if t.cases:
        rec["level"] = "suite_green" if t.status == "green" else "suite_runs"
    rec["detail"] = t.status
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def run(repos, workers=4, redo=False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "ladder.json"
    done = {r["task_id"]: r for r in (json.loads(out.read_text()) if out.exists() and not redo else [])}
    todo = [r for r in repos if r.task_id not in done]
    print(f"phase0: {len(todo)} to probe, {len(done)} cached", flush=True)

    def one(repo):
        try:
            rec = probe(repo)
        except Exception:  # noqa: BLE001 - one repo must not sink the ladder
            rec = {"task_id": repo.task_id, "repo": repo.repo, "level": "failed",
                   "detail": traceback.format_exc()[-400:], "n_cases": 0}
        print(f"  {rec['task_id']:50} {rec['level']:12} {rec.get('detail', '')[:60]}",
              flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, todo))
    allrows = sorted([*done.values(), *rows], key=lambda r: r["task_id"])
    out.write_text(json.dumps(allrows, indent=2))
    return allrows


def load():
    p = RESULTS / "ladder.json"
    return json.loads(p.read_text()) if p.exists() else []


def usable(rows, min_cases=0):
    return [r for r in rows if r["level"] == "suite_green" and r["n_cases"] >= min_cases]


def summarise(rows):
    counts = {lv: sum(1 for r in rows if r["level"] == lv) for lv in LEVELS}
    lines = ["| level | n |", "|---|---:|"]
    lines += [f"| {lv} | {counts[lv]} |" for lv in LEVELS]
    lines.append("")
    lines.append(f"usable pool: {len(usable(rows))} of {len(rows)} probed")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_java_phase0.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Probe a small slice for real**

```bash
.venv/bin/python -c "
from additional_datasets.adapters import migbench_data as MB
from additional_datasets.benchmarks import java_phase0 as P0
repos = MB.pool(MB.load_repos('selected'), min_tests=5, min_test_files=2, max_loc=20000)
print(f'{len(repos)} repos in the capped pool')
rows = P0.run(repos[:10], workers=3)
print(P0.summarise(rows))
"
```
Expected: a ladder table. **Record the `suite_green` fraction — this is the real number behind spec §10's headline risk.** If it is below about 3 in 10, report it before continuing: the pilot may need the `-full` CSV rather than `selected`.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/benchmarks/java_phase0.py tests/test_java_phase0.py
git commit -m "feat(ad): phase-0 executability ladder over the migrationbench pool"
```

---

### Task 5: Java migration candidates

**Files:**
- Create: `additional_datasets/prompts/migrate_java.md`
- Create: `additional_datasets/benchmarks/java_candidates.py`
- Create: `tests/test_java_candidates.py`

**Interfaces:**
- Consumes: `java_mvn`, `harvest.label`, `seed.seed_faults`, `Candidate`.
- Produces: `TRANSFORMATION`; `anchor_migration(workspace) -> dict | None` (the minimal-diff G1); `propose_migration(task, sources, model_tag, temperature, tag) -> dict | None`; `admit(adapter, task, sources, workspace, origin) -> (label, why)`.

Spec §3.4 and §3.5. GOOD admission needs three checks — builds under 17, developer suite green, and no probe disagreement. BAD admission needs two — builds under 17, and the oracle confirms a real behavioural difference.

- [ ] **Step 1: Write the prompt**

`additional_datasets/prompts/migrate_java.md`:

```markdown
Migrate this Java project from Java 8 to Java 17 without changing any externally
observable behaviour.

## The project's source

<<SOURCES>>

## Requirements

- The project must compile under Java 17.
- Replace APIs removed or deprecated since Java 8 with their supported equivalents.
- Preserve observable behaviour exactly: return values, thrown exception types and
  messages, iteration order, numeric results, formatting, and encoding.
- Do not change public signatures, and do not rewrite logic beyond what the migration
  requires.

## Output

Return one fenced `json` block mapping each file path (relative to the project root) to
its complete new contents. Include every file you changed, and no others. `pom.xml` counts
as a file you may change.
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_java_candidates.py`:

```python
from additional_datasets.benchmarks import java_candidates as JC
from additional_datasets.spec import SuiteResult, Task


def _task(tmp_path):
    return Task(task_id="o__r@abcd1234", benchmark="migrationbench_java",
                language="java", repository="o/r", source_commit="abcd1234",
                transformation=JC.TRANSFORMATION, results_root=tmp_path)


class FakeAdapter:
    """Scripted build and suite results, in call order."""

    def __init__(self, build, existing):
        self.build_result, self.existing_result = build, existing
        self.staged = []

    def stage(self, task_id, dest, sources=None, origin=None):
        self.staged.append(dest)
        return dest

    def build(self, workspace, version=17):
        return self.build_result

    def run_existing(self, workspace, version=17):
        return self.existing_result


GREEN = SuiteResult(status="green", cases={"a": "ok"}, rc=0, seconds=1.0)
RED = SuiteResult(status="test_failed", cases={"a": "FAILED"}, rc=1, seconds=1.0)
BROKEN = SuiteResult(status="build_error", cases={}, rc=1, seconds=1.0)


def test_a_candidate_that_does_not_compile_under_17_is_excluded(tmp_path):
    a = FakeAdapter(build=BROKEN, existing=GREEN)
    label, why = JC.admit(a, _task(tmp_path), {}, tmp_path, origin=tmp_path)
    assert label == "EXCLUDE"
    assert "compile" in why


def test_a_compiling_candidate_with_a_green_suite_is_good(tmp_path):
    a = FakeAdapter(build=GREEN, existing=GREEN)
    assert JC.admit(a, _task(tmp_path), {}, tmp_path, origin=tmp_path) == ("GOOD", None)


def test_a_compiling_candidate_whose_suite_regressed_is_bad(tmp_path):
    a = FakeAdapter(build=GREEN, existing=RED)
    label, why = JC.admit(a, _task(tmp_path), {}, tmp_path, origin=tmp_path)
    assert label == "BAD"
    assert "a" in why


def test_transformation_is_the_java_8_to_17_migration_spec():
    assert "Java 8" in JC.TRANSFORMATION and "17" in JC.TRANSFORMATION
    assert "observable behaviour" in JC.TRANSFORMATION
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_java_candidates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.benchmarks.java_candidates'`

- [ ] **Step 4: Implement `additional_datasets/benchmarks/java_candidates.py`**

```python
"""GOOD and BAD Java 8 -> 17 migrations, and the rules that admit them.

The asymmetry between the two admission rules is deliberate and is spec §3.5. GOOD needs
the developer suite to pass, because that is the strongest available evidence that
behaviour was preserved. BAD must *not* be filtered on the developer suite: filtering it
would force Baseline A's rejection rate to zero by construction and make the headline
comparison meaningless. `existing_tests_pass` is recorded per candidate instead, and
Baseline A scores whatever it honestly scores.
"""
import json

from rmt.llm import ask, json_block

from ..arms.naive import _bundle
from ..prompts import load

SYSTEM = "You are a careful Java engineer performing a behaviour-preserving migration."
TRANSFORMATION = ("Migrate this project from Java 8 to Java 17 while preserving all "
                  "externally observable behaviour.")


def propose_migration(task, sources, model_tag, temperature, tag):
    """One candidate migration, or None if the reply was unusable."""
    user = load("migrate_java", sources=_bundle(sources))
    reply = ask(SYSTEM, user, effort="high", max_tokens=32000,
                temperature=temperature, tag=f"{tag}_{model_tag}")
    try:
        files = json_block(reply)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(files, dict) or not files:
        return None
    return {"sources": {**sources, **files}}


def admit(adapter, task, sources, workspace, origin):
    """(GOOD|BAD|EXCLUDE, why).

    EXCLUDE means it does not compile under 17 -- spec §20 forbids counting that as a
    caught fault, so such a candidate leaves the experiment entirely.
    """
    ws = adapter.stage(task.task_id, workspace, sources=sources, origin=origin)
    b = adapter.build(ws, version=17)
    if b.status != "green":
        return "EXCLUDE", f"does not compile under Java 17: {b.status}"
    t = adapter.run_existing(ws, version=17)
    if t.status in ("no_tests", "error"):
        return "EXCLUDE", f"developer suite could not run: {t.status}"
    if t.status == "green":
        return "GOOD", None
    if t.status == "timeout":
        return "BAD", "developer suite timed out"
    return "BAD", f"developer suite regressed: {', '.join(t.failed[:6])}"


def anchor_migration(adapter, task, sources, workspace, origin):
    """G1: the minimal-diff migration -- raise the compiler release, repair what breaks.

    Tried before any model call, because if this does not pass the developer suite then
    the repository has no verified-correct migration and the task has no GOOD anchor.
    """
    from ..adapters import java_mvn as JM

    ws = adapter.stage(task.task_id, workspace, sources=sources, origin=origin)
    JM.set_release(ws, 17)
    pom = (ws / "pom.xml").read_text()
    patched = {**sources, "pom.xml": pom}
    label, why = admit(adapter, task, patched, workspace.parent / "anchor_check", origin)
    return {"sources": patched} if label == "GOOD" else None
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_java_candidates.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add additional_datasets/benchmarks/java_candidates.py \
        additional_datasets/prompts/migrate_java.md tests/test_java_candidates.py
git commit -m "feat(ad): java migration candidates and the asymmetric admission rules"
```

---

### Task 6: Wire the benchmark together and run the pilot

**Files:**
- Create: `additional_datasets/benchmarks/migrationbench_java.py`
- Modify: `run_additional_datasets.py` (add `migrationbench_java` to `BENCHMARKS` and `_module`)
- Create: `tests/test_java_benchmark.py`

**Interfaces:**
- Consumes: everything above, plus the harness's `freeze`, `score_task`, `seed_faults`, `ceiling.score_ceiling`.
- Produces: `build_task(row, results_root) -> Task`; `run_task(row, results_root, n_good, n_bad, workspace) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_java_benchmark.py`:

```python
import subprocess
import sys

from additional_datasets.benchmarks import migrationbench_java as MJ


def test_build_task_carries_the_migration_spec(tmp_path):
    row = {"task_id": "o__r@abcd1234", "repo": "o/r", "commit": "abcd1234",
           "loc": 5000, "level": "suite_green", "n_cases": 12}
    t = MJ.build_task(row, tmp_path)
    assert t.benchmark == "migrationbench_java"
    assert t.language == "java"
    assert t.repository == "o/r"
    assert t.source_commit == "abcd1234"
    assert "Java 8" in t.transformation
    assert t.root == tmp_path / "migrationbench_java" / "o__r@abcd1234"


def test_cli_accepts_the_java_benchmark():
    p = subprocess.run([sys.executable, "run_additional_datasets.py", "run", "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert "migrationbench_java" in p.stdout
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_java_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'additional_datasets.benchmarks.migrationbench_java'`

- [ ] **Step 3: Implement `additional_datasets/benchmarks/migrationbench_java.py`**

```python
"""MigrationBench wired into the shared harness.

The order below is the experiment and nothing may reorder it: the oracle and the arms are
built from P0 alone, the suites are frozen, and only then do candidates appear on disk.
`freeze` refuses once `candidates/` is non-empty, so the ordering is enforced against the
filesystem rather than promised in a comment.
"""
import json
from pathlib import Path

from ..adapters import java_mvn as JM
from ..adapters import migbench_data as MB
from ..arms import ceiling, mutation, naive
from ..runner import freeze, score_task
from ..seed import seed_faults
from ..spec import Candidate, Task, write_metadata
from .java_candidates import TRANSFORMATION, admit, anchor_migration, propose_migration

MODELS = [("claude-opus-5", 1.0), ("claude-haiku-4-5-20251001", 1.0)]
SOURCE_GLOBS = ("pom.xml", "src/main/java/**/*.java")
MAX_SOURCE_CHARS = 180_000


def build_task(row, results_root):
    return Task(task_id=row["task_id"], benchmark="migrationbench_java", language="java",
                repository=row["repo"], source_commit=row["commit"],
                transformation=TRANSFORMATION, results_root=Path(results_root))


def _sources(checkout):
    """The files a migration may change: the POM and the main sources, never the tests.

    Tests are excluded on purpose. They are the labelling instrument and Baseline A; a
    candidate allowed to rewrite them could pass by weakening its own oracle.
    """
    out, used = {}, 0
    for pattern in SOURCE_GLOBS:
        for p in sorted(Path(checkout).glob(pattern)):
            if not p.is_file():
                continue
            text = p.read_text(errors="ignore")
            if used + len(text) > MAX_SOURCE_CHARS:
                continue
            used += len(text)
            out[str(p.relative_to(checkout))] = text
    return out


def run_task(row, results_root, n_good=3, n_bad=3, workspace=None, log=print):
    task = build_task(row, results_root)
    task.root.mkdir(parents=True, exist_ok=True)
    ws = Path(workspace or (task.root / "_work"))
    ws.mkdir(parents=True, exist_ok=True)

    repo = next(r for r in MB.load_repos("selected") if r.task_id == row["task_id"])
    origin = MB.clone(repo, ws / "checkout")
    sources = _sources(origin)

    # 1. Every arm's suite, from P0 and the migration spec only. P0 is the Java 8 build.
    p0 = JM.stage(task.task_id, ws / "p0", origin=origin)
    adapter = _BoundAdapter(origin)
    suites, admissible, records = {}, {}, {}
    # The Java analogue of an import line: the suite is compiled into the project's own
    # test source root, so the classes under test are already on the classpath. Both arms
    # are told this identically (see the Task 8 note in the harness plan).
    imp = ("the classes under test are already on the test classpath; import them by "
           "their own package names")
    n_suite, rec = naive.generate(task, sources, adapter, p0, imp, framework="JUnit 4",
                                  tag=f"{task.task_id}_naive")
    suites["naive"], admissible["naive"], records["naive"] = n_suite, rec["admissible"], rec
    m_suite, rec = mutation.generate(task, sources, adapter, ws / "mut", imp,
                                     framework="JUnit 4", tag=f"{task.task_id}_mut")
    m_run = JM.run_suite(p0, m_suite, "ADMutTest") if m_suite else None
    suites["mutation"] = m_suite
    admissible["mutation"] = m_run.passed if m_run else []
    records["mutation"] = rec

    # 2. Freeze, before any candidate exists.
    hashes = freeze(task, suites)
    log(f"{task.task_id}: froze {list(hashes)}")

    # 3. GOOD candidates. G1 is the minimal-diff anchor; the rest come from models.
    candidates, good_sources = {}, None
    anchor = anchor_migration(adapter, task, sources, ws / "anchor", origin)
    if anchor:
        good_sources = anchor["sources"]
        candidates["good_01"] = (
            Candidate("good_01", task.task_id, "GOOD", None, None, "minimal-diff",
                      "n/a", {"strategy": "raise maven.compiler.release"}, "ok", None),
            anchor["sources"])
    else:
        log(f"{task.task_id}: no minimal-diff anchor -- the repo may not migrate cleanly")

    for i, (model, temp) in enumerate(MODELS * 3):
        n_g = sum(1 for c, _ in candidates.values() if c.label == "GOOD")
        if n_g >= n_good:
            break
        cand = propose_migration(task, sources, model, temp, f"{task.task_id}_c{i}")
        if not cand:
            continue
        lbl, why = admit(adapter, task, cand["sources"], ws / f"c{i}", origin)
        if lbl == "EXCLUDE":
            log(f"{task.task_id}/c{i}: excluded -- {why}")
            continue
        if lbl == "GOOD":
            cid = f"good_{n_g + 1:02d}"
            good_sources = good_sources or cand["sources"]
        else:
            cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        candidates[cid] = (Candidate(cid, task.task_id, lbl,
                                     None if lbl == "GOOD" else "N", None, model, model,
                                     {"temperature": temp}, "ok", why),
                           cand["sources"])

    # 4. Arm S: seeded migration hazards in a verified-GOOD migration.
    if good_sources:
        for j, f in enumerate(seed_faults(task, good_sources, n_bad,
                                          f"{task.task_id}_seed")):
            lbl, why = admit(adapter, task, f["sources"], ws / f"s{j}", origin)
            if lbl != "BAD":
                # Either it broke the build (excluded, §20) or the oracle cannot see it
                # (an equivalent fault, unkillable by anyone -- excluded, §3.5).
                log(f"{task.task_id}/s{j}: not admitted as BAD -- {why or lbl}")
                continue
            cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
            candidates[cid] = (Candidate(cid, task.task_id, "BAD", "S",
                                         f["fault_category"], "claude-opus-5",
                                         "claude-opus-5", {}, "ok", why),
                               f["sources"])

    for cid, (cand, src) in candidates.items():
        d = cand.dir(task)
        d.mkdir(parents=True, exist_ok=True)
        (d / "sources.json").write_text(json.dumps(src, indent=2))

    write_metadata(task, [c for c, _ in candidates.values()],
                   extra={"suite_hashes": hashes, "phase0": row,
                          "arms": {k: {x: v[x] for x in v if x != "suite"}
                                   for k, v in records.items()}})

    # 5. Score. ARM A is the repository's own suite, run per candidate by the adapter.
    rows = score_task(task, candidates, suites, admissible, adapter, ws / "score",
                      existing_text=None, existing_admissible=())
    rows += _score_existing(task, candidates, adapter, ws / "existing", origin)
    if good_sources:
        rows += ceiling.score_ceiling(task, candidates, adapter, ws / "ceiling",
                                      good_sources, imp)
    (task.root / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"{task.task_id}: {len(candidates)} candidates, {len(rows)} rows")
    return rows


class _BoundAdapter:
    """`java_mvn` with the checkout bound in, so it matches the harness adapter protocol."""

    def __init__(self, origin):
        self.origin = origin

    def stage(self, task_id, dest, sources=None, origin=None):
        return JM.stage(task_id, dest, sources=sources, origin=origin or self.origin)

    def build(self, workspace, version=17):
        return JM.build(workspace, version=version)

    def run_existing(self, workspace, version=17):
        return JM.run_existing(workspace, version=version)

    def run_suite(self, workspace, suite_text, suite_name):
        return JM.run_suite(workspace, suite_text, suite_name)


def _score_existing(task, candidates, adapter, workspace, origin):
    """ARM A: the repository's own suite, per candidate. GAR is definitional here (§3.6)."""
    from ..score import ACCEPT, BEHAVIOURAL, REJECT, STRUCTURAL

    rows = []
    for cid, (cand, sources) in sorted(candidates.items()):
        ws = adapter.stage(task.task_id, workspace / cid, sources=sources, origin=origin)
        res = adapter.run_existing(ws, version=17)
        if res.status == "green":
            v, kind = ACCEPT, ""
        elif res.status == "build_error":
            v, kind = REJECT, STRUCTURAL
        else:
            v, kind = REJECT, BEHAVIOURAL
        rows.append({"task_id": task.task_id, "benchmark": task.benchmark,
                     "candidate_id": cid, "ground_truth": cand.label, "label": cand.label,
                     "bad_arm": cand.bad_arm, "fault_category": cand.fault_category,
                     "arm": "existing", "compiles": True,
                     "existing_tests_pass": v == ACCEPT, "generated_tests_pass": v == ACCEPT,
                     "number_generated_tests": len(res.cases),
                     "tests_failed": len(res.failed), "verdict": v, "rejection_kind": kind,
                     "status": res.status, "execution_time": res.seconds,
                     "model": cand.model, "model_version": cand.model_version,
                     "hidden_oracle_failure": cand.hidden_oracle_failure})
    return rows
```

- [ ] **Step 4: Wire the CLI**

In `run_additional_datasets.py`, change:

```python
BENCHMARKS = ("refactorbench_js",)
```

to:

```python
BENCHMARKS = ("refactorbench_js", "migrationbench_java")
```

and extend `_module`:

```python
def _module(name):
    if name == "refactorbench_js":
        from additional_datasets.benchmarks import refactorbench_js
        return refactorbench_js
    if name == "migrationbench_java":
        from additional_datasets.benchmarks import migrationbench_java
        return migrationbench_java
    sys.exit(f"unknown benchmark {name!r}; known: {', '.join(BENCHMARKS)}")
```

and make `cmd_select` and `cmd_run` benchmark-aware rather than JS-only:

```python
def cmd_select(a):
    if a.benchmark == "refactorbench_js":
        from additional_datasets.adapters import js_vitest as JS
        names = JS.pure_logic_fixtures()
    else:
        from additional_datasets.benchmarks import java_phase0 as P0
        names = [r["task_id"] for r in P0.usable(P0.load())]
    print(f"{len(names)} tasks in the pool")
    for n in names:
        print(" ", n)


def cmd_run(a):
    mod = _module(a.benchmark)
    root = RESULTS / a.run if a.run else RESULTS
    if a.benchmark == "refactorbench_js":
        from additional_datasets.adapters import js_vitest as JS
        items = (a.fixtures or JS.pure_logic_fixtures())[:a.tasks]
    else:
        from additional_datasets.benchmarks import java_phase0 as P0
        rows = P0.usable(P0.load())
        if a.fixtures:
            rows = [r for r in rows if r["task_id"] in a.fixtures]
        items = rows[:a.tasks]
    rows = []
    for it in items:
        try:
            rows += mod.run_task(it, root, n_good=a.good, n_bad=a.bad)
        except Exception as e:  # noqa: BLE001 - one task must not sink the run
            name = it if isinstance(it, str) else it["task_id"]
            print(f"  {name}: ERROR {type(e).__name__}: {e}", flush=True)
    out = root / "rows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out} ({len(rows)} rows)")
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `.venv/bin/python -m pytest tests/test_java_benchmark.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all green, both plans' tests together.

- [ ] **Step 7: Probe the pool properly, then run a one-task pilot**

```bash
.venv/bin/python -c "
from additional_datasets.adapters import migbench_data as MB
from additional_datasets.benchmarks import java_phase0 as P0
repos = MB.pool(MB.load_repos('selected'), min_tests=5, min_test_files=2, max_loc=20000)
rows = P0.run(repos, workers=4)
print(P0.summarise(rows))
"
.venv/bin/python run_additional_datasets.py select --benchmark migrationbench_java
.venv/bin/python run_additional_datasets.py run --benchmark migrationbench_java \
    --run java_smoke --tasks 1 --good 2 --bad 2
.venv/bin/python run_additional_datasets.py report --run java_smoke
```
Expected: a REPORT.md with four arms. Check before scaling: `metadata.json` records both suite hashes, every BAD has a non-null `hidden_oracle_failure`, no BAD has `build_status == "build_error"`, and ARM A's GAR reads `definitional`.

- [ ] **Step 8: Run the 3-task pilot**

```bash
.venv/bin/python run_additional_datasets.py run --benchmark migrationbench_java \
    --run java_pilot --tasks 3 --good 3 --bad 3
.venv/bin/python run_additional_datasets.py report --run java_pilot
```

- [ ] **Step 9: Commit**

```bash
git add additional_datasets/benchmarks/migrationbench_java.py \
        run_additional_datasets.py tests/test_java_benchmark.py
git commit -m "feat(ad): migrationbench java 8->17 wired end to end"
```

---

## Verification checklist

Run before declaring this plan done:

- [ ] `.venv/bin/python -m pytest -q -m "not slow"` — all green
- [ ] `.venv/bin/python -m pytest -q -m slow` — the toolchain smoke tests pass
- [ ] `results/additional_datasets/java_phase0/ladder.json` exists, and the `suite_green` count is recorded in the pilot write-up
- [ ] `results/additional_datasets/java_pilot/REPORT.md` has non-empty BRR cells for all four arms
- [ ] ARM A's GAR renders as `definitional`, never as a percentage (spec §3.6)
- [ ] No BAD candidate has `build_status == "build_error"` (spec §20)
- [ ] Every BAD candidate has a non-null `hidden_oracle_failure` (no equivalent faults, spec §3.5)
- [ ] Arm S and Arm N appear as separate rows, never summed (spec §5)
- [ ] ARM C's BRR is ≥ every other arm's
- [ ] `grep -r stage1_mutate additional_datasets/` returns nothing (spec §7.4)
- [ ] No candidate's `sources.json` contains a path under `src/test/java` (candidates may not rewrite their own oracle)
