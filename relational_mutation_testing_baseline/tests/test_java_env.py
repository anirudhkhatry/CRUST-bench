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
