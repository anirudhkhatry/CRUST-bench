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
