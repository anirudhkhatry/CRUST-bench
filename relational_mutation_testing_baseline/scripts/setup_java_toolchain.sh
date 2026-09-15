#!/usr/bin/env bash
# Two JDKs and Maven, from conda-forge, without sudo. The host has no system java.
set -euo pipefail
ENV_ROOT="${AD_JAVA_ROOT:-$HOME/.ad_java}"

conda create -y -p "$ENV_ROOT/jdk8"  -c conda-forge openjdk=8
conda create -y -p "$ENV_ROOT/jdk17" -c conda-forge openjdk=17 maven=3.9

"$ENV_ROOT/jdk8/bin/java"   -version
"$ENV_ROOT/jdk17/bin/java"  -version
# The conda-forge maven launcher requires JAVA_HOME even when invoked by absolute path;
# without it, `mvn -version` fails with "The JAVA_HOME environment variable is not
# defined correctly." (verified: exit 1 with the message above, exit 0 once exported).
JAVA_HOME="$ENV_ROOT/jdk17" "$ENV_ROOT/jdk17/bin/mvn" -version
echo "toolchain ready at $ENV_ROOT"
