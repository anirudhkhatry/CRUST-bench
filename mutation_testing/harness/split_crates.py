"""Split a project's source and its per-setting tests into separate crates.

Phase 2 of ``docs/plan_shared_mutant_set.md``. Until now each setting was a full copy
of the project with that setting's tests in ``src/bin/*.rs`` -- i.e. *inside the crate
that gets mutated*. cargo-mutants marks a mutant ``unviable`` when the crate fails to
compile, and the test code is part of that crate, so a setting whose tests break
compilation silently loses mutants from its own denominator. The four settings then
score over four different mutant populations.

This module rebuilds each project as one small workspace::

    <project>/
      Cargo.toml          [workspace] members = ["core", "tests-src", ...]
      core/               the frozen translation -- the ONLY crate that gets mutated
      tests-src/          tests/*.rs for that setting, depends on core
      tests-src_and_test/
      tests-src_over_test/
      tests-c_and_rust/

which is then mutated once per setting with::

    cargo mutants --package <core> --test-package tests-src ...

``--package`` scopes mutant *generation* to the source crate, so the mutant list is
identical across the four invocations by construction; ``--test-package`` selects which
suite runs. Viability becomes a property of ``core`` alone, so the denominator is shared.

``core`` keeps the project's own package name (``aes128_SIMD``, not ``core``) because the
generated tests import it by that name -- they only ever touch the public API, so moving
them out of the crate is semantics-preserving. :func:`check_public_api_only` re-checks
that assumption per project.

The workspace here encloses ONE project (~3.4 MB), so cargo-mutants' per-run copy of the
enclosing workspace stays cheap -- this does not reintroduce the umbrella-workspace cost
that ``split_manifests.py`` removed.
"""

import re
import shutil
from pathlib import Path

from harness.split_manifests import _FALLBACK_VERSIONS, _INHERITED, _VERSIONED, used_crates

# Setting -> the test crate's package/directory name. Cargo package names admit only
# alphanumerics, '-' and '_', so "src+test" / "src/test" cannot be used verbatim.
TEST_CRATE_NAMES = {
    "src": "tests-src",
    "src+test": "tests-src_and_test",
    "src/test": "tests-src_over_test",
    "c+rust": "tests-c_and_rust",
}

CORE_DIR = "core"


def _manifest_source(proj):
    """The manifest to read dependencies from: the pre-split backup wins.

    ``split_manifests.rewrite_manifest`` keeps the untouched manifest beside the rewritten
    one as ``Cargo.toml.workspace.bak`` and treats it as the source of truth on re-runs;
    do the same, so a project already made standalone still yields its full dep list.
    """
    proj = Path(proj)
    backup = proj / "Cargo.toml.workspace.bak"
    return backup if backup.exists() else proj / "Cargo.toml"


def _dependency_lines(manifest, versions, words):
    """Dependency lines from ``manifest``'s [dependencies], kept only if used in ``words``.

    ``foo = { workspace = true }`` is inlined to a concrete version (there is no parent
    workspace table to inherit from any more); explicitly versioned deps are kept verbatim
    so pins and feature flags survive. Same over-approximate used/unused test as
    ``split_manifests``: a false keep costs build time, a false drop breaks the build.
    """
    manifest = Path(manifest)
    if not manifest.exists():
        return []
    lines = manifest.read_text(encoding="utf-8").splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() != "[dependencies]":
            i += 1
            continue
        i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("["):
            body = lines[i]
            m = _INHERITED.match(body)
            if m:
                name = m.group(1)
                if name.replace("-", "_") in words:
                    rhs = versions.get(name) or _FALLBACK_VERSIONS.get(name)
                    if rhs:
                        out.append(f"{name} = {rhs}")
            elif body.strip():
                explicit = _VERSIONED.match(body)
                if explicit and explicit.group(1).replace("-", "_") in words:
                    out.append(body)
            i += 1
    return out


def _write_manifest(path, package, deps, extra=()):
    body = [
        "[package]",
        f'name = "{package}"',
        'version = "0.1.0"',
        'edition = "2021"',
        *extra,
        "",
        "[dependencies]",
        *deps,
        "",
    ]
    Path(path).write_text("\n".join(body), encoding="utf-8")


# Constructs that stop working once the test moves out of the crate: `crate::` now names
# the test crate rather than the source, and #[path]/include! pull in source files by a
# relative path that no longer resolves. `super::*` is NOT listed -- inside the usual
# `mod tests { use super::*; }` it refers to the test file's own module and is fine.
_INTERNAL_ACCESS = re.compile(r"(\bcrate::|#\[path\b|\binclude!\s*\()")


def check_public_api_only(test_files):
    """Return [(file, snippet)] for test files that reach past the public API.

    The plan requires re-verifying, before migrating, that the generated tests only use
    the public API. Reported rather than raised: such a test simply fails to compile in
    its new crate and is repaired or pruned by the green-baseline phase.
    """
    offenders = []
    for f in test_files:
        try:
            text = Path(f).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = _INTERNAL_ACCESS.search(text)
        if m:
            line = text[: m.start()].count("\n") + 1
            offenders.append((str(f), f"line {line}: {m.group(1)}"))
    return offenders


def build_workspace(project_name, source_proj, tests_by_setting, dst, versions=None):
    """Assemble ``dst`` as a workspace with a ``core`` crate and one crate per setting.

    ``source_proj`` is the frozen translation to copy into ``core``; ``tests_by_setting``
    maps setting -> directory of that setting's ``*.rs`` test files (typically the
    setting copy's ``src/bin``). Settings with no test files are omitted from the
    workspace -- an empty member would just be dead weight.

    Returns {"core": <core dir>, "tests": {setting: <crate dir>}, "api_offenders": [...]}.
    """
    source_proj, dst = Path(source_proj), Path(dst)
    versions = dict(versions or _FALLBACK_VERSIONS)
    if dst.exists():
        shutil.rmtree(dst)
    core = dst / CORE_DIR
    (core / "src").mkdir(parents=True)

    # --- core: the frozen source, minus any in-crate test bins. ---
    src_dir = source_proj / "src"
    for entry in sorted(src_dir.iterdir()) if src_dir.is_dir() else []:
        if entry.name == "bin":
            continue  # tests move to their own crates; nothing else may live here
        (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, core / "src" / entry.name)
    if not (core / "src" / "lib.rs").exists():
        # The generated manifest declares no explicit [lib], so cargo needs the default
        # entry point. Every translated project has one (write_rust_files builds it).
        raise FileNotFoundError(f"{source_proj}: no src/lib.rs to use as the core crate root")
    manifest = _manifest_source(source_proj)
    _write_manifest(core / "Cargo.toml", project_name,
                    _dependency_lines(manifest, versions, used_crates(core)))

    # --- one test crate per setting, each depending on core. ---
    members, test_dirs, offenders = [CORE_DIR], {}, []
    for setting, tests_dir in sorted(tests_by_setting.items()):
        tests_dir = Path(tests_dir) if tests_dir else None
        files = sorted(tests_dir.glob("*.rs")) if tests_dir and tests_dir.is_dir() else []
        if not files:
            continue
        crate_name = TEST_CRATE_NAMES[setting]
        crate = dst / crate_name
        (crate / "tests").mkdir(parents=True)
        (crate / "src").mkdir()
        # An empty lib target, so the crate stays a valid package even after the green
        # baseline prunes its last test. Without it cargo reports "no targets specified
        # in the manifest" -- and that error is fatal for the WHOLE workspace, so one
        # setting losing all its tests would take the other three down with it.
        (crate / "src" / "lib.rs").write_text("", encoding="utf-8")
        for f in files:
            shutil.copy2(f, crate / "tests" / f.name)
        offenders += [(setting, *o) for o in check_public_api_only(files)]
        deps = [f'{project_name} = {{ path = "../{CORE_DIR}" }}']
        deps += _dependency_lines(manifest, versions, used_crates(crate))
        # publish = false: these crates exist only to host a test suite.
        _write_manifest(crate / "Cargo.toml", crate_name, deps, extra=["publish = false"])
        members.append(crate_name)
        test_dirs[setting] = crate

    (dst / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = [\n'
        + "".join(f'    "{m}",\n' for m in members)
        + "]\n",
        encoding="utf-8",
    )
    return {"core": core, "tests": test_dirs, "members": members, "api_offenders": offenders}
