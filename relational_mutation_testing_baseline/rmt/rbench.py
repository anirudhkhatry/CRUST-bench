"""RBench access and Rust test execution (design spec S3/S4).

RBench itself ships only *interfaces* and a ground-truth test suite -- no Rust
implementation. The reference implementation is therefore the codex transpilation, which
we take as correct for every project where it passes the ground-truth suite (94/100).

So this module does two things: it serves the interfaces (the only Rust contract Stage 3
may transpile against, so the transpiler cannot copy an answer out of an implementation),
and it runs generated Rust tests against a throwaway copy of the reference.

A generated test is injected as `src/bin/rmt_test.rs` plus a `[[test]]` stanza, which is
exactly how RBench registers its own ground-truth tests.
"""
import csv
import functools
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench")
RBENCH = ROOT / "datasets/RBench"
REFERENCE = ROOT / "datasets/codex_expts/all/initial_codex_output"
PHASE0 = ROOT / "relational_mutation_testing_baseline/results/phase0"
SCRATCH = Path(os.environ.get("RMT_CARGO_TARGET", "/tmp/rmt_cargo"))

FN_RE = re.compile(r"\bfn\s+(\w+)")
TRAIT_RE = re.compile(r"^(pub\s+)?(?:unsafe\s+)?trait\s+\w+")
IMPL_RE = re.compile(r"^impl\b(?P<head>.*)$")
TEST_NAME = "rmt_test"
# `tests/`, not `src/bin/`. RBench registers its own ground-truth tests as `[[test]]`
# stanzas pointing into `src/bin/`, but cargo *also* auto-discovers everything under
# `src/bin/` as a binary -- and in a binary `#[cfg(test)]` is off, so the `#[test] fn
# rmt_kill` is stripped and any `fn main` calling it fails to compile. That charged the
# generated test with an F2 it had not earned, on every project that ships a `src/bin/`.
TEST_PATH = f"tests/{TEST_NAME}.rs"


def iface_files(project):
    d = RBENCH / project / "src/interfaces"
    return sorted(d.rglob("*.rs")) if d.is_dir() else []


def iface_source(project, max_chars=120_000):
    """The interface .rs files, fenced -- the whole Rust contract Stage 3 may use."""
    parts, used = [], 0
    for f in iface_files(project):
        text = f.read_text(errors="ignore")
        rel = f.relative_to(RBENCH / project).as_posix()
        if used + len(text) > max_chars:
            parts.append(f"### `{rel}` (omitted, prompt budget)\n")
            continue
        used += len(text)
        parts.append(f"### `{rel}`\n\n```rust\n{text}\n```\n")
    return "\n".join(parts)


def pub_fns(text):
    """Function names an outside caller can name -- the observable surface.

    An interface file declares private helpers next to the public API; only the public
    ones are a contract a generated test may call. Counting both told Stage 1 that a
    private helper was observable and Stage 2 to assert on it, and the transpiled test
    then could not compile -- an F2 manufactured by the harness rather than by the
    mutant. Public means: a free `pub fn`, a `pub fn` in an inherent impl, or any method
    of a `pub trait` or of a trait impl (those are public through the trait, with no
    `pub` of their own).
    """
    names, stack, pending, depth = [], [], None, 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//"):
            continue
        if TRAIT_RE.match(s):
            pending = "all" if s.startswith("pub") else "none"
        elif IMPL_RE.match(s):
            pending = "all" if " for " in IMPL_RE.match(s).group("head") else "pub"
        m = FN_RE.search(s)
        if m:
            ctx = stack[-1][1] if stack else None
            if ctx == "all" or (s.startswith("pub") and ctx in (None, "pub")):
                names.append(m.group(1))
        for ch in s:
            if ch == "{":
                depth += 1
                if pending is not None:
                    stack.append((depth, pending))
                    pending = None
            elif ch == "}":
                if stack and stack[-1][0] == depth:
                    stack.pop()
                depth -= 1
    return names


def iface_fns(project):
    names = []
    for f in iface_files(project):
        names += pub_fns(f.read_text(errors="ignore"))
    return sorted(set(names))


def crate_name(project):
    m = re.search(r'^name\s*=\s*"([^"]+)"',
                  (RBENCH / project / "Cargo.toml").read_text(), re.M)
    return m.group(1)


def lib_modules(project):
    """`pub mod x;` names from lib.rs -- the paths a test must `use`."""
    txt = (RBENCH / project / "src/lib.rs").read_text(errors="ignore")
    return re.findall(r"pub\s+mod\s+(\w+)\s*;", txt)


def impl_files(impl_dir):
    """The candidate's own Rust code: src/**/*.rs minus the shared interfaces and tests.

    These are the only files the Rust counterpart mutation (Stage 4b) may touch.
    """
    src = Path(impl_dir) / "src"
    out = []
    for f in sorted(src.rglob("*.rs")):
        rel = f.relative_to(impl_dir)
        if rel.parts[1] in ("interfaces", "bin") or rel.name == "lib.rs":
            continue
        out.append(rel.as_posix())
    return out


def impl_source(impl_dir, max_chars=140_000):
    parts, used = [], 0
    for rel in impl_files(impl_dir):
        text = (Path(impl_dir) / rel).read_text(errors="ignore")
        if used + len(text) > max_chars:
            parts.append(f"### `{rel}` (omitted, prompt budget)\n")
            continue
        used += len(text)
        parts.append(f"### `{rel}`\n\n```rust\n{text}\n```\n")
    return "\n".join(parts)


def reference(project):
    """The assumed-correct Rust implementation, or None if it is not usable."""
    d = REFERENCE / project
    return d if (d / "Cargo.toml").exists() else None


@functools.lru_cache(maxsize=1)
def gt_green_projects():
    """Projects whose reference implementation passed the ground-truth suite (Phase 0)."""
    with open(PHASE0 / "panel_raw.csv") as f:
        return frozenset(r["project"] for r in csv.DictReader(f)
                         if r["source"] == "codex" and r["status"] == "green")


def _inject(dst, test_rs):
    (dst / TEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (dst / TEST_PATH).write_text(test_rs)
    cargo = dst / "Cargo.toml"
    txt = cargo.read_text()
    if f'name = "{TEST_NAME}"' not in txt:
        txt += f'\n[[test]]\nname = "{TEST_NAME}"\npath = "{TEST_PATH}"\n'
        cargo.write_text(txt)


def run_test(impl_dir, test_rs, timeout=600, tag="0", patch=None, test_name=TEST_NAME):
    """Run one generated Rust test against one implementation.

    `patch` is {relpath: new_contents} applied to the copy -- this is how the Rust
    counterpart mutant M' is materialised. Never mutates impl_dir.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rmt_rs_") as td:
        dst = Path(td) / "proj"
        shutil.copytree(impl_dir, dst, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "target"))
        for rel, text in (patch or {}).items():
            (dst / rel).write_text(text)
        if test_rs is None:
            # Build-only probe: is the patched implementation still valid Rust?
            env = {**os.environ, "CARGO_TARGET_DIR": str(SCRATCH / tag)}
            r = subprocess.run(["cargo", "build"], cwd=dst, capture_output=True,
                               text=True, timeout=timeout, env=env)
            return {"status": "green" if r.returncode == 0 else "build_error",
                    "passed": 0, "failed": 0,
                    "out": "" if r.returncode == 0 else (r.stdout + r.stderr)[-4000:]}
        _inject(dst, test_rs)
        env = {**os.environ, "CARGO_TARGET_DIR": str(SCRATCH / tag)}
        try:
            r = subprocess.run(["cargo", "test", "--test", TEST_NAME,
                                "--", "--test-threads=1"],
                               cwd=dst, capture_output=True, text=True,
                               timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "passed": 0, "failed": 0, "out": ""}
    out = r.stdout + r.stderr
    passed = failed = 0
    for m in re.finditer(r"test result: \w+\.\s+(\d+) passed;\s+(\d+) failed;", out):
        passed += int(m.group(1))
        failed += int(m.group(2))
    if passed == 0 and failed == 0:
        status = "build_error" if r.returncode != 0 else "no_tests"
    elif failed:
        status = "test_failed"
    else:
        status = "green"
    return {"status": status, "passed": passed, "failed": failed,
            "out": out[-4000:] if status != "green" else ""}


def run_gt(impl_dir, patch=None, timeout=900, tag="gt"):
    """Run the RBench ground-truth suite (all `[[test]]` targets already in the tree).

    Used as a control on the Rust counterpart mutant M': if the ground-truth suite also
    passes M', then M' may be equivalent or only weakly different, and RT's failure to
    kill it says less about RT than it appears to.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rmt_gt_") as td:
        dst = Path(td) / "proj"
        shutil.copytree(impl_dir, dst, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "target"))
        for rel, text in (patch or {}).items():
            (dst / rel).write_text(text)
        env = {**os.environ, "CARGO_TARGET_DIR": str(SCRATCH / tag)}
        try:
            r = subprocess.run(["cargo", "test", "--", "--test-threads=1"], cwd=dst,
                               capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "passed": 0, "failed": 0}
    out = r.stdout + r.stderr
    passed = failed = 0
    for m in re.finditer(r"test result: \w+\.\s+(\d+) passed;\s+(\d+) failed;", out):
        passed += int(m.group(1))
        failed += int(m.group(2))
    if passed == 0 and failed == 0:
        status = "build_error" if r.returncode != 0 else "no_tests"
    else:
        status = "test_failed" if failed else "green"
    return {"status": status, "passed": passed, "failed": failed}


def compile_check(project, test_rs, tag="c"):
    """Type-check the generated test against the reference -- Stage 3's repair signal."""
    ref = reference(project)
    if ref is None:
        return {"status": "no_reference", "out": "", "passed": 0, "failed": 0}
    return run_test(ref, test_rs, tag=tag)
