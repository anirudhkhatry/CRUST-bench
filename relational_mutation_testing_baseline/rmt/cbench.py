"""Build and run a CBench project against a generated test main.

Phase 0 established the shape (see results/phase0/REPORT.md S2): a uniform gcc path
over *library* sources only, header ancestors on the include path, progressive link
libraries, and a synthesized translation unit for header-only projects. The Makefile is
not a build path -- a `make` success gives no way to link a generated test main.

Everything happens in a throwaway copy of the project, so mutants never touch the
dataset. `Project.run(test_c, patch=...)` is the single primitive Stages 1/2 need:
apply a patch to the sources, compile the test main against them, run it.
"""
import itertools
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench")
CBENCH = ROOT / "datasets/CBench"

MAIN_RE = re.compile(r"^[ \t]*(?:int|void)[ \t\n]+main[ \t]*\(", re.M)

SKIP_DIRS = {
    ".git", "build", "target", "test", "tests", "unit_tests", "testing",
    "bin", "app", "apps", "demo", "demos", "example", "examples",
    "benchmark", "benchmarks", "bench", "fuzz", "tools", "vendor", "third_party",
    "unity", "contrib", "doc", "docs",
}

LIB_SETS = (["-lm"], ["-lm", "-lpthread"], ["-lm", "-lpthread", "-lcrypto", "-lssl"])

# -fcommon restores the pre-gcc-10 handling of tentative definitions (`int nb_vars;` in
# two headers-sharing TUs). Projects that ship several programs out of one source tree
# rely on it; gcc's -fno-common default turns that into "multiple definition" at link.
CFLAGS = ["-w", "-std=gnu11", "-g", "-O0", "-fcommon"]

# Compile-flag fallback chain. Hand-written SIMD kernels use `always_inline` intrinsics
# that gcc refuses to inline unless the target supports them ("target specific option
# mismatch"); the project's own Makefile passes -march. We try plain first so the 90
# projects that need nothing keep their flags, then widen.
CFLAG_SETS = (
    CFLAGS,
    CFLAGS + ["-march=native"],
    CFLAGS + ["-march=native", "-std=gnu99"],
)

# A single-header library ships a `.c` that is nothing but
#     #define SKP_IMPLEMENTATION
#     #include "skp.h"
# The implementation is the header; the `.c` exists only to give it a translation unit.
# Counting real code lines tells such a shim from an actual source file.
MIN_IMPL_LINES = 30

# `#include "foo.c"` -- the included file is not its own translation unit.
INCLUDED_C_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^">]+\.c)[>"]', re.M)
INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^">]+)[>"]', re.M)
# ld's --wrap protocol: a source defining __wrap_X / calling __real_X needs -Wl,--wrap=X.
WRAP_RE = re.compile(r"\b__(?:wrap|real)_(\w+)\b")
UNDEF_RE = re.compile(r"undefined reference to [`'\"]([A-Za-z_]\w*)")

# --- per-project overrides -------------------------------------------------------
# Data, not code: extra cflags / include dirs (relative to the project) / link flags /
# libs / skipped path fragments / force-included sources. Everything here is a last
# resort -- prefer widening the generic rules above.
PROJECT_FLAGS = {
    # e.g. "some_proj": {"cflags": ["-DFOO=1"], "includes": ["compat"],
    #                    "ldflags": ["-Wl,--allow-multiple-definition"],
    #                    "libs": ["-ldl"], "skip": ["src/generated/"],
    #                    "sources": ["compat/shim.c"]},
}

# RBench uses a different name than the CBench directory for a handful of projects.
# Dash/underscore differences normalise away (see `resolve_name`); these do not.
PROJECT_ALIASES = {
    "proj_2DPartInt": "2DPartInt",
    "proj_42_Kocaeli_Printf": "42-Kocaeli-Printf",
}


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_name(name):
    """Map an RBench project name onto its CBench directory name."""
    if (CBENCH / name).is_dir():
        return name
    if name in PROJECT_ALIASES:
        return PROJECT_ALIASES[name]
    want = _norm(name)
    for d in sorted(CBENCH.iterdir()):
        if d.is_dir() and _norm(d.name) == want:
            return d.name
    return name


def _lib_root(proj):
    return proj / "src" if (proj / "src").is_dir() else proj


def _walk(proj, suffix):
    out = []
    for f in sorted(_lib_root(proj).rglob(f"*{suffix}")):
        if any(part in SKIP_DIRS for part in f.relative_to(proj).parts):
            continue
        out.append(f)
    return out


def _all_sources(proj):
    """Every .c under the project, skipping only VCS/build output."""
    out = []
    for f in sorted(proj.rglob("*.c")):
        if any(p in (".git", "build", "target") for p in f.relative_to(proj).parts):
            continue
        out.append(f)
    return out


def _included_c_files(proj, texts):
    """Relative paths of .c files that another translation unit `#include`s.

    Amalgamation builds are common in CBench (`xopt.c` includes `snprintf.c`, `for.c`
    includes the generated `for-gen.c`, `tisp.c` includes `tib/*.c`, `vectors.c`
    includes `vec.c` once per type). Such a file is not a standalone TU: compiling it
    separately either fails outright or produces duplicate symbols at link time.
    """
    by_rel = {f.relative_to(proj).as_posix(): f for f in _all_sources(proj)}
    included = set()
    for text in texts:
        for inc in INCLUDED_C_RE.findall(text):
            inc = inc.lstrip("./")
            for rel in by_rel:
                if rel == inc or rel.endswith("/" + inc):
                    included.add(rel)
    return included


def _wrap_flags(texts):
    syms = set()
    for text in texts:
        syms.update(WRAP_RE.findall(text))
    return [f"-Wl,--wrap={s}" for s in sorted(syms)]


def _defines(text, sym):
    """Does `text` look like it *defines* sym (not merely calls it)?"""
    pat = re.compile(r"^[^#/\n]*?\b" + re.escape(sym) + r"[ \t]*\([^;]*$", re.M)
    return bool(pat.search(text))


SHIM_DIR = "_rmt_shims"


def _write_include_shims(proj, texts, incdirs):
    """Satisfy `#include`s that expect an installed/generated layout.

    Projects are written against their `make install` layout (`<gfc/gfc.h>` for what is
    really `src/gfc.h`) or against a file the Makefile generates from a committed
    variant (`tibs.tsp.h` from `tibs_tsp.h`). Rather than special-case each, any include
    that resolves nowhere on the include path but whose name matches a project header --
    exactly, or once punctuation is normalised -- gets a one-line forwarding shim.
    """
    headers = [f for f in proj.rglob("*.h")
               if not any(p in (".git", "build", "target") for p in f.relative_to(proj).parts)]
    by_base, by_norm = {}, {}
    for h in headers:
        by_base.setdefault(h.name, h)
        by_norm.setdefault(_norm(h.name), h)

    wanted = set()
    for text in texts:
        for inc in INCLUDE_RE.findall(text):
            if not inc.endswith((".h", ".hpp")):
                continue
            if any((d / inc).exists() for d in incdirs):
                continue
            wanted.add(inc)

    shim = proj / SHIM_DIR
    made = False
    for inc in sorted(wanted):
        base = Path(inc).name
        target = by_base.get(base) or by_norm.get(_norm(base))
        if target is None or target.name == base and (proj / inc) == target:
            continue
        dest = shim / inc
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f'#include "{target}"\n')
        made = True
    return shim if made else None


def _include_dirs(proj):
    dirs = {proj, proj / "src", proj / "include"}
    for f in proj.rglob("*.h"):
        if any(part in (".git", "build", "target") for part in f.parts):
            continue
        d = f.parent
        while True:
            dirs.add(d)
            if d == proj or proj not in d.parents:
                break
            d = d.parent
    return sorted(d for d in dirs if d.is_dir())


@dataclass
class RunResult:
    stage: str          # "compile" | "link" | "run" | "ok"
    ok: bool
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    detail: str = ""    # compiler/linker diagnostics, truncated

    @property
    def passed(self):
        """A test main passes iff it built AND exited 0."""
        return self.ok and self.returncode == 0


HEADER_TU = "_rmt_header_tu.c"


class Project:
    """A CBench project prepared for repeated build-and-run cycles."""

    def __init__(self, name, workdir=None):
        self.name = resolve_name(name)
        self.path = CBENCH / self.name
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        self._tmp = None
        if workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix=f"rmt_{name}_")
            workdir = Path(self._tmp.name)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.over = PROJECT_FLAGS.get(self.name, {})
        srcs = _walk(self.path, ".c")
        self.headers = [f.relative_to(self.path).as_posix() for f in _walk(self.path, ".h")]
        self.sources = [f.relative_to(self.path).as_posix() for f in srcs]
        self.mains = [s for s in self.sources
                      if MAIN_RE.search((self.path / s).read_text(errors="ignore"))]

        # Scan for amalgamation includes from library candidates only. A driver or test
        # TU may `#include` the library source (mdb's testmdb_int.c includes mdb.c);
        # that must not disqualify the library source itself.
        texts = [self.read(s) for s in self.sources
                 if s not in self.mains] + [self.read(h) for h in self.headers]
        included = _included_c_files(self.path, texts)
        skip = tuple(self.over.get("skip", ()))
        # Header-only covers two shapes: no .c at all, and a lone .c that is just a
        # driver main (Remimu). Either way the library lives in the headers.
        self.lib_sources = [s for s in self.sources
                            if s not in self.mains and s not in included
                            and not (skip and s.startswith(skip))]
        self.header_only = not self.lib_sources
        # Sources outside the library set that the linker may still need (see
        # `_missing_symbol_sources`): a helper package next to src/, or the project's own
        # test-assertion framework that library files call into.
        self.extra_sources = list(self.over.get("sources", ()))
        self._code_len = {}    # relpath -> code-line count, for mutable_files
        self.link_libs = None  # discovered on first successful link, then reused
        self.cflags = None     # first CFLAG_SETS entry that compiled everything
        self.ldflags = _wrap_flags(texts) + list(self.over.get("ldflags", ()))
        # One Project is shared by all of a project's mutants, which run concurrently,
        # so build directories are handed out from an atomic counter.
        self._seq = itertools.count()

    def close(self):
        if self._tmp:
            self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- the mutable files a mutation may target -------------------------------
    def _code_lines(self, relpath):
        """Lines that are neither blank, nor a comment, nor a preprocessor directive."""
        if relpath not in self._code_len:
            n = 0
            for line in self.read(relpath).splitlines():
                s = line.strip()
                if s and not s.startswith(("#", "//", "*", "/*")):
                    n += 1
            self._code_len[relpath] = n
        return self._code_len[relpath]

    @property
    def mutable_files(self):
        """Files Stage 1 may mutate: whichever of the sources or headers holds the code.

        Normally the library `.c` files. But a single-header library's sources are
        `#define IMPLEMENTATION` shims (skp) or the library has no source at all beyond a
        driver main (Remimu) -- there the code, and so the mutation target, is the header.
        Headers are compiled either way (included by the shim, or via HEADER_TU), so a
        patch to one takes effect without any change to the build.
        """
        if any(self._code_lines(s) >= MIN_IMPL_LINES for s in self.lib_sources):
            return self.lib_sources
        hdrs = [h for h in self.headers if self._code_lines(h) >= MIN_IMPL_LINES]
        return hdrs or self.lib_sources or self.headers

    @property
    def mutation_target(self):
        """"sources" or "headers" -- where `mutable_files` came from."""
        return "sources" if set(self.mutable_files) <= set(self.lib_sources) else "headers"

    def read(self, relpath):
        return (self.path / relpath).read_text(errors="ignore")

    def _missing_symbol_sources(self, ld_stderr):
        """Sources outside the library set that define symbols the link is missing.

        Only files that actually define one of the reported symbols are considered, so
        this cannot drag in a demo or benchmark TU wholesale -- it recovers exactly the
        helper package (`hash-table/` next to `src/`) or in-repo assertion framework
        (`src/testing/`) that library code depends on.
        """
        syms = set(UNDEF_RE.findall(ld_stderr))
        if not syms:
            return []
        taken = set(self.lib_sources) | set(self.extra_sources) | set(self.mains)
        out = []
        for f in _all_sources(self.path):
            rel = f.relative_to(self.path).as_posix()
            if rel in taken:
                continue
            text = f.read_text(errors="ignore")
            if MAIN_RE.search(text):
                continue  # a second main() would collide with the test main
            if any(_defines(text, s) for s in syms):
                out.append(rel)
        return out

    # -- build ---------------------------------------------------------------
    def run(self, test_c=None, patch=None, timeout=20, argv=(), stdin=""):
        """Compile the library (with `patch` applied) plus `test_c`, then run it.

        patch: {relpath: new_contents}. test_c: source text providing main().
        With test_c=None this is a build-only probe (link against a stub main).
        """
        d = self.workdir / f"b{next(self._seq)}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(self.path, d / "proj", symlinks=True,
                        ignore=shutil.ignore_patterns(".git"))
        proj = d / "proj"
        for rel, text in (patch or {}).items():
            (proj / rel).write_text(text)

        incdirs = _include_dirs(proj)
        shim = _write_include_shims(
            proj, [(proj / s).read_text(errors="ignore")
                   for s in self.lib_sources + self.headers], incdirs)
        incs = [f"-I{p}" for p in ([shim] if shim else []) + incdirs]
        incs += [f"-I{proj / p}" for p in self.over.get("includes", ())]

        srcs = [proj / s for s in self.lib_sources]
        if self.header_only:
            tu = proj / HEADER_TU
            tu.write_text("".join(f'#include "{proj / h}"\n' for h in self.headers))
            srcs = [tu]

        main_c = d / "_rmt_main.c"
        main_c.write_text(test_c if test_c is not None else "int main(void){return 0;}\n")
        srcs = srcs + [proj / s for s in self.extra_sources] + [main_c]

        objs, fail = [], None
        for base in ([self.cflags] if self.cflags else CFLAG_SETS):
            cflags = list(base) + list(self.over.get("cflags", ()))
            objs, fail = [], None
            for f in srcs:
                obj = d / (f.name + f".{len(objs)}.o")
                r = subprocess.run(["gcc", "-c", *cflags, str(f), "-o", str(obj), *incs],
                                   capture_output=True, text=True, timeout=180)
                if r.returncode != 0:
                    who = "test main" if f == main_c else f.name
                    fail = f"[{who}]\n{r.stderr[-3000:]}"
                    break
                objs.append(str(obj))
            if fail is None:
                self.cflags = list(base)
                break
        if fail is not None:
            return RunResult("compile", False, detail=fail)

        exe = d / "_rmt_exe"
        # Link, and on undefined symbols pull in the sources that define them (once per
        # project; the set is then reused). This is what makes a project whose library
        # calls a helper living outside the library root linkable, without widening the
        # default source set for every project.
        for _ in range(4):
            tries = [self.link_libs] if self.link_libs else list(LIB_SETS)
            for libs in tries:
                r = subprocess.run(["gcc", *objs, "-o", str(exe), *self.ldflags, *libs],
                                   capture_output=True, text=True, timeout=180)
                if r.returncode == 0:
                    self.link_libs = libs
                    break
            if r.returncode == 0:
                break
            new = self._missing_symbol_sources(r.stderr)
            if not new:
                return RunResult("link", False, detail=r.stderr[-3000:])
            for s in new:
                obj = d / (Path(s).name + f".x{len(objs)}.o")
                c = subprocess.run(["gcc", "-c", *(self.cflags or CFLAGS),
                                    *self.over.get("cflags", ()),
                                    str(proj / s), "-o", str(obj), *incs],
                                   capture_output=True, text=True, timeout=180)
                if c.returncode != 0:
                    return RunResult("link", False, detail=r.stderr[-3000:])
                objs.append(str(obj))
                self.extra_sources.append(s)
        else:
            return RunResult("link", False, detail=r.stderr[-3000:])

        try:
            p = subprocess.run([str(exe), *argv], capture_output=True, text=True,
                               timeout=timeout, input=stdin,
                               env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"})
        except subprocess.TimeoutExpired:
            return RunResult("run", True, returncode=124, stderr="TIMEOUT")
        return RunResult("ok", True, returncode=p.returncode,
                         stdout=p.stdout[-8000:], stderr=p.stderr[-4000:])


def usable_projects():
    """Projects Phase 0 marked drivable, joined with their panel size."""
    import csv
    rows = list(csv.DictReader(
        open(ROOT / "relational_mutation_testing_baseline/results/phase0/eligible.csv")))
    return [r for r in rows if r["c_usable"] == "True"]
