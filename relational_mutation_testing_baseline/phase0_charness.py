"""Phase 0b -- verify the C build harness (design spec S5).

Strategy is Makefile-first with a uniform gcc fallback. Stages 1/2 of the pipeline need
to link a *generated test main* against the project's objects, so what actually matters
is not "does the project build" but:

  1. do all of src/**/*.c compile to objects, and
  2. can we link those objects (minus the project's own main) against a test main.

We probe exactly that with a trivial stub test, so a project marked OK here is one the
real pipeline can drive.

Writes charness.json + charness.csv.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench")
CBENCH = ROOT / "datasets/CBench"
OUT = ROOT / "relational_mutation_testing_baseline/results/phase0"
JOBS = int(os.environ.get("PHASE0_JOBS", "12"))
TIMEOUT = 180

MAIN_RE = re.compile(r"^[ \t]*(?:int|void)[ \t\n]+main[ \t]*\(", re.M)

# Directories that hold demos, benchmarks, vendored test frameworks and CLI front-ends.
# The pipeline mutates *library* code (what the Rust interfaces correspond to), so these
# are not just unnecessary -- pulling them in is what made the first sweep fail.
SKIP_DIRS = {
    ".git", "build", "target", "test", "tests", "unit_tests", "testing",
    "bin", "app", "apps", "demo", "demos", "example", "examples",
    "benchmark", "benchmarks", "bench", "fuzz", "tools", "vendor", "third_party",
    "unity", "contrib", "doc", "docs",
}


def lib_root(proj):
    """The directory holding library code: src/ when present, else the project root."""
    return proj / "src" if (proj / "src").is_dir() else proj


def c_sources(proj):
    """Library C sources. Header-only projects return [] and are handled by c_headers()."""
    root = lib_root(proj)
    out = []
    for f in sorted(root.rglob("*.c")):
        rel = f.relative_to(proj)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out.append(f)
    return out


def c_headers(proj):
    """Library headers -- the implementation itself for header-only projects."""
    root = lib_root(proj)
    out = []
    for f in sorted(root.rglob("*.h")):
        rel = f.relative_to(proj)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out.append(f)
    return out


def include_dirs(proj):
    """Header dirs *and their ancestors* -- sources commonly #include "pkg/pkg.h"."""
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


def try_make(proj, workdir):
    """Attempt the project's own build. Non-destructive: build in a copy."""
    if not any((proj / n).exists() for n in ("Makefile", "makefile", "GNUmakefile")):
        return "no_makefile", ""
    copy = workdir / "mk" / proj.name
    try:
        shutil.copytree(proj, copy, symlinks=True)
    except OSError as e:
        return "copy_failed", str(e)[:300]
    try:
        r = subprocess.run(["make", "-j", "2"], cwd=copy, capture_output=True,
                           text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    return ("ok" if r.returncode == 0 else "failed"), (r.stdout + r.stderr)[-1200:]


def try_gcc(proj, workdir):
    """Uniform fallback: compile every source to an object, then link a stub test main."""
    srcs = c_sources(proj)
    incs = [f"-I{d}" for d in include_dirs(proj)]
    objdir = workdir / "obj" / proj.name
    objdir.mkdir(parents=True, exist_ok=True)
    mode = "sources"

    if not srcs:
        # Header-only library: the implementation lives in src/*.h. Synthesize one
        # translation unit that includes them so there is something to compile and mutate.
        hdrs = c_headers(proj)
        if not hdrs:
            return {"status": "no_sources", "n_sources": 0, "n_compiled": 0,
                    "mains": [], "detail": "", "mode": "none"}
        tu = objdir / "_header_only_tu.c"
        tu.write_text("".join(f'#include "{h}"\n' for h in hdrs))
        srcs = [tu]
        mode = "header_only"

    def rel(f):
        # The synthesized header-only TU lives in the scratch dir, not under proj.
        return str(f.relative_to(proj)) if f.is_relative_to(proj) else f.name

    compiled, failed, mains, detail = [], [], [], []
    for f in srcs:
        try:
            if MAIN_RE.search(f.read_text(errors="ignore")):
                mains.append(rel(f))
        except OSError:
            pass
        obj = objdir / (rel(f).replace("/", "_") + ".o")
        r = subprocess.run(
            ["gcc", "-c", "-w", "-std=gnu11", "-g", "-O0", str(f), "-o", str(obj)] + incs,
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        if r.returncode == 0:
            compiled.append((f, obj))
        else:
            failed.append(rel(f))
            if len(detail) < 3:
                detail.append(r.stderr[-400:])

    result = {"n_sources": len(srcs), "n_compiled": len(compiled), "mode": mode,
              "mains": mains, "failed_files": failed[:10]}

    if failed:
        result["status"] = "compile_failed"
        result["detail"] = "\n".join(detail)
        return result

    # Link probe: objects minus the project's own main(s), plus a stub test main.
    stub = objdir / "_stub_test.c"
    stub.write_text("int main(void){return 0;}\n")
    r = subprocess.run(["gcc", "-c", "-w", str(stub), "-o", str(objdir / "_stub.o")],
                       capture_output=True, text=True, timeout=TIMEOUT)
    objs = [str(o) for f, o in compiled if rel(f) not in mains]
    objs.append(str(objdir / "_stub.o"))
    # Try progressively more system libraries; record which set was needed so the real
    # pipeline can reuse it per project.
    for libs in (["-lm"], ["-lm", "-lpthread"], ["-lm", "-lpthread", "-lcrypto", "-lssl"]):
        r = subprocess.run(["gcc"] + objs + ["-o", str(objdir / "_probe")] + libs,
                           capture_output=True, text=True, timeout=TIMEOUT)
        if r.returncode == 0:
            result["status"] = "ok"
            result["detail"] = ""
            result["link_libs"] = " ".join(libs)
            return result
    result["status"] = "link_failed"
    result["detail"] = r.stderr[-1200:]
    return result


def probe(proj):
    with tempfile.TemporaryDirectory(prefix="rmt_c_") as td:
        workdir = Path(td)
        mk_status, mk_detail = try_make(proj, workdir)
        gcc = try_gcc(proj, workdir)
    usable = gcc["status"] == "ok"
    return {
        "project": proj.name,
        "make": mk_status,
        "gcc": gcc["status"],
        "mode": gcc.get("mode", ""),
        "link_libs": gcc.get("link_libs", ""),
        "n_sources": gcc["n_sources"],
        "n_compiled": gcc["n_compiled"],
        "n_mains": len(gcc.get("mains", [])),
        "mains": ";".join(gcc.get("mains", [])),
        # The pipeline needs the link probe, which only the gcc path provides.
        "usable": usable,
        "build_path": "gcc" if usable else ("make_only" if mk_status == "ok" else "none"),
        "detail": (gcc.get("detail") or "")[:1200],
        "make_detail": mk_detail[:600] if mk_status not in ("ok", "no_makefile") else "",
    }


def main():
    projects = sorted(d for d in CBENCH.iterdir() if d.is_dir())
    print(f"{len(projects)} CBench projects, {JOBS} workers")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(probe, p): p for p in projects}
        for n, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                rows.append({"project": p.name, "make": "error", "gcc": "error",
                             "mode": "", "link_libs": "",
                             "n_sources": 0, "n_compiled": 0, "n_mains": 0, "mains": "",
                             "usable": False, "build_path": "none",
                             "detail": str(e)[:500], "make_detail": ""})
            if n % 20 == 0 or n == len(projects):
                print(f"  {n}/{len(projects)}", flush=True)

    rows.sort(key=lambda r: r["project"])
    with open(OUT / "charness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "charness.json").write_text(json.dumps(rows, indent=2))

    from collections import Counter
    print("\ngcc-fallback status:", dict(Counter(r["gcc"] for r in rows)))
    print("make status:        ", dict(Counter(r["make"] for r in rows)))
    print("usable by pipeline: ", sum(r["usable"] for r in rows), "/", len(rows))
    print(f"wrote {OUT/'charness.csv'}")


if __name__ == "__main__":
    sys.exit(main())
