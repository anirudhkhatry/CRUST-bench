"""Why is each exported port 'bad'?  Re-run the RBench ground-truth suite on it and keep
the failing test names, panic messages and build errors.  Compute only, no LLM.

    python analysis/gt_diag.py -b rhbloom -b dict --jobs 12 -o results/gt_diag

Writes results/gt_diag/<bench>/<impl>.json and a merged results/gt_diag/summary.json.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench")
EXPORTS = ROOT / "exports"
RBENCH = Path(os.environ.get("GT_RBENCH", ROOT / "datasets/RBench"))
SCRATCH = Path("/tmp/gt_diag_target")

PANIC_RE = re.compile(r"thread '([^']+)' \(?\d*\)? ?panicked at ([^\n]+):\n([^\n]*)")
FAIL_RE = re.compile(r"^test (\S+) \.\.\. FAILED", re.M)
OK_RE = re.compile(r"^test (\S+) \.\.\. ok", re.M)
RES_RE = re.compile(r"test result: \w+\.\s+(\d+) passed;\s+(\d+) failed;")
ERR_RE = re.compile(r"^(error(\[E\d+\])?: .*)$", re.M)
RUNNING_RE = re.compile(r"^\s+Running (\S+) \(", re.M)


def _ignore(d, names):
    return {n for n in names if n in ("target", ".git")}


def assemble(bench, impl_dir, dst):
    shutil.copytree(impl_dir, dst, symlinks=True, ignore=_ignore)
    rb = RBENCH / bench
    binsrc = rb / "src/bin"
    if not binsrc.is_dir():
        return []
    shutil.rmtree(dst / "src/bin", ignore_errors=True)
    shutil.copytree(binsrc, dst / "src/bin")
    # fixtures beside the tests (rhbloom-style data files) live in src/bin already.
    # Exactly what export_implementations.py did: the benchmark's src/bin, cargo's own
    # target discovery, a debug build. (A --release [[test]] harness made good ports fail.)
    return [f.stem for f in sorted(binsrc.glob("*.rs"))]


def run_one(bench, impl_dir, slot, timeout):
    with tempfile.TemporaryDirectory(prefix="gt_diag_") as td:
        dst = Path(td) / "proj"
        names = assemble(bench, impl_dir, dst)
        env = {**os.environ, "CARGO_TARGET_DIR": str(Path(td) / "target"),
               "RUST_BACKTRACE": "0"}
        cmd = ["cargo", "test", "--no-fail-fast", "--", "--test-threads=1"]
        try:
            r = subprocess.run(cmd, cwd=dst, capture_output=True, text=True,
                               timeout=timeout, env=env)
            out = r.stdout + r.stderr
            rc = r.returncode
        except subprocess.TimeoutExpired as e:
            out = ((e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")) + \
                  ((e.stderr or b"").decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or ""))
            rc = None
    passed = sum(int(m.group(1)) for m in RES_RE.finditer(out))
    failed = sum(int(m.group(2)) for m in RES_RE.finditer(out))
    fails = FAIL_RE.findall(out)
    oks = OK_RE.findall(out)
    panics = [{"test": m.group(1), "at": m.group(2).strip(), "msg": m.group(3).strip()[:300]}
              for m in PANIC_RE.finditer(out)]
    errors = [m.group(1)[:300] for m in ERR_RE.finditer(out)
              if "could not compile" not in m.group(1) and "aborting" not in m.group(1)]
    if rc is None:
        status = "timeout"
    elif passed == 0 and failed == 0:
        status = "build_error" if rc != 0 else "no_tests"
    else:
        status = "test_failed" if failed else "green"
    return {"bench": bench, "impl": impl_dir.name, "verdict": impl_dir.parent.name,
            "status": status, "rc": rc, "passed": passed, "failed": failed,
            "failed_tests": fails, "ok_tests": oks, "panics": panics,
            "build_errors": errors[:12], "targets": names,
            "tail": out[-3000:] if status != "green" else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--benchmark", action="append", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--verdict", default="bad", choices=["bad", "good", "both"])
    ap.add_argument("-o", "--out", default="results/gt_diag")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rbench", default=None, help="alternative RBench root (e.g. datasets/RBench_fixed)")
    a = ap.parse_args()
    global RBENCH
    if a.rbench:
        RBENCH = Path(a.rbench).resolve()
    out = Path(a.out)
    jobs = []
    for b in a.benchmark:
        for v in (["bad", "good"] if a.verdict == "both" else [a.verdict]):
            for d in sorted((EXPORTS / b / v).glob("*")):
                if (d / "Cargo.toml").exists():
                    jobs.append((b, d))
    print(f"{len(jobs)} ports", flush=True)
    results = []
    def work(i, b, d):
        f = out / b / f"{d.parent.name}__{d.name}.json"
        if f.exists() and not a.refresh:
            return json.loads(f.read_text())
        r = run_one(b, d, i % a.jobs, a.timeout)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, indent=1))
        return r
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(work, i, b, d): (b, d) for i, (b, d) in enumerate(jobs)}
        for fu in as_completed(futs):
            b, d = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                print(f"  {b}/{d.name}: ERROR {e}", flush=True); continue
            results.append(r)
            first = (r["panics"][0]["msg"] if r["panics"] else
                     (r["build_errors"][0] if r["build_errors"] else ""))
            print(f"  {b}/{d.parent.name}/{d.name}: {r['status']} "
                  f"p={r['passed']} f={r['failed']} {first[:90]}", flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
