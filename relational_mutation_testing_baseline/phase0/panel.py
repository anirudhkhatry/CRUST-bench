"""Phase 0a -- measure the soundness panel.

For every RBench project, find each independent Rust implementation of it and
check whether that implementation passes the RBench ground-truth test suite.
Only GT-green implementations may serve as panel members (design spec S4/S6.1).

Builds go to a scratch CARGO_TARGET_DIR so the dataset trees stay clean.
Writes panel.json + panel.csv.
"""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench")
OUT = ROOT / "relational_mutation_testing_baseline/results/phase0"
SCRATCH = Path(os.environ.get("PHASE0_SCRATCH", "/tmp/rmt_phase0_target"))
JOBS = int(os.environ.get("PHASE0_JOBS", "12"))
TIMEOUT = int(os.environ.get("PHASE0_TIMEOUT", "900"))

# One entry per *independent* source of implementations. The `initial` run dirs are
# excluded: they are byte-identical to the corresponding `*_output` dirs (codex 98/100,
# claude 100/100), so they would inflate the panel without adding independence.
SOURCES = {
    "codex": ROOT / "datasets/codex_expts/all/initial_codex_output",
    "claude": ROOT / "datasets/claude_expts/all/initial_claude_output",
    "gpt55": ROOT / "mutation_testing/results/gpt/results_gpt55_4settings/_frozen_source",
    "opus5": ROOT / "mutation_testing/results/claude/results_opus5_4settings/_frozen_source",
    "gpt55_full": ROOT / "mutation_testing/results/gpt/results_gpt55_full/_frozen_source",
    "o4mini": ROOT / "mutation_testing/results/gpt/results_o4mini/_frozen_source",
}
# Same model family -> not independent evidence for the agreement oracle.
FAMILY = {
    "codex": "codex",
    "claude": "claude",
    "gpt55": "gpt5.5",
    "gpt55_full": "gpt5.5",
    "opus5": "opus5",
    "o4mini": "o4mini",
}


def sanitize(name):
    """Mirror run_mutation._sanitize_project_name closely enough to match frozen dirs."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return s if s[0].isalpha() or s[0] == "_" else "proj_" + s


def resolve(root, project):
    """Frozen-source trees use sanitized names; dataset trees use the raw name."""
    for cand in (project, sanitize(project), project.replace("-", "_"),
                 sanitize(project.replace("-", "_"))):
        p = root / cand
        if (p / "Cargo.toml").exists():
            return p
    return None


def impl_hash(proj_dir):
    """Hash the implementation .rs files only (interfaces and tests are shared)."""
    src = proj_dir / "src"
    if not src.is_dir():
        return None
    blobs = []
    for f in sorted(src.rglob("*.rs")):
        rel = f.relative_to(src)
        if rel.parts[0] in ("interfaces", "bin"):
            continue
        try:
            blobs.append(str(rel).encode() + b"\0" + f.read_bytes())
        except OSError:
            pass
    return hashlib.sha256(b"".join(blobs)).hexdigest()[:16] if blobs else None


def gt_green(proj_dir, tag):
    """Run the RBench ground-truth suite; return (status, passed, failed, detail)."""
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(SCRATCH / tag)
    try:
        r = subprocess.run(
            ["cargo", "test", "-j", "2", "--", "--test-threads=1"],
            cwd=proj_dir, capture_output=True, text=True, timeout=TIMEOUT, env=env,
        )
    except subprocess.TimeoutExpired:
        return "timeout", 0, 0, ""
    out = r.stdout + r.stderr
    passed = failed = 0
    for m in re.finditer(r"test result: \w+\.\s+(\d+) passed;\s+(\d+) failed;", out):
        passed += int(m.group(1))
        failed += int(m.group(2))
    if r.returncode != 0 and passed == 0 and failed == 0:
        status = "build_error"
    elif failed > 0:
        status = "test_failed"
    elif passed == 0:
        status = "no_tests"
    else:
        status = "green"
    return status, passed, failed, out[-1500:] if status != "green" else ""


def main():
    projects = sorted(d.name for d in (ROOT / "datasets/RBench").iterdir() if d.is_dir())
    jobs = []
    for project in projects:
        for source, root in SOURCES.items():
            d = resolve(root, project)
            if d is not None:
                jobs.append((project, source, d))

    print(f"{len(projects)} projects, {len(jobs)} candidate implementations, {JOBS} workers")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {
            ex.submit(gt_green, d, f"{source}_{i % JOBS}"): (project, source, d)
            for i, (project, source, d) in enumerate(jobs)
        }
        for n, fut in enumerate(as_completed(futs), 1):
            project, source, d = futs[fut]
            try:
                status, passed, failed, detail = fut.result()
            except Exception as e:  # noqa: BLE001 - record, never abort the sweep
                status, passed, failed, detail = "error", 0, 0, str(e)[:500]
            rows.append({
                "project": project, "source": source, "family": FAMILY[source],
                "path": str(d), "status": status, "passed": passed, "failed": failed,
                "impl_hash": impl_hash(d) or "", "detail": detail,
            })
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}", flush=True)

    with open(OUT / "panel_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Panel = GT-green, deduplicated by implementation hash, counted by model family.
    panel = {}
    for project in projects:
        mine = [r for r in rows if r["project"] == project]
        green = [r for r in mine if r["status"] == "green"]
        seen, uniq = set(), []
        for r in green:
            h = r["impl_hash"]
            if h and h in seen:
                continue
            seen.add(h)
            uniq.append(r)
        panel[project] = {
            "n_panel": len(uniq),
            "n_families": len({r["family"] for r in uniq}),
            "members": [{"source": r["source"], "family": r["family"], "path": r["path"]}
                        for r in uniq],
            "rejected": {r["source"]: r["status"] for r in mine if r["status"] != "green"},
        }

    (OUT / "panel.json").write_text(json.dumps(panel, indent=2))

    from collections import Counter
    print("\npanel size (GT-green, dedup) -> #projects:",
          dict(sorted(Counter(v["n_panel"] for v in panel.values()).items())))
    print("distinct model families    -> #projects:",
          dict(sorted(Counter(v["n_families"] for v in panel.values()).items())))
    print("status totals:", dict(Counter(r["status"] for r in rows)))
    print(f"\nwrote {OUT/'panel.json'} and {OUT/'panel_raw.csv'}")


if __name__ == "__main__":
    sys.exit(main())
