"""Measure the mutation-score ceiling: RBench ground-truth tests vs the frozen
generated source already produced under results_gpt55/_frozen_source.

Reuses helpers from run_mutation. No LLM calls -- ground-truth tests are used
as-is; any that don't compile/pass against the generated source are pruned
(no repair), exactly like a real "ideal tests" baseline would behave.
"""
import sys
import json
import shutil
import subprocess
from pathlib import Path

sys.path.append(str(Path("/nas/CRUST-bench-COLM/CRUST-bench/src")))
# Scripts live one level down; put the study root on sys.path so ``harness`` imports
# work when this file is run directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness.run_mutation as rm  # noqa: E402  (adds src to path itself)
from harness.paths import RESULTS as _RESULTS  # noqa: E402

RUN = _RESULTS / "gpt/results_gpt55"
FROZEN = RUN / "_frozen_source"
RBENCH = Path("/nas/CRUST-bench-COLM/CRUST-bench/datasets/RBench")
GT_DIR = RUN / "ground_truth"
SHARD = "1/8"
JOBS = 16


def resolve_rbench(project_name):
    """Find the RBench dir for a (sanitized) project name, trying common variants."""
    cands = [
        project_name,
        project_name[0].upper() + project_name[1:],
        project_name.replace("proj_", ""),
        project_name.replace("_", "-"),
        project_name.replace("proj_", "").replace("_", "-"),
    ]
    for c in cands:
        if (RBENCH / c).exists():
            return RBENCH / c
    # last resort: match by sanitized name
    for d in RBENCH.iterdir():
        if d.is_dir() and rm._sanitize_project_name(d.name) == project_name:
            return d
    return None


def prune_failing_tests(proj):
    """Keep only ground-truth test bins that compile and pass against this source."""
    bin_dir = proj / "src" / "bin"
    if not bin_dir.exists():
        return 0, 0
    bins = sorted(bin_dir.glob("*.rs"))
    kept = 0
    for f in bins:
        passed, _ = rm._cargo_test_bin(proj, f.stem)
        if passed:
            kept += 1
        else:
            f.unlink()
    return kept, len(bins)


def main():
    projects = sorted(d.name for d in FROZEN.iterdir() if d.is_dir() and d.name != "target")
    GT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for p in projects:
        rb = resolve_rbench(p)
        gt_tests = sorted((rb / "src" / "bin").glob("*.rs")) if rb else []
        if not gt_tests:
            print(f"{p}: no RBench ground-truth tests found; skipping")
            results[p] = {"baseline_valid": False, "reason": "no_ground_truth_tests"}
            continue

        dst = GT_DIR / p
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(FROZEN / p, dst, ignore=shutil.ignore_patterns("target", "mutants.out"))
        bin_dir = dst / "src" / "bin"
        if bin_dir.exists():
            shutil.rmtree(bin_dir)
        bin_dir.mkdir(parents=True)
        for t in gt_tests:
            content = t.read_text(encoding="utf-8", errors="ignore")
            if "fn main()" not in content:
                content += "\nfn main() {}\n"
            (bin_dir / t.name).write_text(content)

        rm.create_top_level_cargo_toml(GT_DIR)

        kept, total = prune_failing_tests(dst)
        print(f"{p}: ground-truth tests kept {kept}/{total}")
        if kept == 0:
            results[p] = {"baseline_valid": False, "reason": "no_passing_tests",
                          "tests_kept": 0, "tests_total": total}
            continue

        out_dir = dst / "mutants.out"
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        cmd = [
            "cargo", "mutants", "--no-shuffle", "-j", str(JOBS),
            "--output", str(dst),
            "--exclude", "**/src/bin/**", "--exclude", "**/src/interfaces/**",
            "--shard", SHARD,
        ]
        subprocess.run(cmd, cwd=dst, capture_output=True, text=True, timeout=3600)
        oc = out_dir / "outcomes.json"
        if not oc.exists():
            results[p] = {"baseline_valid": False, "reason": "no_outcomes",
                          "tests_kept": kept, "tests_total": total}
            continue
        s = rm._summarize_outcomes(oc)
        s.update({"baseline_valid": True, "tests_kept": kept, "tests_total": total})
        results[p] = s
        print(f"  {p}: killed {s['killed']}/{s['viable_mutants']} (score {s['mutation_score']:.1%})")

    valid = [r for r in results.values() if r.get("baseline_valid")]
    avg = sum(r["mutation_score"] for r in valid) / len(valid) if valid else None
    report = {"setting": "ground_truth", "n_valid": len(valid),
              "avg_mutation_score": avg, "per_project": results}
    (RUN / "ground_truth_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== ground_truth over %d valid projects: %s ===" % (
        len(valid), f"{avg:.2%}" if avg is not None else "n/a"))


if __name__ == "__main__":
    main()
