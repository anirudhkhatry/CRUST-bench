"""Relabel an arm-comparison long CSV with verdicts from a re-run of the (fixed) GT suite.

    python analysis/relabel.py results/gt_diag_fixed  <long.csv>  <out_long.csv>  [--flips flips.csv]

Verdict: GT green -> good, anything else (test_failed, build_error, timeout, no_tests) -> bad.
Ports of benchmarks without a gt_diag_fixed entry keep their old verdict.
"""
import csv, json, sys, glob, collections
from pathlib import Path

diag_dir, src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
flips_path = Path(sys.argv[sys.argv.index("--flips") + 1]) if "--flips" in sys.argv else None

new = {}
for f in glob.glob(str(diag_dir / "*" / "*.json")):
    d = json.loads(Path(f).read_text())
    new[(d["bench"], d["impl"])] = ("good" if d["status"] == "green" else "bad", d["status"], d["verdict"])

rows = list(csv.DictReader(src.open()))
flips = collections.OrderedDict()
n_changed = 0
for r in rows:
    k = (r["benchmark"], r["implementation"])
    if k in new:
        v, status, old = new[k]
        if v != r["verdict"]:
            flips[k] = (r["verdict"], v, status)
            r["verdict"] = v
            n_changed += 1
with dst.open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"relabelled {len(new)} ports known; {len(flips)} implementations flipped ({n_changed} rows)")
c = collections.Counter((k[0], f"{a}->{b}") for k, (a, b, _) in flips.items())
for k, n in sorted(c.items()): print(f"  {k[0]:20s} {k[1]:10s} {n}")
if flips_path:
    with flips_path.open("w", newline="") as h:
        w = csv.writer(h); w.writerow(["benchmark", "implementation", "old", "new", "gt_status_fixed"])
        for (b, i), (a, n, s) in flips.items(): w.writerow([b, i, a, n, s])
