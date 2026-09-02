"""Line-accurate coverage/oracle decomposition of the src+test -> c+rust differential.

For each differential mutant (src+test missed & c+rust caught), run cargo llvm-cov on
the src+test project's (green) test suite and check whether the mutated LINE was
executed. Not executed -> true coverage gap; executed -> oracle weakness.
"""
import json, glob, os, re, subprocess, collections, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.paths import RESULTS  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.5"
R = str(RESULTS / ("gpt/results_gpt55_4settings" if MODEL == "gpt-5.5"
                   else "claude/results_opus5_4settings"))
DIR = {"src+test": "src+test", "c+rust": "c_and_rust"}
env = dict(os.environ, CARGO_TARGET_DIR=f"{R}/_shared_target")

def load(setting, p, which):
    fp = os.path.join(R, DIR[setting], p, "mutants.out", which + ".txt")
    return set(l.strip() for l in open(fp) if l.strip()) if os.path.isfile(fp) else None

def loc(mut):  # -> (relpath, line)   from "proj/src/f.rs:L:C: ..."
    m = re.match(r"(.+?\.rs):(\d+):\d+:", mut)
    if not m: return None
    path, line = m.group(1), int(m.group(2))
    rel = path.split("/", 1)[1] if "/" in path else path   # strip leading "<proj>/"
    return rel, line

projs = sorted(set(os.path.basename(os.path.dirname(os.path.dirname(m)))
                   for m in glob.glob(R + "/c_and_rust/*/mutants.out/caught.txt")))
cov_gap = collections.Counter(); oracle = collections.Counter()
n_cov = n_orc = n_err = 0
per_proj = {}
for p in projs:
    stm = load("src+test", p, "missed"); crc = load("c+rust", p, "caught")
    if stm is None or crc is None: continue
    diff = stm & crc
    if not diff: continue
    pdir = os.path.join(R, "src+test", p)
    covjson = f"/tmp/cov_{MODEL}_{p}.json"
    try:
        r = subprocess.run(["cargo", "llvm-cov", "--json", "--output-path", covjson],
                           cwd=pdir, env=env, capture_output=True, text=True, timeout=900)
        d = json.load(open(covjson))
    except Exception as e:
        print(f"  [skip] {p}: llvm-cov failed ({type(e).__name__})"); n_err += len(diff); continue
    covered = set()  # (relpath, line) executed at least once
    for fdata in d["data"][0]["files"]:
        fn = fdata["filename"]
        if f"/src+test/{p}/" not in fn: continue
        rel = fn.split(f"/src+test/{p}/", 1)[1]
        for seg in fdata.get("segments", []):
            line, _c, count, has = seg[0], seg[1], seg[2], seg[3]
            if has and count > 0:
                covered.add((rel, line))
    gap = orc = 0
    for mut in diff:
        lc = loc(mut)
        if lc is None: continue
        (covered_ := lc in covered)
        if lc in covered:
            orc += 1; oracle[p] += 1
        else:
            gap += 1; cov_gap[p] += 1
    n_cov += gap; n_orc += orc
    per_proj[p] = (gap, orc)
    print(f"  {p:32} coverage_gap={gap:3}  oracle_weak={orc:3}")

tot = n_cov + n_orc + n_err
print(f"\n### {MODEL} llvm-cov decomposition (differential = {tot} mutants)")
print(f"  COVERAGE gap (mutated line NOT executed by src+test tests): {n_cov}  ({n_cov/tot*100:.0f}%)")
print(f"  ORACLE weak  (line executed, mutant still survived):        {n_orc}  ({n_orc/tot*100:.0f}%)")
print(f"  llvm-cov errors (projects skipped):                         {n_err}")
json.dump({"model": MODEL, "coverage_gap": n_cov, "oracle_weak": n_orc, "errors": n_err,
           "per_project": per_proj},
          open(f"{M}/llvmcov_decomposition_{MODEL}.json", "w"), indent=2)
print(f"wrote {M}/llvmcov_decomposition_{MODEL}.json")
