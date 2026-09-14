"""Per-benchmark report: why each bad port is bad (ground-truth failure) versus what the
mutation tests did to it.  Reads results/gt_diag/<bench>/*.json, the top-up arm CSV and
mutants.json; writes results/gt_diag/report_<bench>.md and a machine table miss_table.csv.
"""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/nas/CRUST-bench-COLM/CRUST-bench")
RB = ROOT / "relational_mutation_testing_baseline"
ARM = ROOT / "results/arm_comparison_topup.csv"
DIAG = RB / "results/gt_diag"
RBENCH = ROOT / "datasets/RBench"
REJECT = {"test_failed", "timeout", "crash", "runtime_error", "failed"}


def arm_rows(bench):
    per = defaultdict(dict)
    with ARM.open() as f:
        for r in csv.DictReader(f):
            if r["benchmark"] == bench and r["arm"] == "mutation":
                per[(r["verdict"], r["implementation"])][r["test"]] = r["outcome"]
    return per


def excerpt(at, n=5):
    m = re.match(r"(.+?):(\d+):\d+", at)
    if not m:
        return ""
    rel, line = m.group(1), int(m.group(2))
    p = None
    for cand in (RBENCH / bench / rel, RBENCH / bench / "src/bin" / Path(rel).name):
        if cand.exists():
            p = cand; break
    if p is None:
        return ""
    lines = p.read_text(errors="ignore").splitlines()
    lo, hi = max(0, line - 1 - n), min(len(lines), line + n)
    return "\n".join(f"{i+1:4d}| {lines[i]}" for i in range(lo, hi))


def code(o):
    return {"green": ".", "test_failed": "F", "build_error": "B", "timeout": "T"}.get(o, "?")


rows_out = []
for bench in sys.argv[1:]:
    muts = json.load(open(RB / "results/full100_topup" / bench / "mutants.json"))["mutants"]
    per = arm_rows(bench)
    diag = {}
    for f in (DIAG / bench).glob("*.json"):
        d = json.loads(f.read_text()); diag[(d["verdict"], d["impl"])] = d
    md = [f"# {bench}\n", "## Mutation tests (top-up)\n"]
    for m in muts:
        md.append(f"- **{m['id']}** `{m['function']}` [{m['fault_class']}] — "
                  f"{m.get('observable_effect','')[:160]}")
    md.append("\n## Ground truth on good ports (sanity)\n")
    for (v, impl), d in sorted(diag.items()):
        if v == "good":
            md.append(f"- {impl}: {d['status']} p={d['passed']} f={d['failed']}")
    md.append("\n## Bad ports\n")
    md.append("Mutation outcomes: `.` green, `F` test_failed, `B` build_error, `T` timeout, in m01..m10 order.\n")
    seen_excerpts = {}
    for (v, impl), d in sorted(diag.items()):
        if v != "good" and v != "bad":
            continue
        if v == "good":
            continue
        outs = per.get(("bad", impl), {})
        ids = [m["id"] for m in muts]
        pat = "".join(code(outs.get(i, "?")) for i in ids)
        rejected = any(outs.get(i) in REJECT for i in ids)
        n_b = sum(outs.get(i) == "build_error" for i in ids)
        fails = ", ".join(d["failed_tests"]) or "-"
        first = d["panics"][0] if d["panics"] else None
        why = (f"panic in `{first['test']}` at {first['at']}: {first['msg']}" if first
               else ("; ".join(d["build_errors"][:2]) if d["build_errors"] else d["status"]))
        md.append(f"### {impl}\n- GT: **{d['status']}** ({d['passed']} pass / {d['failed']} fail); failing: {fails}")
        md.append(f"- why: {why}")
        md.append(f"- mutation: `{pat}` → {'REJECTED' if rejected else 'MISSED'}"
                  + (f" ({n_b} build errors)" if n_b else ""))
        if first and first["at"] not in seen_excerpts:
            seen_excerpts[first["at"]] = excerpt(first["at"])
        rows_out.append({"benchmark": bench, "impl": impl, "gt_status": d["status"],
                         "gt_failed_tests": fails, "gt_first_panic": first["msg"] if first else "",
                         "gt_first_at": first["at"] if first else "",
                         "build_errors": " | ".join(d["build_errors"][:2]),
                         "mutation_pattern": pat, "mutation_rejected": rejected,
                         "mutation_build_errors": n_b})
    md.append("\n## Ground-truth code at the panic sites\n")
    for at, ex in seen_excerpts.items():
        md.append(f"**{at}**\n```rust\n{ex}\n```")
    (DIAG / f"report_{bench}.md").write_text("\n".join(md) + "\n")
    c = Counter(r["mutation_rejected"] for r in rows_out if r["benchmark"] == bench)
    print(f"{bench}: bad ports={sum(c.values())} rejected={c[True]} missed={c[False]}")

with (DIAG / "miss_table.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
