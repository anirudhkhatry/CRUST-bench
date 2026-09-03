"""Turn a run's per-project JSON into the funnel table and the failure taxonomy."""
import collections
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

FUNNEL = [
    ("M1", "mutants proposed", lambda p: len(p["stage1_verdicts"])),
    ("M2", "valid mutants (apply, compile, differ)", lambda p: len(p["mutants"])),
    ("M3", "killing C test found", lambda p: sum(
        1 for m in p["mutants"] if not m["outcome"].startswith("F1"))),
    ("M4", "transpiled RT compiles", lambda p: sum(
        1 for m in p["mutants"] if m["outcome"] not in ("F1", "F2_transpile_no_compile")
        and not m["outcome"].startswith(("F1", "F2")))),
    ("M5", "RT sound on correct Rust (passes M)", lambda p: sum(
        1 for m in p["mutants"] if m["outcome"] in ("KILLED", "F4_not_discriminating"))),
    ("M6", "RT kills the Rust mutant M'", lambda p: sum(
        1 for m in p["mutants"] if m["outcome"] == "KILLED")),
]

LABEL = {
    "KILLED": "the C-derived test discriminates M from M' -- full success",
    "F1": "no killing C test: the LLM could not distinguish P from P'",
    "F2_transpile_no_compile": "RT never compiled against the Rust interfaces",
    "F3_unsound_on_correct_rust": "RT fails correct Rust -- transpilation changed meaning",
    "F4_not_discriminating": "RT passes M and M' -- discriminating power lost",
    "F5_no_rust_counterpart": "the fault has no Rust counterpart",
}


def load(run_name):
    root = RESULTS / run_name
    return [json.loads(f.read_text()) for f in sorted(root.glob("*/mutants.json"))]


def summarise(run_name):
    projects = load(run_name)
    if not projects:
        return f"no results under {RESULTS / run_name}"
    out = [f"# Run `{run_name}`\n",
           f"{len(projects)} projects, "
           f"{sum(len(p['mutants']) for p in projects)} valid mutants\n"]

    out.append("## Funnel\n")
    out.append("| | stage | n | % of valid |")
    out.append("|---|---|---:|---:|")
    valid = sum(len(p["mutants"]) for p in projects) or 1
    for key, desc, fn in FUNNEL:
        n = sum(fn(p) for p in projects)
        out.append(f"| {key} | {desc} | {n} | {100 * n / valid:.0f}% |")

    out.append("\n## Outcomes\n")
    counts = collections.Counter(m["outcome"] for p in projects for m in p["mutants"])
    out.append("| outcome | n | meaning |")
    out.append("|---|---:|---|")
    for k, n in counts.most_common():
        meaning = LABEL.get(k) or LABEL.get(k.split("_")[0], "")
        out.append(f"| `{k}` | {n} | {meaning} |")

    out.append("\n## Per project\n")
    out.append("| project | loc | iface | valid | killed | kill rate |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for p in sorted(projects, key=lambda p: p["profile"]["c_loc"]):
        n = len(p["mutants"])
        k = sum(1 for m in p["mutants"] if m["outcome"] == "KILLED")
        out.append(f"| {p['project']} | {p['profile']['c_loc']} | "
                   f"{p['profile']['n_iface_fns']} | {n} | {k} | "
                   f"{100 * k / n:.0f}% |" if n else
                   f"| {p['project']} | - | - | 0 | 0 | - |")

    out.append("\n## Fault class vs outcome\n")
    grid = collections.Counter((m.get("fault_class", "?"),
                                m["outcome"].split("_")[0])
                               for p in projects for m in p["mutants"])
    cols = sorted({c for _, c in grid})
    out.append("| fault class | " + " | ".join(cols) + " |")
    out.append("|---" * (len(cols) + 1) + "|")
    for fc in sorted({f for f, _ in grid}):
        out.append(f"| {fc} | " + " | ".join(str(grid[(fc, c)]) for c in cols) + " |")

    # The ground-truth control: does the RBench suite catch M' when RT does not?
    gt = collections.Counter()
    for p in projects:
        for m in p["mutants"]:
            s4 = m.get("stage4") or {}
            if "gt_kills_mp" in s4:
                gt[(m["outcome"] == "KILLED", s4["gt_kills_mp"])] += 1
    if gt:
        out.append("\n## RT vs the RBench ground-truth suite on M'\n")
        out.append("| | GT kills M' | GT misses M' |")
        out.append("|---|---:|---:|")
        out.append(f"| RT kills M' | {gt[(True, True)]} | {gt[(True, False)]} |")
        out.append(f"| RT misses M' | {gt[(False, True)]} | {gt[(False, False)]} |")

    return "\n".join(out) + "\n"
