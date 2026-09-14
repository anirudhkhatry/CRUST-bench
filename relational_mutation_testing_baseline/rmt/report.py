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


def crosstab(projects, field, title, killrate=False):
    """`field` x outcome-family grid, one row per observed value of `field`."""
    grid = collections.Counter((m.get(field) or "?", m["outcome"].split("_")[0])
                               for p in projects for m in p["mutants"])
    if not grid:
        return []
    cols = sorted({c for _, c in grid})
    head = cols + ["kill rate"] if killrate else cols
    out = [f"\n## {title.capitalize()}\n",
           "| " + field.replace("_", " ") + " | " + " | ".join(head) + " |",
           "|---" * (len(head) + 1) + "|"]
    for v in sorted({f for f, _ in grid}):
        cells = [str(grid[(v, c)]) for c in cols]
        if killrate:
            n = sum(grid[(v, c)] for c in cols)
            cells.append(f"{100 * grid[(v, 'KILLED')] / n:.0f}%" if n else "-")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    return out


STAGE_LABEL = {
    "stage1_mutate": "Stage 1 — propose mutants (1 call/project)",
    "stage2_testgen": "Stage 2 — killing C test (+ repair rounds)",
    "stage3_transpile": "Stage 3 — transpile to Rust (+ repair rounds)",
    "stage4_rust_mutant": "Stage 4b — build the Rust counterpart M'",
}


def cost_table(run_name, n_valid):
    """Per-stage token spend, written by the driver at the end of a run.

    `usd` is what the calls cost the first time they ran; `paid` excludes the ones this
    invocation served from the disk cache, so on a fresh run the two agree and on a
    resumed one the gap is what resumability saved.
    """
    f = RESULTS / run_name / "cost.json"
    if not f.exists():
        return []
    c = json.loads(f.read_text())
    t = c["total"]
    out = [f"\n## Cost by stage\n",
           f"`{c['model']}`, priced at ${c['price_in']}/${c['price_out']} per Mtok "
           f"in/out.\n",
           "| stage | calls | cached | input tok | output tok | usd | paid |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for k in list(STAGE_LABEL) + sorted(set(c["by_stage"]) - set(STAGE_LABEL)):
        s = c["by_stage"].get(k)
        if not s:
            continue
        out.append(f"| {STAGE_LABEL.get(k, k)} | {s['calls']} | {s['cache_hits']} | "
                   f"{s['input']:,} | {s['output']:,} | ${s['usd']:.2f} | "
                   f"${s['paid_usd']:.2f} |")
    out.append(f"| **total** | {t['calls']} | {t['cache_hits']} | {t['input']:,} | "
               f"{t['output']:,} | **${t['usd']:.2f}** | **${t['paid_usd']:.2f}** |")
    if n_valid:
        out.append(f"\n${t['usd'] / n_valid:.2f} per valid mutant carried through the "
                   f"whole funnel ({n_valid} mutants"
                   + (f", {c['seconds']}s wall clock" if c.get("seconds") else "") + ").")
    return out


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

    # The reference M is only assumable-correct where it passed RBench's own suite in
    # Phase 0. On the rest, an F3 may be the reference's fault rather than the
    # transpilation's, so the two strata are never pooled into one kill rate.
    if any(not p["profile"].get("gt_green", True) for p in projects):
        out.append("\n## Reference implementation stratum\n")
        out.append("| stratum | projects | valid | killed | kill rate |")
        out.append("|---|---:|---:|---:|---:|")
        for green, name in ((True, "GT-green (M assumed correct)"),
                            (False, "not GT-green (M unverified)")):
            grp = [p for p in projects if bool(p["profile"].get("gt_green", True)) is green]
            n = sum(len(p["mutants"]) for p in grp)
            k = sum(1 for p in grp for m in p["mutants"] if m["outcome"] == "KILLED")
            out.append(f"| {name} | {len(grp)} | {n} | {k} | "
                       f"{100 * k / n:.0f}% |" if n else f"| {name} | {len(grp)} | 0 | 0 | - |")

    out.append("\n## Per project\n")
    out.append("| project | gt | loc | iface | valid | killed | kill rate |")
    out.append("|---|:-:|---:|---:|---:|---:|---:|")
    for p in sorted(projects, key=lambda p: (not p["profile"].get("gt_green", True),
                                             p["profile"]["c_loc"])):
        n = len(p["mutants"])
        k = sum(1 for m in p["mutants"] if m["outcome"] == "KILLED")
        g = "yes" if p["profile"].get("gt_green", True) else "**no**"
        out.append(f"| {p['project']} | {g} | {p['profile']['c_loc']} | "
                   f"{p['profile']['n_iface_fns']} | {n} | {k} | "
                   f"{100 * k / n:.0f}% |" if n else
                   f"| {p['project']} | {g} | - | - | 0 | 0 | - |")

    # Semantic mutants are the object of study; the operator control group says whether a
    # low kill rate is about C->Rust translation or merely about harder faults.
    out += crosstab(projects, "mutant_kind", "mutant kind vs outcome", killrate=True)
    out += crosstab(projects, "fault_class", "fault class vs outcome")

    out += cost_table(run_name, valid)

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
