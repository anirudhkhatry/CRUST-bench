"""The arm-by-arm comparison on one denominator: RT, naive suite, RBench GT, rust-native.

Every rate here is over the mutants where the arms could be scored -- the Rust
counterpart M' exists, RT was run against it, and the naive suite's M' replay succeeded.
Anything else is reported as an excluded count rather than folded into a percentage.

The rust-native arm (`baseline_rustnative`) is folded in as a fourth column whenever its
`rustnative.json` files are present, and omitted entirely when they are not -- so a run
that has only ever had the first three arms reports exactly what it used to.
"""
import collections
import json
from pathlib import Path

from . import llm

RESULTS = Path(__file__).resolve().parent.parent / "results"

LOC_BUCKETS = [(0, 150), (150, 500), (500, 1500), (1500, 10 ** 9)]

BASE_ARMS = [("rt_kills_mp", "RT"), ("naive_kills_mp", "naive"), ("gt_kills_mp", "GT")]
RN_ARM = ("rn_kills_mp", "rust-native")


def load(run_name):
    root = RESULTS / run_name
    return [json.loads(f.read_text()) for f in sorted(root.glob("*/naive.json"))]


def load_rustnative(run_name):
    """{(project, mutant id): record} for the rust-native arm, empty if it never ran."""
    out = {}
    for f in sorted((RESULTS / run_name).glob("*/rustnative.json")):
        blob = json.loads(f.read_text())
        for m in blob.get("mutants", []):
            out[(blob["project"], m["id"])] = m
    return out


def _rows(projects, rn=()):
    """One row per scorable mutant, carrying every arm's verdict."""
    out = []
    for p in projects:
        for m in p.get("mutants", []):
            if m.get("naive_kills_mp") is None:
                continue
            row = {**m, "project": p["project"],
                   "c_loc": p["profile"]["c_loc"],
                   "gt_green": p["profile"].get("gt_green", True)}
            hit = (rn or {}).get((p["project"], m["id"]))
            if hit:
                row["rn_kills_mp"] = hit.get("rn_kills_mp")
                row["rn_outcome"] = hit.get("outcome")
            out.append(row)
    return out


def charged_rows(run_name, projects, rn=()):
    """Every mutant RT was scored on, with an unscored naive arm charged as a miss.

    A project whose naive suite never compiled contributes no rows to `_rows`, and those
    are exactly the projects the naive arm did worst on -- so the shared-denominator rate
    is conditional on the naive arm having cleared its own first gate, the same way the
    funnel's 97% is conditional on RT surviving Stages 2 and 3. This is the other end of
    that: the naive arm charged for everything it could not test.
    """
    scored = {(p["project"], m["id"]): m for p in projects for m in p.get("mutants", [])}
    out = []
    for f in sorted((RESULTS / run_name).glob("*/mutants.json")):
        blob = json.loads(f.read_text())
        for m in blob["mutants"]:
            if m["outcome"] not in ("KILLED", "F4_not_discriminating"):
                continue
            if not ((m.get("stage4") or {}).get("rust_mutant") or {}).get("applied"):
                continue
            hit = scored.get((blob["project"], m["id"])) or {}
            rn_hit = (rn or {}).get((blob["project"], m["id"]))
            out.append({"project": blob["project"], "id": m["id"],
                        # Charged like the naive arm: `False` wherever the arm reached
                        # this M' and did not detect it, the unreplayable ones included.
                        # `None` where it never got to the mutant at all -- a project it
                        # skipped, or a `--max-mutants` pilot that stopped short. Charging
                        # those would make a capped pilot look like a total failure.
                        "rn_kills_mp": (bool(rn_hit.get("rn_kills_mp")) if rn_hit
                                        else None),
                        "c_loc": blob["profile"]["c_loc"],
                        "gt_green": blob["profile"].get("gt_green", True),
                        "fault_class": m.get("fault_class"),
                        "mutant_kind": m.get("mutant_kind"),
                        "rt_kills_mp": m["outcome"] == "KILLED",
                        "gt_kills_mp": (m.get("stage4") or {}).get("gt_kills_mp"),
                        "naive_kills_mp": bool(hit.get("naive_kills_mp"))})
    return out


def _rate(rows, key):
    n = sum(1 for r in rows if r.get(key) is not None)
    k = sum(1 for r in rows if r.get(key))
    return k, n, (f"{100 * k / n:.0f}%" if n else "-")


def _arm_table(rows, title, groups, arms=BASE_ARMS):
    """`groups` is [(label, subset)] -- the same arm columns for each subset.

    The project count is carried alongside the mutant count because a stratum's rate is
    only as broad as the projects underneath it: 1500+ LOC is 65 mutants, but reading that
    as 65 independent observations would overstate it badly.
    """
    out = [f"\n## {title}\n",
           "| | projects | mutants | "
           + " | ".join(f"{label} kills M'" for _, label in arms) + " |",
           "|---|---:|---:|" + "---:|" * len(arms)]
    for label, sub in groups:
        if not sub:
            out.append(f"| {label} | 0 | 0 |" + " - |" * len(arms))
            continue
        cells = []
        for key, _ in arms:
            k, n, pct = _rate(sub, key)
            cells.append(f"{k} ({pct})")
        out.append(f"| {label} | {len({r['project'] for r in sub})} | {len(sub)} | "
                   + " | ".join(cells) + " |")
    return out


def _pair(rows, a, b, la, lb):
    ok = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
    g = collections.Counter((bool(r[a]), bool(r[b])) for r in ok)
    return [f"\n### {la} vs {lb} ({len(ok)} mutants)\n",
            f"| | {lb} kills | {lb} misses |",
            "|---|---:|---:|",
            f"| **{la} kills** | {g[(True, True)]} | **{g[(True, False)]}** |",
            f"| **{la} misses** | **{g[(False, True)]}** | {g[(False, False)]} |"]


def summarise(run_name):
    projects = load(run_name)
    if not projects:
        return f"no naive-baseline results under {RESULTS / run_name}"
    rn = load_rustnative(run_name)
    arms = BASE_ARMS + ([RN_ARM] if rn else [])
    rows = _rows(projects, rn)
    scored = [p for p in projects if p["status"] == "ok"]

    out = [f"# Naive baseline vs the pipeline — run `{run_name}`\n",
           "The naive arm gets the **C source and the Rust interfaces** and is asked for a "
           "Rust test suite. It never sees a mutant, a diff, or a C test. Both arms are "
           "scored on the same Rust mutants `M'`, built by the pipeline's Stage 4 and "
           "replayed here from the recorded edits.\n"]
    if rn:
        out.append(
            "The **rust-native** arm is the oracle upper bound: for each of those same "
            "`M'` it is shown the reference implementation and the exact `M → M'` diff, "
            "and asked for one Rust test that passes the first and fails the second. It "
            "never sees C. Read it as a ceiling, not as a technique — it starts from the "
            "fault the other arms have to find.\n")

    out.append("## Suite generation\n")
    bad = collections.Counter(p["status"] for p in projects if p["status"] != "ok")
    n_tests = sum(p.get("suite", {}).get("n_tests", 0) for p in scored)
    n_cases = sum(p.get("on_m", {}).get("n_cases", 0) for p in scored)
    n_sound = sum(p.get("on_m", {}).get("n_sound", 0) for p in scored)
    out += ["| | n |", "|---|---:|",
            f"| projects attempted | {len(projects)} |",
            f"| projects with a usable suite | {len(scored)} |",
            f"| `#[test]` functions written | {n_tests} |",
            f"| test cases that ran against M | {n_cases} |",
            f"| **cases sound on M** (the ones that can score a mutant) | {n_sound} "
            f"({100 * n_sound / n_cases:.0f}%) |" if n_cases else "| cases | 0 |"]
    for k, v in bad.most_common():
        out.append(f"| projects dropped: `{k}` | {v} |")
    if scored:
        out.append(f"\nMedian suite: "
                   f"{sorted(p['on_m']['n_sound'] for p in scored)[len(scored) // 2]} "
                   f"sound tests per project.")

    unscorable = sum(1 for p in projects for m in p.get("mutants", [])
                     if m.get("naive_kills_mp") is None)
    out.append(f"\n## Kill rates on the shared denominator\n")
    out.append(f"{len(rows)} mutants have a Rust counterpart that all three arms could be "
               f"scored against"
               + (f"; {unscorable} more were excluded because their M' could not be "
                  f"replayed." if unscorable else "."))
    charged = charged_rows(run_name, projects, rn)
    n_dark = len(charged) - len(rows)
    out += _arm_table(rows, "Headline", [
        ("shared denominator (naive suite exists)", rows),
        (f"charged the whole panel (+{n_dark} untestable)", charged)], arms)
    if rn:
        k, n, pct = _rate(rows, "rn_kills_mp")
        out.append(f"\nThe rust-native column is over the {n} of those mutants it has been "
                   f"run on so far ({pct}).")
        by_outcome = collections.Counter(m.get("outcome") for m in rn.values())
        if by_outcome:
            unrep = by_outcome.pop("UNREPLAYABLE", 0)
            scorable = sum(by_outcome.values())
            killed = by_outcome.get("KILLED", 0)
            out += [f"\n### Rust-native arm outcomes (all {len(rn)} mutants it ran)\n",
                    "| outcome | n |", "|---|---:|"]
            out += [f"| `{o}` | {c} |" for o, c in by_outcome.most_common()]
            out.append(
                f"\n`F_*` are misses charged to the arm: it had the diff and still "
                f"produced no test that passes `M` and fails `M'`. `UNREACHABLE` is the "
                f"arm's own claim that the fault is not observable through the public "
                f"API — the most interesting bucket, because those are the mutants no "
                f"arm can fairly be charged for. Over everything it could score, the arm "
                f"kills **{killed}/{scorable} ({100 * killed / scorable:.1f}%)**.")
            if unrep:
                out.append(
                    f"\nA further **{unrep}** mutants are excluded, not charged: their "
                    f"recorded Stage 4 edits are truncated at 600 characters, so the "
                    f"replayed `M'` substitutes one prefix for another and no longer "
                    f"compiles. Stage 4's own `M'` did build — this is a limit of what "
                    f"the run wrote down, not of the mutant. The arm build-probes every "
                    f"`M'` before spending a call on it. They are also, with one "
                    f"exception, exactly the mutants the naive arm could not score, so "
                    f"the `charged the whole panel` row above charges both arms for a "
                    f"harness artifact; the shared-denominator row is the honest one.")
    out.append(
        "\nThe second row charges the naive arm for the mutants it could not reach at "
        "all — a project whose suite never compiled, or never produced a case that "
        "passes M, tests nothing. Those are the projects it did worst on, so the first "
        "row is conditional on the naive arm clearing its own gate, exactly as the "
        "funnel's 97% is conditional on RT surviving Stages 2–3. RT and GT are "
        "unaffected: both rows score them on the same mutants.")

    out += _pair(rows, "rt_kills_mp", "naive_kills_mp", "RT", "naive")
    out += _pair(rows, "naive_kills_mp", "gt_kills_mp", "naive", "GT")
    if rn:
        out += _pair(rows, "rn_kills_mp", "rt_kills_mp", "rust-native", "RT")
        out.append(
            "\nThe bottom-left cell is the one to read: mutants RT kills that the "
            "oracle arm does not. A large count there means the ceiling is not a "
            "ceiling — the arm's per-mutant single test is losing to a suite, or its "
            "soundness gate is charging it for tests RT was allowed to keep.")

    out += _arm_table(rows, "By C project size", [
        (f"{lo}–{hi if hi < 10 ** 9 else '∞'} LOC",
         [r for r in rows if lo <= r["c_loc"] < hi]) for lo, hi in LOC_BUCKETS], arms)

    out += _arm_table(rows, "By fault class", [
        (fc, [r for r in rows if r.get("fault_class") == fc])
        for fc in sorted({r.get("fault_class") or "?" for r in rows})], arms)

    out += _arm_table(rows, "By mutant kind", [
        (k, [r for r in rows if r.get("mutant_kind") == k])
        for k in sorted({r.get("mutant_kind") or "?" for r in rows})], arms)

    out += _arm_table(rows, "Reference implementation stratum", [
        ("GT-green (M assumed correct)", [r for r in rows if r["gt_green"]]),
        ("not GT-green (M unverified)", [r for r in rows if not r["gt_green"]])], arms)

    labels = [label for _, label in arms]
    out.append("\n## Per project\n")
    out.append("| project | loc | sound tests | mutants | " + " | ".join(labels) + " |")
    out.append("|---|---:|---:|---:|" + "---:|" * len(arms))
    by_project = collections.defaultdict(list)
    for r in rows:
        by_project[r["project"]].append(r)
    for p in sorted(scored, key=lambda p: p["profile"]["c_loc"]):
        sub = by_project.get(p["project"], [])
        cells = [_rate(sub, k)[2] for k, _ in arms]
        out.append(f"| {p['project']} | {p['profile']['c_loc']} | "
                   f"{p['on_m']['n_sound']} | {len(sub)} | " + " | ".join(cells) + " |")

    out += cost_section(rows)
    if rn:
        out += cost_section([r for r in rows if r.get("rn_kills_mp") is not None],
                            prefix="rn_", title="Cost — rust-native arm",
                            tail="This arm pays per *mutant*, like the funnel, not per "
                                 "project: it writes one test about one diff.")
    return "\n".join(out) + "\n"


def cost_section(rows, prefix="nb_", title="Cost",
                 tail="Compare the funnel's per-mutant cost in `REPORT.md`: the naive "
                      "arm pays for one suite per project, not one test per mutant."):
    """What an arm cost, read off the LLM cache rather than a run's ledger.

    `cost_naive.json` is per *invocation*, and the naive arm took two (the second re-ran
    the projects whose first suite came back empty), so a ledger would report whichever
    ran last. Every `nb_`/`rn_` cache entry is one call that arm ever made, which is the
    total we actually want.
    """
    calls = empty = tin = tout = 0
    waste = 0.0
    for f in llm.CACHE.glob(f"{prefix}*.json"):
        b = json.loads(f.read_text())
        if b.get("payload", {}).get("model") != llm.MODEL:
            continue
        u = b.get("usage") or {}
        i = (u.get("input_tokens", 0) + (u.get("cache_read_input_tokens") or 0)
             + (u.get("cache_creation_input_tokens") or 0))
        o = u.get("output_tokens", 0)
        calls, tin, tout = calls + 1, tin + i, tout + o
        if not b.get("text", "").strip():
            empty += 1
            waste += (i * llm.PRICE_IN + o * llm.PRICE_OUT) / 1e6
    if not calls:
        return []
    usd = (tin * llm.PRICE_IN + tout * llm.PRICE_OUT) / 1e6
    out = [f"\n## {title}\n",
           f"`{llm.MODEL}`: {calls} calls over the whole arm, {tin:,} in / {tout:,} out, "
           f"**${usd:.2f}**."]
    if empty:
        out.append(f"\n${waste:.2f} of that went on {empty} replies that came back empty "
                   f"— adaptive thinking spending the whole budget before emitting any "
                   f"text. Those projects were re-run with a wider retry budget, so the "
                   f"arm's *necessary* cost is nearer ${usd - waste:.2f}"
                   + (f" (${(usd - waste) / len(rows):.3f} per mutant scored)."
                      if rows else "."))
    elif rows:
        out.append(f"\n${usd / len(rows):.3f} per mutant scored.")
    out.append("\n" + tail)
    return out
