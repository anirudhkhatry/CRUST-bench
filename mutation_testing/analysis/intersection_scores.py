"""Score every setting over the SAME mutant set (Phase 1 of docs/plan_shared_mutant_set.md).

Today each setting is a full copy of the project with that setting's tests inside the
mutated crate, so cargo-mutants marks a mutant *unviable* whenever that setting's tests
break compilation. The denominator therefore moves between settings and `killed/viable`
compares scores computed over different mutant populations.

This pass recomputes each project's score over the mutants that are viable in **every**
setting -- the intersection -- and reports the intersection size alongside it, so the
comparison table stops carrying incomparable denominators. Nothing is re-run: it reads
the `outcomes.json` files cargo-mutants already wrote.

Mutants are identified across settings by the name cargo-mutants gives them,
``<file>:<line>:<col>: replace <fn> with <expr>``, which encodes file, position and
replacement and is stable across runs of the same frozen source. Repeats of one name
within a setting are disambiguated by their (deterministic, ``--no-shuffle``) order.

Usage::

    python intersection_scores.py results_gpt55_4settings results_opus5_4settings
    python intersection_scores.py <results_dir> --min-common 10 --fold-timeouts

Reads  <results_dir>/_mutants_out/<setting>/<project>/mutants.out/outcomes.json
Writes <results_dir>/intersection_table.{md,csv}, intersection_long.csv,
       intersection_report.json
"""

import argparse
import csv
import json
from pathlib import Path

# Setting -> filesystem-safe directory name, mirroring run_mutation.SETTING_DIRS.
# Duplicated rather than imported: run_mutation pulls in the whole CRUST-bench pipeline
# (endpoints, tree-sitter, ...) which this read-only pass has no use for.
SETTING_DIRS = {
    "src": "src",
    "src+test": "src+test",
    "src/test": "src_over_test",
    "c+rust": "c_and_rust",
}

# outcomes.json summary -> our status. Anything else (Success/Failure baselines) is dropped.
STATUS = {
    "CaughtMutant": "killed",
    "MissedMutant": "survived",
    "Timeout": "timeout",
    "Unviable": "unviable",
}
# A mutant counts toward a denominator only if the crate compiled with it applied.
VIABLE = {"killed", "survived", "timeout"}


def outcomes_path(results_dir, setting, project):
    """Locate a run's outcomes.json, newest layout first.

    cargo-mutants originally wrote ``mutants.out`` inside the project tree; runs since
    the concurrency fix write it to an external ``_mutants_out/`` (see
    ``mutate_with_cargo(out_base=...)``). Where both exist the external one is the
    re-run and wins.
    """
    results_dir, sdir = Path(results_dir), SETTING_DIRS[setting]
    external = results_dir / "_mutants_out" / sdir / project / "mutants.out" / "outcomes.json"
    if external.exists():
        return external
    return results_dir / sdir / project / "mutants.out" / "outcomes.json"


def load_mutants(path):
    """Read one outcomes.json -> {mutant_key: status}, or None if unreadable.

    ``name`` alone can repeat when two mutants share a file position and replacement,
    so the n-th repeat is keyed ``name#n``. cargo-mutants is run with --no-shuffle, so
    that ordering is the source order and matches across settings.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None

    mutants, seen = {}, {}
    for outcome in data.get("outcomes", []):
        scenario = outcome.get("scenario")
        if not isinstance(scenario, dict) or "Mutant" not in scenario:
            continue  # Baseline
        status = STATUS.get(outcome.get("summary"))
        if status is None:
            continue
        name = scenario["Mutant"].get("name")
        if not name:
            continue
        n = seen.get(name, 0)
        seen[name] = n + 1
        mutants[name if n == 0 else f"{name}#{n}"] = status
    return mutants


def discover_projects(results_dir, settings):
    """Every project name that has a mutants.out under any of ``settings``, either layout."""
    results_dir = Path(results_dir)
    names = set()
    for setting in settings:
        for base in (results_dir / "_mutants_out" / SETTING_DIRS[setting],
                     results_dir / SETTING_DIRS[setting]):
            if base.is_dir():
                names.update(p.name for p in base.iterdir()
                             if (p / "mutants.out" / "outcomes.json").exists())
    return sorted(names)


def score(mutants, keys, fold_timeouts):
    """killed/score over exactly ``keys`` (all of which are viable here)."""
    killed = sum(1 for k in keys
                 if mutants[k] == "killed" or (fold_timeouts and mutants[k] == "timeout"))
    return {
        "killed": killed,
        "n": len(keys),
        "mutation_score": (killed / len(keys)) if keys else None,
    }


def analyze_project(results_dir, project, settings, fold_timeouts):
    """Per-setting raw counts plus the intersection-scoped scores for one project."""
    per_setting, missing, sources = {}, [], {}
    for setting in settings:
        path = outcomes_path(results_dir, setting, project)
        mutants = load_mutants(path)
        if mutants is None:
            missing.append(setting)
        else:
            per_setting[setting] = mutants
            # "rerun" = external _mutants_out (post-concurrency-fix); "stored" = in-tree.
            sources[setting] = "rerun" if "_mutants_out" in path.parts else "stored"

    row = {"project": project, "missing_settings": missing, "sources": sources, "settings": {}}
    for setting, mutants in per_setting.items():
        counts = {s: 0 for s in ("killed", "survived", "timeout", "unviable")}
        for status in mutants.values():
            counts[status] += 1
        viable = counts["killed"] + counts["survived"] + counts["timeout"]
        killed = counts["killed"] + (counts["timeout"] if fold_timeouts else 0)
        row["settings"][setting] = {
            **counts,
            "total": len(mutants),
            "viable": viable,
            # the as-reported score, over this setting's own moving denominator.
            "mutation_score_own": (killed / viable) if viable else None,
        }

    if missing:
        row["n_common"] = 0
        row["note"] = "missing:" + ",".join(missing)
        return row

    # The intersection: present AND viable in every setting.
    common = set.intersection(*(
        {k for k, s in m.items() if s in VIABLE} for m in per_setting.values()
    ))
    union_all = set.union(*(set(m) for m in per_setting.values()))
    row["n_common"] = len(common)
    row["n_union"] = len(union_all)
    # Verification hook (plan: "assert the intersection equals each setting's set where
    # totals already match"): true when no setting lost a mutant to unviability or to a
    # differing mutant list, i.e. the denominators already agreed.
    row["sets_identical"] = all(set(m) == union_all for m in per_setting.values())
    row["denominators_agree"] = all(
        len(common) == d["viable"] for d in row["settings"].values()
    )
    for setting in settings:
        row["settings"][setting]["common"] = score(per_setting[setting], common, fold_timeouts)
    row["note"] = "ok"
    return row


def analyze(results_dir, settings, fold_timeouts, min_common):
    results_dir = Path(results_dir)
    projects = discover_projects(results_dir, settings)
    rows = [analyze_project(results_dir, p, settings, fold_timeouts) for p in projects]

    # Average only over projects whose intersection is big enough to mean anything;
    # a 3-mutant intersection is noise, not a score.
    included = [r for r in rows if r["note"] == "ok" and r["n_common"] >= min_common]
    averages = {}
    for setting in settings:
        scores = [r["settings"][setting]["common"]["mutation_score"] for r in included]
        scores = [s for s in scores if s is not None]
        averages[setting] = {
            "avg_mutation_score_common": (sum(scores) / len(scores)) if scores else None,
            "total_killed_common": sum(r["settings"][setting]["common"]["killed"] for r in included),
            "avg_viable_own": (
                sum(r["settings"][setting]["viable"] for r in included) / len(included)
            ) if included else None,
        }
    total_common = sum(r["n_common"] for r in included)

    return {
        "results_dir": str(results_dir),
        "settings": settings,
        "fold_timeouts": fold_timeouts,
        "min_common": min_common,
        "n_projects": len(rows),
        "n_included": len(included),
        "included_projects": [r["project"] for r in included],
        "excluded_projects": {
            r["project"]: (r["note"] if r["note"] != "ok" else f"n_common={r['n_common']}")
            for r in rows if r not in included
        },
        # How often the pre-existing per-setting denominators already agreed -- the
        # size of the problem this pass fixes.
        "n_denominators_agree": sum(1 for r in rows if r.get("denominators_agree")),
        "n_sets_identical": sum(1 for r in rows if r.get("sets_identical")),
        "total_common_mutants": total_common,
        # Pooled score: killed over all included projects' intersections. Unlike the
        # per-project average this is not dominated by tiny projects.
        "pooled_over_common": {
            s: (averages[s]["total_killed_common"] / total_common) if total_common else None
            for s in settings
        },
        "averages_over_common": averages,
        "per_project": rows,
    }


def _cell(row, setting):
    if row["note"] != "ok":
        return "-"
    c = row["settings"][setting]["common"]
    if c["mutation_score"] is None:
        return "0-common"
    return f"{c['mutation_score']*100:.0f}% ({c['killed']}/{c['n']})"


def write_report(report, results_dir):
    results_dir = Path(results_dir)
    settings = report["settings"]
    included = set(report["included_projects"])
    headers = ["project", "n_common", *settings, "viable_own (" + "/".join(settings) + ")", "note"]

    rows = []
    for r in report["per_project"]:
        viable_own = "/".join(
            str(r["settings"][s]["viable"]) if s in r["settings"] else "-" for s in settings
        )
        note = "included" if r["project"] in included else (
            r["note"] if r["note"] != "ok" else "below-floor")
        rows.append([r["project"], r["n_common"], *(_cell(r, s) for s in settings),
                     viable_own, note])

    avg = [f"AVERAGE (n={report['n_included']})", report["total_common_mutants"]]
    for s in settings:
        a = report["averages_over_common"][s]["avg_mutation_score_common"]
        avg.append("n/a" if a is None else f"{a*100:.1f}%")
    avg.append("")
    avg.append("mean of per-project scores")

    pooled = ["POOLED (killed/common)", report["total_common_mutants"]]
    for s in settings:
        p = report["pooled_over_common"][s]
        k = report["averages_over_common"][s]["total_killed_common"]
        pooled.append("n/a" if p is None else f"{p*100:.1f}% ({k})")
    pooled.append("")
    pooled.append("over all included mutants")

    all_rows = rows + [avg, pooled]
    widths = [max(len(str(r[i])) for r in [headers] + all_rows) for i in range(len(headers))]
    fmt = lambda r: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    sep = "  ".join("-" * w for w in widths)
    print(f"\n=== {results_dir.name}: scores over the mutants viable in ALL settings "
          f"(cell = score (killed/n_common)) ===")
    print(fmt(headers))
    print(sep)
    for r in rows:
        print(fmt(r))
    print(sep)
    print(fmt(avg))
    print(fmt(pooled))

    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in all_rows]
    (results_dir / "intersection_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with open(results_dir / "intersection_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(all_rows)

    # Long form: one row per (project, setting). Keeps the per-setting `viable` numbers
    # -- their divergence is itself a finding (tests that break compilation).
    long_headers = ["project", "setting", "n_common", "killed_common", "score_common",
                    "total", "viable_own", "killed_own", "survived", "timeout", "unviable",
                    "score_own", "included"]
    with open(results_dir / "intersection_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(long_headers)
        for r in report["per_project"]:
            for s in settings:
                d = r["settings"].get(s)
                if d is None:
                    continue
                c = d.get("common", {})
                w.writerow([
                    r["project"], s, r["n_common"], c.get("killed", ""),
                    "" if c.get("mutation_score") is None else round(c["mutation_score"], 4),
                    d["total"], d["viable"], d["killed"], d["survived"], d["timeout"],
                    d["unviable"],
                    "" if d["mutation_score_own"] is None else round(d["mutation_score_own"], 4),
                    r["project"] in included,
                ])

    with open(results_dir / "intersection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{report['n_denominators_agree']}/{report['n_projects']} project(s) already had "
          f"the same viable count in every setting; {report['n_sets_identical']} had identical "
          f"mutant sets.")
    excluded = report["excluded_projects"]
    if excluded:
        print(f"{len(excluded)} project(s) excluded from the averages: "
              + ", ".join(f"{p} ({why})" for p, why in sorted(excluded.items())))
    print(f"Wrote {results_dir/'intersection_table.md'}, "
          f"{results_dir/'intersection_table.csv'}, {results_dir/'intersection_long.csv'}, "
          f"{results_dir/'intersection_report.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Rescore mutation results over the mutants viable in every setting.")
    parser.add_argument("results_dirs", nargs="+",
                        help="one or more <results_*> dirs containing _mutants_out/")
    parser.add_argument("--settings", default=",".join(SETTING_DIRS),
                        help="comma-separated settings to intersect (default: all four)")
    parser.add_argument("--min-common", type=int, default=10,
                        help="exclude projects whose intersection is smaller than this "
                             "from the averaged rows (default: 10)")
    parser.add_argument("--fold-timeouts", action="store_true",
                        help="count Timeout as killed (a timeout IS a detection) instead "
                             "of leaving it in the denominator only")
    args = parser.parse_args()

    settings = [s for s in args.settings.split(",") if s.strip()]
    unknown = [s for s in settings if s not in SETTING_DIRS]
    if unknown:
        parser.error(f"unknown setting(s): {unknown}; known: {list(SETTING_DIRS)}")

    for d in args.results_dirs:
        if not (Path(d) / "_mutants_out").is_dir():
            print(f"{d}: no _mutants_out/, skipping")
            continue
        report = analyze(d, settings, args.fold_timeouts, args.min_common)
        write_report(report, d)


if __name__ == "__main__":
    main()
