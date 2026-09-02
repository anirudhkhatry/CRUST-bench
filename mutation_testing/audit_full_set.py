"""Audit results_opus5_4settings for whether the transpilation-operator run can use it.

The transpile harness needs three things per project, and this checks each one rather
than assuming it:

1. a frozen translation with ``src/lib.rs`` -- that is what ``core`` is built from and
   the only crate that gets mutated;
2. at least one setting with test files in ``<setting>/<proj>/src/bin/*.rs`` -- these
   become the ``tests-<setting>`` crates;
3. a prior per-setting ``metadata/mutation.json`` with ``baseline_valid`` true, which is
   the record that the suite was green when it was generated. A red baseline means the
   suite cannot distinguish a mutant from the original and the project has to be dropped.

Prints one row per project and a usable/unusable split at the end.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/claude/results_opus5_4settings"
SETTINGS = [("src", "src"), ("src+test", "src+test"),
            ("src/test", "src_over_test"), ("c+rust", "c_and_rust")]


def probe(project):
    fz = ROOT / "_frozen_source" / project
    row = {"project": project,
           "has_lib": (fz / "src" / "lib.rs").is_file(),
           "n_src_rs": len(list((fz / "src").rglob("*.rs"))) if (fz / "src").is_dir() else 0,
           "settings": {}}
    for setting, sdir in SETTINGS:
        p = ROOT / sdir / project
        binp = p / "src" / "bin"
        md = p / "metadata" / "mutation.json"
        meta = {}
        if md.is_file():
            try:
                meta = json.loads(md.read_text())
            except json.JSONDecodeError:
                meta = {"error": "unparseable"}
        row["settings"][setting] = {
            "present": p.is_dir(),
            "n_tests": len(list(binp.glob("*.rs"))) if binp.is_dir() else 0,
            "baseline_valid": meta.get("baseline_valid"),
            "score": meta.get("mutation_score"),
            "viable": meta.get("viable_mutants"),
            "kept": meta.get("tests_kept"),
            "total": meta.get("tests_total"),
        }
    return row


def main():
    projects = sorted(p.name for p in (ROOT / "_frozen_source").iterdir() if p.is_dir())
    rows = [probe(p) for p in projects]

    print(f"{'project':22} {'lib':>4} {'.rs':>4}  " +
          "  ".join(f"{s:>12}" for s, _ in SETTINGS))
    for r in rows:
        cells = []
        for s, _ in SETTINGS:
            d = r["settings"][s]
            if not d["present"]:
                cells.append(f"{'ABSENT':>12}")
            elif d["n_tests"] == 0:
                cells.append(f"{'no-tests':>12}")
            else:
                bv = {True: "", False: "!RED", None: "?"}[d["baseline_valid"]]
                cells.append(f"{d['n_tests']:>3}t{bv:>4}{'':>5}")
        print(f"{r['project'][:22]:22} {'y' if r['has_lib'] else 'NO':>4} "
              f"{r['n_src_rs']:>4}  " + "  ".join(cells))

    # ---- the usable split ----
    def usable_settings(r):
        return [s for s, _ in SETTINGS
                if r["settings"][s]["n_tests"] > 0
                and r["settings"][s]["baseline_valid"] is not False]

    all4 = [r["project"] for r in rows if r["has_lib"] and len(usable_settings(r)) == 4]
    some = [r["project"] for r in rows
            if r["has_lib"] and 0 < len(usable_settings(r)) < 4]
    none = [r["project"] for r in rows
            if not r["has_lib"] or not usable_settings(r)]

    print()
    print(f"projects in _frozen_source : {len(rows)}")
    print(f"usable in ALL 4 settings   : {len(all4)}")
    print(f"usable in 1-3 settings     : {len(some)}  {some}")
    print(f"unusable                   : {len(none)}  {none}")
    print()
    print("ALL4 =", json.dumps(all4))
    if "--write" in sys.argv:
        out = Path(__file__).resolve().parent / "full_set_projects.json"
        out.write_text(json.dumps({"all4": all4, "partial": some, "unusable": none},
                                  indent=2))
        print("wrote", out)


if __name__ == "__main__":
    main()
