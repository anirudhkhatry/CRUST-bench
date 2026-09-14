"""Choose the project set to run on.

Eligibility has two hard filters, both measured in Phase 0:
  - the reference Rust implementation passes the RBench ground-truth suite, and
  - the C project is drivable by our harness (compiles, and links a generated test main).
Plus one soft filter: at least a few interface functions, since a mutant is only useful
if a test can reach it through the Rust API.

Among the eligible we want spread rather than the easiest N, so the sample is stratified
over C size, with a header-only and a parser-heavy project forced in.
"""
import csv
import json
import re
from pathlib import Path

from . import cbench, rbench

OUT = Path(__file__).resolve().parent.parent / "results/pilot.json"
# The full panel is frozen separately so selecting it does not clobber the pilot's set.
PANEL = Path(__file__).resolve().parent.parent / "results/panel.json"
PARSER_HINT = re.compile(r"pars|json|lex|token|toml|yaml|xml|regex|csv|ini", re.I)
MIN_IFACE_FNS = 5
# Enough mutable code to host ~10 mutants in distinct functions.
MIN_C_LOC = 60


def eligible(gt_green_only=True):
    """Projects the harness can drive.

    `gt_green_only` is the hard filter that makes M assumable-correct. Dropping it keeps
    the six projects whose reference Rust already fails its own ground-truth suite; they
    are runnable, but a failure on them cannot be charged to the pipeline, so the report
    keeps them in a separate stratum rather than mixing them into the headline number.
    """
    with open(cbench.ROOT / "relational_mutation_testing_baseline"
              / "results/phase0/eligible.csv") as f:
        rows = list(csv.DictReader(f))
    green = rbench.gt_green_projects()
    return [r for r in rows if r["c_usable"] == "True"
            and (r["project"] in green or not gt_green_only)]


def profile(row):
    name = row["project"]
    p = cbench.Project(row["cbench"])
    loc = sum(len((p.path / f).read_text(errors="ignore").splitlines())
              for f in p.mutable_files)
    mode, n_files = p.mutation_target, len(p.mutable_files)
    p.close()
    fns = rbench.iface_fns(name)
    return {"project": name, "cbench": row["cbench"],
            "mode": mode,
            "c_files": n_files, "c_loc": loc,
            "n_iface_fns": len(fns), "iface_fns": fns,
            "gt_green": name in rbench.gt_green_projects(),
            "parserish": bool(PARSER_HINT.search(name))}


def select(n=10):
    profs = [p for p in map(profile, eligible())
             if p["n_iface_fns"] >= MIN_IFACE_FNS and p["c_loc"] >= MIN_C_LOC]
    profs.sort(key=lambda p: p["c_loc"])

    picked, seen = [], set()

    def take(p):
        if p and p["project"] not in seen and len(picked) < n:
            seen.add(p["project"])
            picked.append(p)

    take(next((p for p in profs if p["mode"] == "headers"), None))
    take(max((p for p in profs if p["parserish"]), key=lambda p: p["c_loc"], default=None))

    rest = [p for p in profs if p["project"] not in seen]
    k = n - len(picked)
    for i in range(k):
        take(rest[round(i * (len(rest) - 1) / max(k - 1, 1))])
    for p in rest:  # backfill if strata collided
        take(p)

    picked.sort(key=lambda p: p["c_loc"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_eligible": len(profs), "pilot": picked}, indent=2))
    return profs, picked


def select_all(out=None):
    """The whole panel: every harness-drivable project, soft filters off.

    The stratified pick exists to make a 10-project sample representative; at full scale
    there is nothing to sample, so the size filters go too -- a project with three
    interface functions simply yields fewer distinct mutation sites, which is a result
    rather than a reason to exclude it.
    """
    profs = list(map(profile, eligible(gt_green_only=False)))
    profs.sort(key=lambda p: (not p["gt_green"], p["c_loc"]))
    out = out or PANEL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_eligible": len(profs), "pilot": profs}, indent=2))
    return profs, profs


def load(path=None):
    return json.loads((path or OUT).read_text())["pilot"]
