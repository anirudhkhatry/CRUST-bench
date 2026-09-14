"""Orchestrate Stage 5 over a finished run: widen every KILLED mutant's test in place.

Runs on a copy of a finished run directory, so the narrow-only tests stay available for
comparison. Per mutant it writes `probe.c`, `probe.tsv`, `wide.rs`, `wide.json`, keeps
the original test as `test_narrow.rs`, and rewrites `test.rs` as narrow + `mod rmt_wide`
when widening succeeds. A failure at any step leaves `test.rs` untouched.
"""
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import cbench, llm, rbench
from . import stage1_mutate as s1
from . import stage5_widen as s5
from .driver import RESULTS, _alt_references

FINAL = {"merged", "probe_failed", "probe_refused", "twin_no_compile", "twin_refused", "too_many_mismatches",
         "did_not_converge", "reference_not_green", "twin_panics_on_reference",
         "merged_not_green"}


def _one(project, name, mdir, ctx, retries, log, redo):
    rec = json.loads((mdir / "verdicts.json").read_text())
    mid = str(rec["id"])
    saved = mdir / "wide.json"
    if saved.exists() and not redo:
        w = json.loads(saved.read_text())
        if w.get("status") in FINAL:
            log(f"{name}/{mid}: wide {w['status']} (cached)")
            return w
    if rec.get("outcome") != "KILLED" or not (mdir / "test.rs").exists():
        return None
    narrow_path = mdir / "test_narrow.rs"
    if not narrow_path.exists():
        narrow_path.write_text((mdir / "test.rs").read_text())
    narrow = narrow_path.read_text()
    w = {"id": mid}

    kill_test = (mdir / "test.c").read_text() if (mdir / "test.c").exists() else ""
    pr = s5.probe(project, rec, ctx["iface_names"], ctx["sources"], kill_test=kill_test,
                  retries=retries, tag=f"{name}_{mid}")
    w["probe"] = {"attempts": pr["attempts"], "n_obs": len(pr["obs"])}
    if not pr["probe_c"]:
        w["status"] = ("probe_refused" if pr["attempts"] and pr["attempts"][-1]["outcome"] == "refused"
                       else "probe_failed")
    else:
        (mdir / "probe.c").write_text(pr["probe_c"])
        (mdir / "probe.tsv").write_text(
            "".join(f"{l}\t{k}\t{raw}\n" for l, k, raw in pr["obs"]))
        tw = s5.transpile_twin(name, pr["probe_c"], pr["obs"], narrow_rs=narrow,
                               retries=retries, tag=f"{name}_{mid}")
        w["twin"] = {"attempts": tw["attempts"], "compiles": tw["compiles"],
                     "n_omitted": tw["n_omitted"]}
        if not tw["compiles"]:
            w["status"] = ("twin_refused" if tw["attempts"] and tw["attempts"][-1]["outcome"] == "refused"
                           else "twin_no_compile")
            (mdir / "wide_failed.rs").write_text(tw["twin_rs"] or "")
        else:
            st = s5.settle(name, tw["twin_rs"], pr["obs"], alternates=(),
                           tag=f"{name}_{mid}")
            # Up to two rounds of "the twin panicked inside the library" repair.
            w["panic_repairs"] = []
            for k in range(2):
                if st["status"] != "twin_panics_on_reference":
                    break
                rp = s5.repair_panic(name, tw["twin_rs"], pr["obs"], st.get("out", ""),
                                     tag=f"{name}_{mid}_{k}")
                w["panic_repairs"].append({"attempts": rp["attempts"],
                                           "compiles": rp["compiles"]})
                if not rp["compiles"]:
                    break
                tw = rp
                w["twin"]["n_omitted"] = rp["n_omitted"]
                st = s5.settle(name, tw["twin_rs"], pr["obs"], alternates=(),
                               tag=f"{name}_{mid}_p{k}")
            w["settle"] = {k: v for k, v in st.items() if k != "wide_rs"}
            w["status"] = st["status"]
            if st["status"] != "settled":
                (mdir / "wide_failed.rs").write_text(tw["twin_rs"] or "")
            if st["wide_rs"]:
                merged = narrow + st["wide_rs"]
                chk = rbench.run_test(rbench.reference(name), merged, tag=f"{name}_{mid}_mg")
                if chk["status"] == "green":
                    (mdir / "wide.rs").write_text(st["wide_rs"])
                    (mdir / "test.rs").write_text(merged)
                    w["status"] = "merged"
                    w["n_checked"] = len(pr["obs"]) - len(st["skip"])
                else:
                    w["status"] = "merged_not_green"
                    w["merge_out"] = chk["out"][-1500:]
    (mdir / "wide.json").write_text(json.dumps(w, indent=2))
    extra = (f" ({w.get('n_checked')} of {w['probe']['n_obs']} observations kept)"
             if w["status"] == "merged" else "")
    log(f"{name}/{mid}: wide {w['status']}{extra}")
    return w


def widen_project(prof, outdir, retries, workers, log=print, redo=False):
    name = prof["project"]
    t0 = time.time()
    mdirs = sorted(d for d in (outdir / "mutants").iterdir()
                   if (d / "verdicts.json").exists())
    with cbench.Project(prof["cbench"]) as project:
        ctx = {"iface_names": rbench.iface_source(name),
               "sources": s1.source_bundle(project)}

        def one(d):
            try:
                return _one(project, name, d, ctx, retries, log, redo)
            except Exception:  # noqa: BLE001
                (d / "wide_ERROR.txt").write_text(traceback.format_exc())
                log(f"{name}/{d.name}: wide EXCEPTION (see wide_ERROR.txt)")
                return {"id": d.name, "status": "exception"}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = [r for r in ex.map(one, mdirs) if r]
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"project": name, "counts": counts, "seconds": round(time.time() - t0, 1)}


def run(run_name, projects, retries=3, project_workers=4, mutant_workers=4, redo=False):
    root = RESULTS / run_name
    todo = [p for p in projects if (root / p["project"] / "mutants").is_dir()]
    print(f"widen run={run_name} projects={len(todo)} workers={project_workers}x"
          f"{mutant_workers}", flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(prof):
        try:
            return widen_project(prof, root / prof["project"], retries, mutant_workers,
                                 log, redo)
        except Exception:  # noqa: BLE001
            (root / prof["project"] / "wide_ERROR.txt").write_text(traceback.format_exc())
            log(f"{prof['project']}: PROJECT ERROR (see wide_ERROR.txt)")
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=project_workers) as ex:
        for n, r in enumerate(ex.map(one, todo), 1):
            if r:
                print(f"[{n}/{len(todo)}] {r['project']}: {r['counts']} ({r['seconds']}s)",
                      flush=True)
    cost = llm.cost_ledger()
    cost["seconds"] = round(time.time() - t0)
    (root / "cost_widen.json").write_text(json.dumps(cost, indent=2))
    print(f"done in {cost['seconds']}s; {cost['total']['calls']} LLM calls, "
          f"${cost['total']['paid_usd']:.2f} paid (${cost['total']['usd']:.2f} incl. "
          f"cache hits)", flush=True)
