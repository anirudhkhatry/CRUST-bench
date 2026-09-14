"""The naive baseline -- Rust tests written straight from the C source.

The pipeline's claim is that the *mutant* is what makes its tests strong: a test survives
Stage 2 only because it was executed and shown to separate P from P'. The obvious cheaper
thing to try is to skip all of that -- hand a model the C library and the Rust interfaces,
ask for Rust tests, and see how many of the same faults they catch. That is this arm.

It is scored against the very Rust mutants M' the funnel already built, replayed from the
edits recorded in each mutant's `verdicts.json`, so the naive suite, the pipeline's RT and
the RBench ground-truth suite are all measured on one denominator.

Two design points, both about making the comparison fair rather than flattering:

- **One suite per project**, not one test per mutant. The naive arm never sees a mutant,
  so it has nothing to write a per-mutant test *about*; a project-level suite is also what
  GT is, which makes the three-way table read straight across. It is allowed the same
  compile repair Stage 3 gives RT, and repair may not weaken an assertion.
- **Only the cases that pass M count.** A test that already fails the assumed-correct
  implementation cannot be evidence against a mutant of it -- it would "kill" every M'
  including the ones nothing else can tell apart. Pruning to the green subset is the same
  allowance the RT arm gets implicitly, since RT reaches the M' comparison only where it
  passed M (Stage 4 charges the rest as F3, and they are outside the shared denominator).
"""
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import cbench, llm, rbench as R
from . import stage1_mutate as s1
from . import stage4_relate as s4
from .llm import ask, fenced
from .prompts import load as load_prompt

RESULTS = Path(__file__).resolve().parent.parent / "results"

SYSTEM = ("You are a precise Rust programmer writing tests. You assert exact values, you "
          "use only APIs you have been shown, and you never write a test that could fail "
          "a correct implementation.")

TEST_FN_RE = re.compile(r"#\[test\]")


def generate(name, sources, retries=3, tag="n"):
    """Write the suite and repair it until it builds. Returns the record + the M run.

    `R.compile_check` does not type-check, it *runs* the suite against the reference, so
    the successful attempt already carries the on-M result we need to prune with -- no
    extra cargo invocation.
    """
    mods = R.lib_modules(name) or ["<module>"]
    base = load_prompt("naive_testgen", sources=sources, iface=R.iface_source(name),
                       crate=R.crate_name(name), modules=", ".join(mods), module0=mods[0])

    attempts, user, test_rs = [], base, None
    for i in range(retries + 1):
        # 32k: the suite is a dozen-plus tests written from a whole C library, several
        # times the length of a Stage 3 transpilation of a single test. On a large C
        # library adaptive thinking can still spend the entire budget and return no text
        # at all -- the same failure Stage 1 and Stage 4b hit -- so a retry gets double.
        # Only the retries are widened: the first call's budget is part of the cache key,
        # and raising it would re-bill every project that already succeeded.
        reply = ask(SYSTEM, user, effort="high",
                    max_tokens=32000 if i == 0 else 64000, tag=f"nb_{tag}_{i}")
        test_rs = fenced(reply, "rust")
        if not test_rs:
            attempts.append({"attempt": i, "outcome": "no_code"})
            user = base + (
                "\n\nYour reply ran out of room before it reached the code block. Return "
                "the test file and nothing else -- no reasoning, no commentary -- and if "
                "the suite would be long, write the 12 highest-value tests rather than "
                "25.")
            continue

        res = R.compile_check(name, test_rs, tag=f"nbc_{tag}_{i}")
        attempts.append({"attempt": i, "outcome": res["status"],
                         "passed": res["passed"], "failed": res["failed"]})
        if res["status"] != "build_error":
            return {"test_rs": test_rs, "compiles": True, "attempts": attempts,
                    "n_tests": len(TEST_FN_RE.findall(test_rs))}, res

        errs = "\n".join(l for l in res["out"].splitlines()
                         if l.startswith(("error", "  -->")))[-4000:]
        user = base + "\n\n---\n\n" + load_prompt(
            "naive_repair", errors=errs or res["out"][-3000:], prev=test_rs)

    return {"test_rs": test_rs, "compiles": False, "attempts": attempts,
            "n_tests": len(TEST_FN_RE.findall(test_rs or ""))}, None


def kill_verdict(on_mp, sound):
    """Did the suite detect M'? Returns (killed|None, detail).

    `None` means the trial could not be scored at all, which is not the same as a miss and
    is kept out of every rate.
    """
    cases = on_mp.get("cases") or {}
    if on_mp["status"] == "timeout":
        return True, "timeout"
    if not cases:
        # No per-test line at all: the suite never ran against M'.
        return None, on_mp["status"]
    failed = sorted(t for t in sound if cases.get(t) == "FAILED")
    if failed:
        return True, "failed: " + ", ".join(failed[:6])
    # The suite ran but stopped early -- an abort (a Rust panic is caught, a SIGSEGV or a
    # stack overflow in the mutated code is not) takes the process down mid-run, which is
    # a detection, not a miss.
    missing = sorted(t for t in sound if t not in cases)
    if missing and on_mp.get("rc"):
        return True, "aborted before: " + ", ".join(missing[:6])
    return False, "survived"


def _one_mutant(impl_dir, test_rs, sound, mutant, tag):
    """Replay one recorded M' and run the naive suite against it."""
    rec = {"id": mutant["id"], "outcome": mutant["outcome"],
           "fault_class": mutant.get("fault_class"),
           "mutant_kind": mutant.get("mutant_kind"),
           "rt_kills_mp": mutant["outcome"] == "KILLED",
           "gt_kills_mp": (mutant.get("stage4") or {}).get("gt_kills_mp")}
    try:
        patch = s4.apply_patch(impl_dir, (mutant["stage4"]["rust_mutant"]))
    except (ValueError, KeyError) as e:
        # The stored edits are truncated to 600 chars each, so a handful of large ones
        # can no longer be applied verbatim. Unscorable, not a miss.
        return {**rec, "naive_kills_mp": None, "detail": f"patch_unavailable: {e}"}
    on_mp = R.run_test(impl_dir, test_rs, patch=patch, tag=tag)
    killed, detail = kill_verdict(on_mp, sound)
    return {**rec, "naive_kills_mp": killed, "detail": detail,
            "on_mp": on_mp["status"]}


def score_project(prof, outdir, retries=3, workers=4, log=print, redo=False):
    """Generate the naive suite for one project and score it on every recorded M'."""
    name = prof["project"]
    out = outdir / "naive.json"
    if out.exists() and not redo:
        rec = json.loads(out.read_text())
        # `ok` is a result; the rest are a suite we failed to obtain, and a rerun after a
        # prompt or budget change should retry exactly those. The LLM cache still serves
        # every call whose prompt did not change, so retrying is nearly free.
        if rec["status"] == "ok":
            log(f"{name}: {rec['status']} (cached)")
            return rec

    t0 = time.time()
    payload = {"project": name, "profile": prof}
    impl_dir = R.reference(name)
    funnel = outdir / "mutants.json"
    if impl_dir is None or not funnel.exists():
        payload |= {"status": "no_reference" if impl_dir is None else "no_funnel_run",
                    "mutants": []}
        out.write_text(json.dumps(payload, indent=2))
        log(f"{name}: {payload['status']}")
        return payload

    # The C the model gets to read: the same mutable library sources Stage 1 mutates, so
    # both arms see the same program. It never sees a mutant, a diff, or a C test.
    with cbench.Project(prof["cbench"]) as project:
        sources = s1.source_bundle(project)
    gen, on_m = generate(name, sources, retries=retries, tag=name)
    payload["suite"] = {k: v for k, v in gen.items() if k != "test_rs"}
    (outdir / "naive_test.rs").write_text(gen["test_rs"] or "")
    if not gen["compiles"]:
        payload |= {"status": "no_compile", "mutants": []}
        out.write_text(json.dumps(payload, indent=2))
        log(f"{name}: suite does not compile")
        return payload

    sound = sorted(t for t, s in on_m["cases"].items() if s == "ok")
    payload["on_m"] = {"status": on_m["status"], "n_cases": len(on_m["cases"]),
                       "n_sound": len(sound), "sound": sound,
                       "unsound": sorted(t for t, s in on_m["cases"].items()
                                         if s == "FAILED")}
    if not sound:
        payload |= {"status": "no_sound_tests", "mutants": []}
        out.write_text(json.dumps(payload, indent=2))
        log(f"{name}: no test passes M ({len(on_m['cases'])} cases)")
        return payload

    # Every mutant the funnel built a Rust counterpart for and scored RT against. Any
    # other mutant has no M' to run, so no arm can be scored on it.
    mutants = [m for m in json.loads(funnel.read_text())["mutants"]
               if m["outcome"] in ("KILLED", "F4_not_discriminating")
               and ((m.get("stage4") or {}).get("rust_mutant") or {}).get("applied")]

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_mutant, impl_dir, gen["test_rs"], sound, m,
                          f"nb_{name}_{m['id']}"): m for m in mutants}
        for f in as_completed(futs):
            mid = futs[f]["id"]
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001 - one mutant must not sink a project
                log(f"{name}/{mid}: ERROR {type(e).__name__}: {e}")
                results.append({"id": mid, "naive_kills_mp": None,
                                "detail": f"exception: {type(e).__name__}",
                                "error": traceback.format_exc()[-1500:]})

    results.sort(key=lambda r: str(r["id"]))
    n = sum(1 for r in results if r["naive_kills_mp"] is not None)
    k = sum(1 for r in results if r["naive_kills_mp"])
    payload |= {"status": "ok", "mutants": results,
                "seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(payload, indent=2))
    log(f"{name}: {len(sound)}/{len(on_m['cases'])} sound tests, "
        f"naive kills {k}/{n}" + (f" ({100 * k / n:.0f}%)" if n else ""))
    return payload


def run(run_name, projects, retries=3, project_workers=4, mutant_workers=4, redo=False):
    root = RESULTS / run_name
    if not root.is_dir():
        raise SystemExit(f"no funnel run at {root} -- the baseline scores its mutants")
    todo = [p for p in projects if (root / p["project"] / "mutants.json").exists()]
    print(f"naive baseline: run={run_name} projects={len(todo)} "
          f"(+{len(projects) - len(todo)} with no funnel result) "
          f"workers={project_workers}x{mutant_workers}", flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(prof):
        outdir = root / prof["project"]
        try:
            return score_project(prof, outdir, retries, mutant_workers, log, redo)
        except Exception:  # noqa: BLE001 - record and keep going
            (outdir / "NAIVE_ERROR.txt").write_text(traceback.format_exc())
            log(f"{prof['project']}: PROJECT ERROR (see NAIVE_ERROR.txt)")
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=project_workers) as ex:
        list(ex.map(one, todo))

    cost = llm.cost_ledger()
    cost["seconds"] = round(time.time() - t0)
    (root / "cost_naive.json").write_text(json.dumps(cost, indent=2))
    print(f"done in {cost['seconds']}s; {cost['total']['calls']} LLM calls, "
          f"${cost['total']['paid_usd']:.2f} paid", flush=True)
