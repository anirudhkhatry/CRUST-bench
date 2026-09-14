"""The Rust-native baseline -- a test written from the Rust mutant itself.

The other two arms both pay for the C detour. RT is a C test transpiled into Rust; the
naive arm reads the C source and guesses where a re-implementer would go wrong. This arm
deletes C from the loop entirely: it is shown the reference implementation `M`, the exact
diff that turns it into `M'`, and nothing else, and is asked for a Rust test that passes
the first and fails the second.

That makes it the **oracle upper bound** on the study's question. It knows the fault
exactly, in the language the test is written in, with no translation step to lose it in --
so whatever it cannot kill is a mutant that is hard or impossible to kill through the
public API, not one that the pipeline mislaid. `1 - rn_kill_rate` is therefore the floor
under every other arm's miss rate, and RT's shortfall against it is what the C detour
costs.

Apples to apples with the other arms, by construction:

- **The same `M'`.** Every mutant is the one Stage 4 already built, replayed here from the
  edits recorded in `mutants.json` -- byte-identical to the implementation RT, the naive
  suite and the RBench GT suite were scored against. Nothing new is mutated.
- **The same denominator.** Only mutants whose `M'` applies verbatim are scorable, which
  is the same set the naive arm can replay, so the four-way table has one row count.
- **The same gates in the same order.** Compile repair (build errors only), then the
  soundness gate on `M`, then the kill check on `M'`. RT reaches the `M'` comparison only
  after passing `M`; so does this.

The one asymmetry is deliberate and is the whole point: this arm sees the mutant, and sees
it in Rust. It also sees more of the implementation than RT ever does -- RT is transpiled
against the interfaces alone. Reported rates must be read as an upper bound, not as a
technique anyone could run without already having the fault in hand.
"""
import difflib
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import llm, rbench as R
from . import stage4_relate as s4
from .llm import ask, fenced
from .prompts import load as load_prompt

RESULTS = Path(__file__).resolve().parent.parent / "results"

SYSTEM = ("You are a precise Rust programmer writing a fault-detecting test. You reason "
          "about concrete values, you use only the public API you have been shown, and "
          "you never write a test that could fail a correct implementation.")

TEST_FN_RE = re.compile(r"#\[test\]")
UNREACHABLE_RE = re.compile(r"^UNREACHABLE:", re.M)


def diff_text(impl_dir, patch, context=6):
    """A unified diff of M -> M', built from the applied patch rather than the record.

    The edits stored in `mutants.json` are truncated to 600 characters each, so quoting
    them to the model would sometimes hand it half a fault. Diffing the *applied* files
    recovers the change at full fidelity -- and only the change, which is a fraction of
    the tokens the two file versions would cost.
    """
    parts = []
    for rel, new in sorted(patch.items()):
        old = (Path(impl_dir) / rel).read_text(errors="ignore")
        parts += list(difflib.unified_diff(
            old.splitlines(), new.splitlines(),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="", n=context))
    return "\n".join(parts)


def mp_detected(on_mp):
    """Did the test detect M'? `(killed, kind)`.

    `run_test` reports "build_error" for anything that produced no `test result:` line
    with a non-zero exit, which covers two very different events: the test genuinely not
    compiling, and the test *binary* dying mid-run. A stack overflow or a SIGSEGV in the
    mutated code takes the process down before cargo can print a result, and that is a
    detection -- the same rule the naive arm applies to a suite that aborts early. The
    harness's own "running N tests" line is what separates them: it is printed after the
    binary is built and has started.
    """
    if on_mp["status"] in ("test_failed", "timeout"):
        return True, on_mp["status"]
    if on_mp["status"] == "build_error":
        if "running " in on_mp.get("out", ""):
            return True, "aborted"
        return False, "test_no_compile_on_mp"
    return False, on_mp["status"]


def _feedback(on_m, on_mp):
    """Why the gate rejected this test, in the terms the model can act on."""
    if on_m["status"] == "build_error":
        errs = "\n".join(l for l in on_m["out"].splitlines()
                         if l.startswith(("error", "  -->")))[-4000:]
        return ("Your test does not COMPILE against the correct implementation.\n```\n"
                + (errs or on_m["out"][-3000:]) + "\n```\nFix only the compile errors --"
                " imports, paths, types, borrows, mutability. If a test calls something"
                " that is not in the public interfaces at all, delete that whole `#[test]`"
                " function rather than weakening it into something that builds.")
    if on_m["status"] == "timeout":
        return ("Your test TIMED OUT on the correct implementation -- an infinite loop or"
                " a blocking read. It must terminate quickly.")
    if on_m["status"] != "green":
        return ("Your test FAILS the correct implementation:\n```\n"
                + on_m["out"][-2500:] + "\n```\nThe expected value is wrong, or the call"
                " is outside the function's domain (an `Option` returning `None`"
                " type-checks and then fails your assertion). Trace the reference code"
                " above by hand for your exact input and assert what *it* produces.")
    if on_mp["status"] == "build_error":
        return ("Your test compiles against the correct implementation but NOT against"
                " the faulty one:\n```\n" + on_mp["out"][-2500:] + "\n```\nThe fault"
                " changes something your test depends on at the type level. Reach it"
                " through a call whose signature both versions share.")
    return ("Your test passes the correct implementation, but it also PASSES the faulty"
            " one -- it detects nothing. Your assertions are not sensitive to the changed"
            " lines. Find an input that reaches them and whose result the public API"
            " actually returns differently.")


def generate(impl_dir, patch, mutant, iface, crate, mods, retries=3, tag="rn"):
    """One mutant: write a test, repair it, gate it on M then on M'.

    Returns a record whose `outcome` is the arm's verdict for this mutant.
    """
    base = load_prompt(
        "rustnative_testgen", impl=R.impl_source(impl_dir), diff=diff_text(impl_dir, patch),
        function=mutant.get("function", "?"), fault_class=mutant.get("fault_class", "?"),
        iface=iface, crate=crate, modules=", ".join(mods), module0=mods[0])

    attempts, user, test_rs = [], base, None
    for i in range(retries + 1):
        # 24k, between Stage 2's 16k (one C test) and the naive arm's 32k (a whole suite):
        # one focused test, but read out of a full Rust implementation.
        reply = ask(SYSTEM, user, effort="high", max_tokens=24000, tag=f"rn_{tag}_{i}")
        if UNREACHABLE_RE.search(reply):
            attempts.append({"attempt": i, "outcome": "declared_unreachable",
                             "note": reply.strip()[:400]})
            return {"test_rs": None, "outcome": "UNREACHABLE", "attempts": attempts,
                    "n_tests": 0}
        test_rs = fenced(reply, "rust")
        if not test_rs:
            attempts.append({"attempt": i, "outcome": "no_code"})
            user = base + ("\n\nYour reply ran out of room before it reached the code"
                           " block. Return the test file and nothing else -- no"
                           " reasoning, no commentary.")
            continue

        # One CARGO_TARGET_DIR for both halves of the gate: the two runs are sequential
        # and differ only in the implementation, so M' reuses M's dependency build.
        on_m = R.run_test(impl_dir, test_rs, tag=f"rn_{tag}")
        on_mp = None
        if on_m["status"] == "green":
            on_mp = R.run_test(impl_dir, test_rs, patch=patch, tag=f"rn_{tag}")
            killed, kind = mp_detected(on_mp)
            if killed:
                attempts.append({"attempt": i, "outcome": "killed", "detected_by": kind})
                return {"test_rs": test_rs, "outcome": "KILLED", "attempts": attempts,
                        "n_tests": len(TEST_FN_RE.findall(test_rs)),
                        "on_m": on_m["status"], "on_mp": on_mp["status"],
                        "detected_by": kind}

        fb = _feedback(on_m, on_mp)
        if on_m["status"] != "green":
            step = "no_compile" if on_m["status"] == "build_error" else "unsound_on_m"
        else:
            step = ("mp_no_compile" if on_mp["status"] == "build_error"
                    else "not_discriminating")
        attempts.append({"attempt": i, "outcome": step,
                         "on_m": on_m["status"],
                         "on_mp": (on_mp or {}).get("status"),
                         "feedback": fb[:1200]})
        user = base + "\n\n---\n\n" + load_prompt("rustnative_retry", feedback=fb,
                                                  prev=test_rs)

    last = attempts[-1]["outcome"] if attempts else "no_code"
    return {"test_rs": test_rs, "n_tests": len(TEST_FN_RE.findall(test_rs or "")),
            "outcome": {"no_compile": "F_NO_COMPILE",
                        "unsound_on_m": "F_UNSOUND_ON_M",
                        "mp_no_compile": "F_MP_NO_COMPILE",
                        "not_discriminating": "F_NOT_DISCRIMINATING",
                        "no_code": "F_NO_CODE"}.get(last, "F_NO_CODE"),
            "attempts": attempts}


def _one_mutant(impl_dir, mutant, iface, crate, mods, outdir, retries, tag, redo=False):
    """Score one recorded M'. Cached per mutant, so a killed run resumes for free."""
    cache = outdir / f"{mutant['id']}.json"
    if cache.exists() and not redo:
        rec = json.loads(cache.read_text())
        if rec.get("outcome") not in ("F_NO_CODE", "ERROR"):
            return rec

    rec = {"id": mutant["id"], "function": mutant.get("function"),
           "fault_class": mutant.get("fault_class"),
           "mutant_kind": mutant.get("mutant_kind"),
           "rt_kills_mp": mutant["outcome"] == "KILLED",
           "gt_kills_mp": (mutant.get("stage4") or {}).get("gt_kills_mp")}
    try:
        patch = s4.apply_patch(impl_dir, mutant["stage4"]["rust_mutant"])
    except (ValueError, KeyError) as e:
        # The same exclusion the naive arm makes: an M' whose recorded edits no longer
        # apply verbatim cannot be replayed, so no arm can be scored on it.
        rec |= {"rn_kills_mp": None, "outcome": "UNREPLAYABLE",
                "detail": f"patch_unavailable: {e}"}
        cache.write_text(json.dumps(rec, indent=2))
        return rec

    # Applying cleanly is not the same as replaying faithfully. The recorded snippets are
    # truncated at 600 characters, so a large edit can still substitute one 600-char
    # prefix for another and leave Rust that does not parse -- Stage 4 verified that *its*
    # M' built, but this reconstruction of it is a different program. Scoring a test
    # against something that does not compile would charge the arm for a mutant that no
    # longer exists, so probe the build first and exclude it if it fails. Free, too: the
    # probe is one cargo build and no model call.
    probe = R.run_test(impl_dir, None, patch=patch, tag=f"rn_{tag}")
    if probe["status"] != "green":
        rec |= {"rn_kills_mp": None, "outcome": "UNREPLAYABLE",
                "detail": "replayed M' does not build (truncated edits)",
                "out": probe["out"][-1200:]}
        cache.write_text(json.dumps(rec, indent=2))
        return rec

    gen = generate(impl_dir, patch, mutant, iface, crate, mods, retries=retries, tag=tag)
    if gen["test_rs"]:
        (outdir / f"{mutant['id']}_test.rs").write_text(gen["test_rs"])
    rec |= {k: v for k, v in gen.items() if k != "test_rs"}
    # A mutant this arm could not produce a usable test for is a miss, not an exclusion:
    # it had the fault in hand and still did not detect it.
    rec["rn_kills_mp"] = gen["outcome"] == "KILLED"
    cache.write_text(json.dumps(rec, indent=2))
    return rec


def score_project(prof, outdir, retries=3, workers=4, log=print, redo=False,
                  max_mutants=None):
    name = prof["project"]
    out = outdir / "rustnative.json"
    if out.exists() and not redo:
        cached = json.loads(out.read_text())
        # `ok` alone is not enough: a `--max-mutants` pilot leaves an `ok` record covering
        # part of the project, and taking it at face value would silently drop the rest
        # from a later full run. The count is checked against the funnel below.
        if cached["status"] != "ok":
            cached = None
    else:
        cached = None

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

    # Exactly the naive arm's mutant set: Stage 4 built an M' and RT was scored on it.
    mutants = [m for m in json.loads(funnel.read_text())["mutants"]
               if m["outcome"] in ("KILLED", "F4_not_discriminating")
               and ((m.get("stage4") or {}).get("rust_mutant") or {}).get("applied")]
    if max_mutants:
        mutants = mutants[:max_mutants]
    if cached and len(cached.get("mutants", [])) >= len(mutants):
        log(f"{name}: ok (cached)")
        return cached

    tests = outdir / "rustnative"
    tests.mkdir(exist_ok=True)
    iface, crate = R.iface_source(name), R.crate_name(name)
    mods = R.lib_modules(name) or ["<module>"]

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_mutant, impl_dir, m, iface, crate, mods, tests, retries,
                          f"{name}_{m['id']}", redo): m for m in mutants}
        for f in as_completed(futs):
            mid = futs[f]["id"]
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001 - one mutant must not sink a project
                log(f"{name}/{mid}: ERROR {type(e).__name__}: {e}")
                results.append({"id": mid, "rn_kills_mp": None, "outcome": "ERROR",
                                "detail": f"exception: {type(e).__name__}",
                                "error": traceback.format_exc()[-1500:]})

    results.sort(key=lambda r: str(r["id"]))
    n = sum(1 for r in results if r["rn_kills_mp"] is not None)
    k = sum(1 for r in results if r["rn_kills_mp"])
    payload |= {"status": "ok", "mutants": results,
                "seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(payload, indent=2))
    log(f"{name}: rust-native kills {k}/{n}" + (f" ({100 * k / n:.0f}%)" if n else ""))
    return payload


def run(run_name, projects, retries=3, project_workers=4, mutant_workers=4, redo=False,
        max_mutants=None):
    root = RESULTS / run_name
    if not root.is_dir():
        raise SystemExit(f"no funnel run at {root} -- the baseline scores its mutants")
    todo = [p for p in projects if (root / p["project"] / "mutants.json").exists()]
    print(f"rust-native baseline: run={run_name} projects={len(todo)} "
          f"(+{len(projects) - len(todo)} with no funnel result) "
          f"workers={project_workers}x{mutant_workers}"
          + (f" max-mutants={max_mutants}" if max_mutants else ""), flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(prof):
        outdir = root / prof["project"]
        try:
            return score_project(prof, outdir, retries, mutant_workers, log, redo,
                                 max_mutants)
        except Exception:  # noqa: BLE001 - record and keep going
            (outdir / "RUSTNATIVE_ERROR.txt").write_text(traceback.format_exc())
            log(f"{prof['project']}: PROJECT ERROR (see RUSTNATIVE_ERROR.txt)")
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=project_workers) as ex:
        list(ex.map(one, todo))

    cost = llm.cost_ledger()
    cost["seconds"] = round(time.time() - t0)
    (root / "cost_rustnative.json").write_text(json.dumps(cost, indent=2))
    print(f"done in {cost['seconds']}s; {cost['total']['calls']} LLM calls, "
          f"${cost['total']['paid_usd']:.2f} paid", flush=True)
