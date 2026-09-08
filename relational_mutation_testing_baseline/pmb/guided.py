"""Experiment 2 -- can the migrated tests be used to *improve* a migration?

Experiment 1 asks whether T' can tell a good migration from a bad one. This asks the
question that would make T' worth generating in practice: hand the failing test back to
the model that wrote the migration and let it try again. If the candidate gets better,
the tests are not merely a measuring instrument, they are a repair signal.

Two arms, and the second is what makes the first mean anything:

    guided     the model is shown the migrated tests that reject its candidate, with the
               pytest output, and asked to fix the migration.
    blind      the model is told only that its candidate may be wrong and asked to try
               again. Same model, same temperature, same number of extra attempts, no
               test feedback.

Without `blind`, any improvement is unattributable: a second sample from a strong model
is better than a first sample often enough to manufacture a result. The quantity of
interest is the *difference* between the arms, not the guided arm's raw repair rate.

Scoring uses the same three oracles as Experiment 1, so a repair that satisfies the tests
while breaking the developer's recorded edits is visible as such:

    suite      every test that passed on M still passes on the repaired candidate
    diff       the recorded code changes are all present (changed_dev)
    compiles   it still parses

And the failure mode this experiment exists to detect is **overfitting to T'**: a repair
that turns T' green without making the migration right. That shows up as the guided arm
gaining on the T' verdict while flat or falling on the suite and diff oracles, and it is
reported explicitly rather than left to be inferred.
"""
import ast
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import llm
from . import pybench as PB
from . import validate as V
from .llm import MODEL, ask, fenced
from .prompts import load as load_prompt

RESULTS = Path(__file__).resolve().parent.parent / "results"

SYSTEM = "You are a helpful assistant that migrates Python code between libraries."

MAX_FEEDBACK = 3       # failing tests shown per attempt
MAX_OUT = 1500         # characters of pytest output per test


def failing(per):
    """The rejecting test records from a `validate` judgement, worst first.

    A test that failed an assertion is better feedback than one that failed to load, so
    behavioural rejections are shown before structural ones: the first says what the
    migration got wrong, the second only says the two files differ in shape.
    """
    rej = [p for p in per if p.get("verdict") in (V.REJECT, V.LOAD_FAILED)]
    rej.sort(key=lambda p: V.rejection_kind(p.get("out", "")) != "behavioural")
    return rej[:MAX_FEEDBACK]


def render_feedback(rej, tests):
    out = []
    for p in rej:
        tid = p["test"]
        out.append(f"### Test `{tid}` fails on your migration\n\n```python\n"
                   f"{tests.get(tid, '(source unavailable)')}\n```\n\n"
                   f"pytest output:\n\n```\n{(p.get('out') or '').strip()[-MAX_OUT:]}\n```")
    return "\n\n".join(out)


def repair(mig, pre_src, candidate_src, rel, arm, feedback, tag, retries=2):
    """One repair attempt. Returns (text|None, trail)."""
    prompt = "repair_guided" if arm == "guided" else "repair_blind"
    user = load_prompt(prompt, source=mig.source, target=mig.target,
                       impl=pre_src, candidate=candidate_src, feedback=feedback)
    trail = []
    for i in range(retries + 1):
        reply = ask(SYSTEM, user, effort="high", max_tokens=64000,
                    tag=f"rep_{arm}_{tag}_{i}")
        code = fenced(reply, "python")
        if not code or not code.strip():
            trail.append({"attempt": i, "outcome": "no_code_block"})
            user += ("\n\n---\n\nThat reply contained no fenced Python code block. "
                     "Provide the complete modified file in one ```python block.")
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            trail.append({"attempt": i, "outcome": "syntax_error", "detail": str(e)[:200]})
            user += (f"\n\n---\n\nThat code does not parse: {e}. Provide the complete "
                     "modified file in one ```python block.")
            continue
        ratio = len(code.splitlines()) / max(1, len(pre_src.splitlines()))
        if ratio < 0.6:
            trail.append({"attempt": i, "outcome": "truncated", "ratio": round(ratio, 2)})
            user += (f"\n\n---\n\nThat reply contained only {round(100 * ratio)}% of the "
                     "file's lines. Provide the *complete* file.")
            continue
        trail.append({"attempt": i, "outcome": "ok"})
        return code, trail
    return None, trail


def score(mig, post, text, rel, tests):
    """Judge one repaired candidate on all three oracles, exactly as Experiment 1 does.

    Deliberately the same calls as `validate.one_migration`, not a reimplementation: the
    whole claim of this experiment is a before/after comparison, so the "after" has to be
    measured by the instrument that produced the "before".
    """
    per = [V.judge_file(post, t, text, rel, tid) for tid, t in tests]
    adm = [p for p in per if p["admissible"]]
    rej = [p for p in adm if p.get("verdict") in (V.REJECT, V.LOAD_FAILED)]
    behav = [p for p in rej if p.get("verdict") == V.REJECT
             and V.rejection_kind(p.get("out", "")) == "behavioural"]
    label, detail = V.suite_label(post, text, rel)
    return {"n_admissible": len(adm), "n_rejecting": len(rej),
            "n_behavioural": len(behav),
            "rt_verdict": (V.REJECT if rej else V.ACCEPT if adm else None),
            "rt_verdict_behavioural": (V.REJECT if behav else V.ACCEPT if adm else None),
            "truth": label, "truth_detail": detail,
            "compiles": True, "tests": per}


def one_migration(run_root, mid, model, before, arm, redo=False, log=print):
    """Repair one rejected candidate under one arm, and re-score it.

    Only candidates T' *rejects* are repaired. A candidate the tests already accept has
    no feedback to offer the guided arm, so including it would dilute both arms with a
    task that is not the one under study -- and on the blind arm it would amount to asking
    the model to churn code it was given no reason to doubt.
    """
    d = run_root / mid / "repair" / model / arm
    d.mkdir(parents=True, exist_ok=True)
    done = d / "repair.json"
    if done.is_file() and not redo:
        return json.loads(done.read_text())

    mig = V.by_id(mid)
    cand_dir = run_root / mid / "candidates" / model
    candidate_src = (cand_dir / "candidate.py").read_text()
    pre, post = PB.open_migration(mig)
    rel = before.get("file") or post.rel

    tests = V.tests_of(run_root, mid)
    rec = {"id": mid, "model": model, "arm": arm, "file": rel,
           "source": mig.source, "target": mig.target,
           "before": {k: before.get(k) for k in
                      ("rt_verdict", "rt_verdict_behavioural", "n_rejecting",
                       "n_behavioural", "n_admissible", "truth")}}

    fb = ""
    if arm == "guided":
        rej = failing(before.get("tests", []))
        if not rej:
            rec["status"] = "no_feedback"
            done.write_text(json.dumps(rec, indent=2))
            return rec
        fb = render_feedback(rej, dict(tests))
        rec["n_tests_shown"] = len(rej)

    text, trail = repair(mig, pre.read(), candidate_src, rel, arm, fb, tag=mid)
    rec["trail"] = trail
    if text is None:
        rec["status"] = "no_patch"
        done.write_text(json.dumps(rec, indent=2))
        return rec

    (d / "repaired.py").write_text(text)
    probe = PB.run_test(post, None, patch={rel: text})
    rec["imports"] = probe["status"] == "green"
    if not rec["imports"]:
        rec["status"] = "no_import"
        rec["import_error"] = probe["out"][-1200:]
        done.write_text(json.dumps(rec, indent=2))
        return rec

    rec["after"] = score(mig, post, text, rel, tests)
    rec["status"] = "scored"
    # The single quantity the experiment turns on: did the repair change the verdict?
    rec["repaired"] = (before.get("rt_verdict") == V.REJECT
                       and rec["after"]["rt_verdict"] == V.ACCEPT)
    log(f"{mid} [{arm}]: {before.get('rt_verdict')} -> "
        f"{rec['after']['rt_verdict']} truth={rec['after']['truth']} "
        f"(was {before.get('truth')})")
    done.write_text(json.dumps(rec, indent=2))
    return rec


def run(run_name, model, arms=("guided", "blind"), ids=None, workers=2, redo=False):
    root = RESULTS / run_name
    rows = json.loads((root / f"validation_{model}.json").read_text())
    rejected = [r for r in rows
                if r.get("status") == "judged" and r.get("rt_verdict") == V.REJECT]
    if ids is not None:
        rejected = [r for r in rejected if r["id"] in set(ids)]
    jobs = [(r, arm) for r in rejected for arm in arms]
    print(f"guided: run={run_name} model={model} rejected={len(rejected)} "
          f"arms={list(arms)} jobs={len(jobs)} workers={workers}", flush=True)

    def log(msg):
        print(f"  {msg}", flush=True)

    def one(job):
        before, arm = job
        try:
            return one_migration(root, before["id"], model, before, arm,
                                 redo=redo, log=log)
        except Exception as e:  # noqa: BLE001 - one repo must not sink the panel
            return {"id": before["id"], "arm": arm, "status": "error",
                    "err": f"{type(e).__name__}: {e}",
                    "error": traceback.format_exc()[-1500:]}

    t0, out = time.time(), []
    out_path = root / f"repair_{model}.json"
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            out_path.write_text(json.dumps(out, indent=2))
            print(f"[{n}/{len(jobs)}] {out[-1]['id']} [{out[-1]['arm']}]: "
                  f"{out[-1]['status']}", flush=True)

    out.sort(key=lambda r: (r["id"], r["arm"]))
    out_path.write_text(json.dumps(out, indent=2))
    cost = llm.cost_ledger()
    print(f"\nrepair done in {round(time.time() - t0)}s; "
          f"${cost['total']['paid_usd']:.2f} paid cumulative", flush=True)
    return out


def report(run_name, model):
    """Guided vs blind, on T' and on the two oracles that can contradict it."""
    root = RESULTS / run_name
    rows = json.loads((root / f"repair_{model}.json").read_text())

    def pct(a, b):
        return f"{100 * a / b:.0f}%" if b else "n/a"

    L = [f"# Can the migrated tests *improve* a migration? ({run_name}, {model})", "",
         "Every candidate below was rejected by T' in Experiment 1. Each was handed back "
         "to the same model twice: once with the failing tests (**guided**), once with "
         "only the news that it might be wrong (**blind**). Same model, same effort, same "
         "retry budget. The quantity of interest is the *difference*, because a second "
         "sample from a strong model beats a first sample often enough to manufacture a "
         "result on its own.", ""]

    by_arm = {}
    for arm in ("guided", "blind"):
        a = [r for r in rows if r.get("arm") == arm]
        by_arm[arm] = a
        L.append(f"- **{arm}**: {len(a)} attempts, "
                 + ", ".join(f"{k}={sum(1 for r in a if r.get('status') == k)}"
                             for k in sorted({r.get("status") for r in a})))

    L += ["", "## Did the tests go green?", "",
          "| arm | scored | T' now accepts | behavioural accept |",
          "|---|---:|---:|---:|"]
    for arm in ("guided", "blind"):
        s = [r for r in by_arm[arm] if r.get("status") == "scored"]
        acc = sum(1 for r in s if r["after"]["rt_verdict"] == V.ACCEPT)
        bacc = sum(1 for r in s if r["after"]["rt_verdict_behavioural"] == V.ACCEPT)
        L.append(f"| {arm} | {len(s)} | {acc} ({pct(acc, len(s))}) | "
                 f"{bacc} ({pct(bacc, len(s))}) |")

    # The check that decides whether a green T' means anything. A repair that satisfies
    # the tests without making the migration right shows up here and nowhere else.
    L += ["", "## Did the migration actually get better?", "",
          "The suite oracle is independent of T': it is the repository's own tests, "
          "scored as 'every case that passed on M still passes'. If the guided arm gains "
          "on T' but not here, the repairs are overfitting to the instrument.", "",
          "| arm | suite-labelled | correct after | incorrect after | was incorrect |",
          "|---|---:|---:|---:|---:|"]
    for arm in ("guided", "blind"):
        s = [r for r in by_arm[arm] if r.get("status") == "scored"]
        lab = [r for r in s if r["after"]["truth"] in ("correct", "incorrect")]
        ok = sum(1 for r in lab if r["after"]["truth"] == "correct")
        bad = sum(1 for r in lab if r["after"]["truth"] == "incorrect")
        was = sum(1 for r in lab if r["before"].get("truth") == "incorrect")
        L.append(f"| {arm} | {len(lab)} | {ok} ({pct(ok, len(lab))}) | {bad} | {was} |")

    # Per-migration turnaround on the suite oracle, which is the only place a genuine
    # repair can be distinguished from a satisfied test.
    fixed = {arm: [r for r in by_arm[arm] if r.get("status") == "scored"
                   and r["before"].get("truth") == "incorrect"
                   and r["after"]["truth"] == "correct"]
             for arm in ("guided", "blind")}
    L += ["", "### Migrations the repair genuinely fixed (suite: incorrect -> correct)", ""]
    for arm in ("guided", "blind"):
        L.append(f"- **{arm}**: {len(fixed[arm])}"
                 + ("".join(f"\n  - `{r['id']}` ({r['source']}->{r['target']})"
                            for r in fixed[arm]) if fixed[arm] else ""))

    broke = {arm: [r for r in by_arm[arm] if r.get("status") == "scored"
                   and r["before"].get("truth") == "correct"
                   and r["after"]["truth"] == "incorrect"]
             for arm in ("guided", "blind")}
    if any(broke.values()):
        L += ["", "### Migrations the repair broke (suite: correct -> incorrect)", ""]
        for arm in ("guided", "blind"):
            for r in broke[arm]:
                L.append(f"- **{arm}** `{r['id']}` ({r['source']}->{r['target']})")

    g = [r for r in by_arm["guided"] if r.get("status") == "scored"]
    b = [r for r in by_arm["blind"] if r.get("status") == "scored"]
    ga = sum(1 for r in g if r["after"]["rt_verdict"] == V.ACCEPT)
    ba = sum(1 for r in b if r["after"]["rt_verdict"] == V.ACCEPT)
    L += ["", "## Reading", "",
          f"T' acceptance after repair: guided {pct(ga, len(g))}, blind {pct(ba, len(b))}. "
          f"Suite-confirmed fixes: guided {len(fixed['guided'])}, "
          f"blind {len(fixed['blind'])}.", "",
          "Both numbers matter and they answer different questions. The first says the "
          "feedback is usable; only the second says it is *true*. A large gap on the "
          "first with none on the second is the overfitting result, and it is still a "
          "finding -- it would say the tests are a repair signal the model can satisfy "
          "without migrating correctly."]

    md = "\n".join(L) + "\n"
    (root / f"REPAIR_{model}.md").write_text(md)
    print(md)
    return md
