"""RefactorBench-JS wired into the shared harness.

The order below is the experiment: prepare and verify the oracle split, generate every
arm's suite from P0 alone, freeze them, and only then build candidates. Nothing may
reorder these steps -- `freeze` refuses once candidates exist.
"""
import json

from rmt import llm

from ..adapters import js_oracle, js_vitest as JS
from ..arms import ceiling, mutation, naive
from ..harvest import label, propose_candidate
from ..runner import freeze, score_task
from ..seed import seed_faults
from ..spec import Candidate, Task, sources_hash, write_metadata

# Arm N is the PRIMARY bad population (spec §5), so the candidate set has to contain both
# correct and incorrect work.
#
# WHAT THIS LIST ACTUALLY DOES, which is less than it looks: `rmt.llm.ask` takes a
# `temperature` argument and never forwards it, and has no `model` argument at all -- it
# always calls `rmt.llm.MODEL`. `rmt/llm.py` is shared with two other pipelines and is out
# of scope to change here. So every entry below produces a call to the SAME model at the
# SAME temperature, and the only thing the entries vary is the cache tag, which is enough
# to make each attempt a fresh sample rather than a cache hit. The candidate spread is
# therefore sampling noise at whatever temperature the API defaults to, and nothing that
# is recorded on a Candidate may claim otherwise (see `_candidate_provenance`).
# The strings are kept verbatim because they are part of the LLM cache key (`ask` names
# its cache entry `<tag>_<hash>.json`): renaming them would re-pay for every candidate
# already generated. They are attempt LABELS, not a model selection.
MODELS = [("claude-haiku-4-5-20251001", 1.0), ("claude-haiku-4-5-20251001", 1.0),
          ("claude-opus-5", 1.0), ("claude-haiku-4-5-20251001", 0.3)]
TRANSFORMATION = ("Decompose the target file into smaller, well-structured modules while "
                  "preserving all externally observable behaviour.")


def task_pool():
    """The benchmark's task pool. The entrypoint asks the BENCHMARK for this rather than
    reaching into a JS adapter, so `select` and `run` cannot drift apart per benchmark."""
    return JS.pure_logic_fixtures()


def build_task(name, results_root):
    return Task(task_id=name, benchmark="refactorbench_js", language="javascript",
                repository="Create-Inc/refactor-bench", source_commit="main",
                transformation=TRANSFORMATION, results_root=results_root)


def _candidate_provenance():
    """What a Candidate may honestly claim about how it was produced (spec §9).

    `rmt.llm.ask` neither forwards a temperature nor accepts a model, so the only true
    statement available is the module-level model it really calls. `temperature` is
    recorded as an explicit null rather than the number we asked for and did not get: a
    metadata field that names a temperature nobody applied is worse than an absent one,
    because it is indistinguishable from a real setting when the artifact is read later.
    `llm.MODEL` is read at call time, not import time, so it reflects RMT_MODEL for the
    run that actually produced the candidate.
    """
    return llm.MODEL, {"temperature": None,
                       "temperature_note": "rmt.llm.ask does not forward a temperature; "
                                           "candidate spread is sampling noise only"}


def _sources(name):
    src = JS.FIXTURES / name / "src"
    return {str(p.relative_to(src)): p.read_text()
            for p in sorted(src.rglob("*")) if p.is_file()}


def run_task(name, results_root, n_good=3, n_bad=3, workspace=None, log=print):
    task = build_task(name, results_root)
    task.root.mkdir(parents=True, exist_ok=True)
    ws = workspace or (task.root / "_work")
    ws.mkdir(parents=True, exist_ok=True)
    sources = _sources(name)
    target = JS.fixture_config(name)["targetFile"]

    # 1. Oracle split, verified against P0 (spec §4.4).
    oracle = js_oracle.prepare_oracle(name, ws / "oracle_ws")
    task.hidden_oracle_dir.mkdir(parents=True, exist_ok=True)
    (task.hidden_oracle_dir / "oracle.test.jsx").write_text(oracle["oracle_text"])
    log(f"{name}: arm_a_usable={oracle['arm_a_usable']} {oracle['reason']}")

    # `prepare_oracle` always PRESERVES the oracle, but preserved is not the same as
    # populated: a fixture whose holdout does not build reports zero admissible cases.
    # Such a fixture can label nothing, so every candidate would come back GOOD by
    # default and the task would be silently meaningless -- while still paying for a full
    # set of LLM calls. Bail before spending any of them. Measured: 4 of the 18 pure-logic
    # fixtures fail this way (their holdouts import modules the fixture does not ship).
    if not oracle["admissible_oracle"]:
        log(f"{name}: SKIPPED -- oracle has no admissible case ({oracle['reason']})")
        write_metadata(task, [], extra={"skipped": "oracle has no admissible case",
                                        "oracle_reason": oracle["reason"]})
        return []

    # 2. Every arm's suite, from P0 and the transformation only.
    p0_ws = JS.stage(name, ws / "p0")
    suites, admissible, records = {}, {}, {}
    imp = f"import * as mod from '{JS.import_path(name)}';"
    # §7.2: the held-out developer tests are the only text that exists at this point and
    # must not reach a generating arm's prompt. P0's own sources are in both prompts by
    # design and so are deliberately NOT forbidden; no candidate exists yet (the freeze
    # below enforces that), and a candidate is mostly P0 verbatim, so passing candidate
    # text here would fire on files nobody changed.
    forbidden = [oracle["oracle_text"], oracle["existing_text"]]
    n_suite, rec = naive.generate(task, sources, JS, p0_ws, imp, tag=f"{name}_naive",
                                  forbidden=forbidden)
    records["naive"] = rec
    # Mirrors the ARM M guard below. `naive.generate`'s repair is build-errors-only, so a
    # suite that COMPILES but fails every assertion on P0 (or never built at all, leaving
    # `n_suite` `None`) comes back with an empty `admissible` list. Registering it
    # unconditionally would hand `score_task` an arm with an empty admissible set, which
    # it refuses outright -- discarding this task's rows, and every paid call already
    # spent on its candidates, for an arm that was never scorable to begin with.
    if n_suite and rec["admissible"]:
        suites["naive"], admissible["naive"] = n_suite, rec["admissible"]
    else:
        log(f"{name}: ARM N produced no scorable suite "
            f"(compiles={rec.get('compiles')}, n_admissible={len(rec.get('admissible') or [])}) "
            f"-- omitted from scoring")
    m_suite, rec = mutation.generate(task, sources, JS, ws / "mut", imp,
                                     tag=f"{name}_mut", forbidden=forbidden)
    m_run = JS.run_suite(p0_ws, m_suite, "mut_p0") if m_suite else None
    records["mutation"] = rec
    # An arm that produced NO suite is omitted rather than registered empty. `score_task`
    # refuses an arm with an empty admissible set, because such an arm silently accepts
    # every candidate -- but ARM M keeping nothing is a legitimate, informative outcome
    # (no mutant yielded a killing test), not a reason to crash the task and lose the
    # naive arm's results with it. An omitted arm simply has no row, which is honest;
    # `records["mutation"]["n_kept"] == 0` says why.
    if m_suite and m_run and m_run.passed:
        suites["mutation"] = m_suite
        admissible["mutation"] = m_run.passed
    else:
        log(f"{name}: ARM M produced no scorable suite "
            f"(n_kept={rec.get('n_kept')}, composite={rec.get('composite_status')}) "
            f"-- omitted from scoring")

    # 3. Freeze. Candidates must not exist yet. Baseline A's half of the holdout is frozen
    # alongside the generated arms so it passes through the same gate they do.
    to_freeze = dict(suites)
    if oracle["arm_a_usable"]:
        to_freeze["existing"] = oracle["existing_text"]
    hashes = freeze(task, to_freeze)
    log(f"{name}: froze {list(hashes)}")

    # 4. Candidates. Arm N is the primary BAD population, so this loop runs until it has
    # enough of BOTH labels rather than stopping at n_good: a run that collected only GOOD
    # candidates has nothing to measure detection against.
    candidates, good_sources = {}, None
    for i, (model, temp) in enumerate(MODELS * 3):
        n_g = sum(1 for c, _ in candidates.values() if c.label == "GOOD")
        n_b = sum(1 for c, _ in candidates.values()
                  if c.label == "BAD" and c.bad_arm == "N")
        if n_g >= n_good and n_b >= n_bad:
            break
        cand = propose_candidate(task, sources, target, model, temp, f"{name}_c{i}")
        if not cand:
            continue
        lbl, why = label(JS, task, cand["sources"], oracle["oracle_text"], ws / f"c{i}")
        if lbl == "EXCLUDE":
            continue
        if lbl == "GOOD":
            cid = f"good_{sum(1 for c, _ in candidates.values() if c.label == 'GOOD') + 1:02d}"
            good_sources = good_sources or cand["sources"]
        else:
            cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        used_model, gen_params = _candidate_provenance()
        candidates[cid] = (Candidate(cid, name, lbl, None if lbl == "GOOD" else "N",
                                     None, used_model, used_model, gen_params, "ok", why,
                                     candidate_commit=sources_hash(cand["sources"])),
                           cand["sources"])

    # 5. Arm S: seeded faults in a verified-GOOD migration, confirmed by the oracle.
    # DIAGNOSTIC, not primary (spec §5): measured yield is ~8% on complex subjects, since
    # a blind snippet edit usually lands where the tests do not look. Its value is the
    # per-fault-category label, which Arm N cannot supply. A task whose Arm S yield is zero
    # is not a failed task.
    #
    # With no verified-GOOD migration there is nothing to seed INTO. Falling back to P0
    # would seed faults into the pre-migration program -- not a migration at all, so not an
    # Arm S candidate -- and would reintroduce the circularity `seed.py` exists to prevent,
    # since ARM M's tests were built to kill mutants in that very P0.
    if good_sources is None:
        seeded = []
        log(f"{name}: no verified-GOOD migration -- Arm S skipped (seeding into P0 would "
            f"score faults in the pre-migration program as a migration, and ARM M's tests "
            f"were built against that same P0)")
    else:
        seeded = seed_faults(task, good_sources, n_bad, f"{name}_seed")
    seed_model, _ = _candidate_provenance()
    for j, f in enumerate(seeded):
        lbl, why = label(JS, task, f["sources"], oracle["oracle_text"], ws / f"s{j}")
        if lbl != "BAD":
            continue          # equivalent fault, or it broke the build: excluded (§3.5)
        cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        candidates[cid] = (Candidate(cid, name, "BAD", "S", f["fault_category"],
                                     seed_model, seed_model, {}, "ok", why,
                                     candidate_commit=sources_hash(f["sources"])),
                           f["sources"])

    for cid, (cand, src) in candidates.items():
        d = cand.dir(task)
        d.mkdir(parents=True, exist_ok=True)
        (d / "sources.json").write_text(json.dumps(src, indent=2))

    write_metadata(task, [c for c, _ in candidates.values()],
                   extra={"suite_hashes": hashes, "oracle": {
                       "arm_a_usable": oracle["arm_a_usable"], "reason": oracle["reason"],
                       "assignment": oracle["assignment"]},
                       "arms": {k: {x: v[x] for x in v if x != "suite"}
                                for k, v in records.items()}})

    rows = score_task(task, candidates, suites, admissible, JS, ws / "score",
                      existing_text=oracle["existing_text"] if oracle["arm_a_usable"] else None,
                      existing_admissible=oracle["admissible_existing"])
    # ARM C is exempt from the freeze: it is handed the GOOD->BAD diff by construction,
    # which is exactly why it is a ceiling and not a technique.
    if good_sources:
        rows += ceiling.score_ceiling(task, candidates, JS, ws / "ceiling", good_sources,
                                      imp)
    else:
        log(f"{name}: no verified-GOOD candidate, ARM C skipped")
    (task.root / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"{name}: {len(candidates)} candidates, {len(rows)} rows")
    return rows
