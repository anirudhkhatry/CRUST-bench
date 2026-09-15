"""RefactorBench-JS wired into the shared harness.

The order below is the experiment: prepare and verify the oracle split, generate every
arm's suite from P0 alone, freeze them, and only then build candidates. Nothing may
reorder these steps -- `freeze` refuses once candidates exist.
"""
import json

from ..adapters import js_oracle, js_vitest as JS
from ..arms import mutation, naive
from ..harvest import label, propose_candidate
from ..runner import freeze, score_task
from ..seed import seed_faults
from ..spec import Candidate, Task, write_metadata

# Arm N is the PRIMARY bad population (spec §5). A spread of model strength and
# temperature is what makes the candidate set contain both correct and incorrect work: a
# single strong model at temperature 0 produces near-identical candidates that are almost
# all GOOD, and yields no BADs to measure against.
MODELS = [("claude-haiku-4-5-20251001", 1.0), ("claude-haiku-4-5-20251001", 1.0),
          ("claude-opus-5", 1.0), ("claude-haiku-4-5-20251001", 0.3)]
TRANSFORMATION = ("Decompose the target file into smaller, well-structured modules while "
                  "preserving all externally observable behaviour.")


def build_task(name, results_root):
    return Task(task_id=name, benchmark="refactorbench_js", language="javascript",
                repository="Create-Inc/refactor-bench", source_commit="main",
                transformation=TRANSFORMATION, results_root=results_root)


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
    n_suite, rec = naive.generate(task, sources, JS, p0_ws, imp, tag=f"{name}_naive")
    suites["naive"], admissible["naive"], records["naive"] = n_suite, rec["admissible"], rec
    m_suite, rec = mutation.generate(task, sources, JS, ws / "mut", imp,
                                     tag=f"{name}_mut")
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
        candidates[cid] = (Candidate(cid, name, lbl, None if lbl == "GOOD" else "N",
                                     None, model, model, {"temperature": temp}, "ok", why),
                           cand["sources"])

    # 5. Arm S: seeded faults in a verified-GOOD migration, confirmed by the oracle.
    # DIAGNOSTIC, not primary (spec §5): measured yield is ~8% on complex subjects, since
    # a blind snippet edit usually lands where the tests do not look. Its value is the
    # per-fault-category label, which Arm N cannot supply. A task whose Arm S yield is zero
    # is not a failed task.
    for j, f in enumerate(seed_faults(task, good_sources or sources, n_bad,
                                      f"{name}_seed") if (good_sources or sources) else []):
        lbl, why = label(JS, task, f["sources"], oracle["oracle_text"], ws / f"s{j}")
        if lbl != "BAD":
            continue          # equivalent fault, or it broke the build: excluded (§3.5)
        cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        candidates[cid] = (Candidate(cid, name, "BAD", "S", f["fault_category"],
                                     "claude-opus-5", "claude-opus-5", {}, "ok", why),
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
    (task.root / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"{name}: {len(candidates)} candidates, {len(rows)} rows")
    return rows
