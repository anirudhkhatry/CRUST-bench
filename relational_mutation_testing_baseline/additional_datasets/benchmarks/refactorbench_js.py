"""RefactorBench-JS wired into the shared harness.

The order below is the experiment: prepare and verify the oracle (the whole developer
suite, which is also ARM A and whose verdicts are therefore definitional), generate every
arm's suite from P0 alone, freeze them, and only then build candidates. Nothing may
reorder these steps -- `freeze` refuses once candidates exist.
"""
import json

from rmt import llm

from ..adapters import js_oracle, js_split, js_vitest as JS
from ..arms import ceiling, mutation, naive
from ..arms.assemble import N_DRAWS, assemble
from ..harvest import label, propose_candidate
from ..runner import freeze, score_task, suite_files
from ..seed import seed_faults
from ..spec import Candidate, Task, sources_hash, write_metadata

# Arm N is the PRIMARY bad population (spec §5), so the candidate set has to contain both
# correct and incorrect work -- a fixture that yields only GOOD candidates measures no
# arm's ability to reject, and one that yields only BAD measures no arm's ability to
# accept. Either way it is unusable for the comparison, and `run_task` says so in metadata
# rather than letting it flatter whichever arm rejects more.
#
# This is a MODEL-STRENGTH LADDER, not a temperature sweep. The temperature sweep it
# replaces produced 0 GOOD candidates from 12 Haiku draws on a 1,600-line React component
# (measured, run `expt2`): every single one failed the oracle, so GAR was n/a for every arm
# and the fixture could not test half the hypothesis. Two reasons it could not work:
#
#   1. `temperature` reaches no API call. anthropic 1.5.0 exposes no temperature parameter
#      on messages.stream() or messages.create(). It still varies the CACHE KEY, so each
#      entry is an independently sampled draw -- but the draws differ only by sampling
#      noise, not by any controlled setting.
#   2. Sampling noise does not span correctness. Refactoring a large component either is
#      or is not within a model's reach; resampling a model that cannot do it yields twelve
#      different wrong answers.
#
# Model strength does span it, which is what the design wanted all along: a weak model
# supplies the BAD candidates and a strong one the GOOD, so the population contains both
# labels by construction rather than by luck. The label is then a property of which model
# drew the candidate -- recorded per candidate, so it is visible in the results and can be
# controlled for.
STRONG_MODEL = "claude-opus-5"
MID_MODEL = "claude-sonnet-5"
WEAK_MODEL = "claude-haiku-4-5-20251001"
CANDIDATE_MODEL = WEAK_MODEL          # kept: referenced elsewhere as the default drawer
# Strongest first, so a GOOD candidate is found early and Arm S (which needs a
# verified-GOOD migration to seed into) is not skipped for want of one.
MODELS = [(STRONG_MODEL, 1.0), (MID_MODEL, 1.0), (WEAK_MODEL, 1.0),
          (STRONG_MODEL, 1.0), (MID_MODEL, 1.0), (WEAK_MODEL, 1.0)]
# The ladder alone does not span the labels either, in the opposite direction: once the
# output budget is large enough to finish the rewrite, 11 of 12 draws come back GOOD
# (measured, run `expt4` on analytics_dashboard), and Arm S admitted 0 BAD from 8 seeded
# faults. So the sweep ENDS with draws from the weak model under a tight output budget --
# the regime that produced the only genuine harvested BAD candidates we have seen. They are
# ordinary Arm N candidates: the same prompt, no sight of any test, labelled only by the
# oracle. Only the room they are given differs, which is a property of the migration agent,
# not of the experiment. They come last because the loop stops as soon as it holds both
# labels, so a fixture whose weak draws already broke something never pays for them.
CONSTRAINED = [(WEAK_MODEL, 1.0, "tight")] * 4
# The sweep is exhausted after this many draws. Repeated entries are not cache hits (the
# cache tag carries the draw index), so each is a fresh sample from that model.
CANDIDATE_SWEEP = [(m, t, "full") for m, t in MODELS * 2] + CONSTRAINED
TRANSFORMATION = ("Decompose the target file into smaller, well-structured modules while "
                  "preserving all externally observable behaviour.")
# How the two generating arms are named in the log and in `matched_skipped`. The keys are
# the record/arm keys used throughout `run_task`.
ARM_NAMES = {"naive": "ARM N", "mutation": "ARM M"}


def task_pool():
    """The benchmark's task pool. The entrypoint asks the BENCHMARK for this rather than
    reaching into a JS adapter, so `select` and `run` cannot drift apart per benchmark."""
    return JS.pure_logic_fixtures()


def build_task(name, results_root):
    return Task(task_id=name, benchmark="refactorbench_js", language="javascript",
                repository="Create-Inc/refactor-bench", source_commit="main",
                transformation=TRANSFORMATION, results_root=results_root)


def _candidate_provenance(cand):
    """What a Candidate may honestly claim about how it was produced (spec §9).

    The values come back from `propose_candidate`, which is the function that made the
    call -- not from the constants above, which are only what we asked for. Before 9176dcb
    `ask` accepted no model and dropped the temperature, so this recorded an explicit null;
    it now records the numbers that were actually applied.
    """
    return cand["model"], {"temperature": cand["temperature"],
                           "output_budget": cand.get("budget"),
                           "sampled_by": "rmt.llm.ask(model=..., temperature=...)"}


def _seed_provenance():
    """Arm S's faults are seeded by `rmt.llm.ask` at its module default model.

    `llm.MODEL` is read at call time, not import time, so it reflects RMT_MODEL for the
    run that actually produced the fault. `seed.seed_faults` passes no temperature, so
    none is claimed.
    """
    return llm.MODEL, {"temperature": None}


def _sources(name):
    src = JS.FIXTURES / name / "src"
    return {str(p.relative_to(src)): p.read_text()
            for p in sorted(src.rglob("*")) if p.is_file()}


def run_task(name, results_root, n_good=3, n_bad=3, draws=N_DRAWS, workspace=None,
             log=print):
    """`draws` sets BOTH arms' draw count -- ARM N's `n_draws` and ARM M's `n_mutants`.

    One knob, not two, because their equality IS the matched control: two parameters could
    be given different values by a caller shrinking a pilot, and the comparison would
    silently become "six filtered samples against three". The equality is additionally
    asserted below against what the arms report they were asked for.
    """
    task = build_task(name, results_root)
    task.root.mkdir(parents=True, exist_ok=True)
    ws = workspace or (task.root / "_work")
    ws.mkdir(parents=True, exist_ok=True)
    sources = _sources(name)
    target = JS.fixture_config(name)["targetFile"]

    # 1. The oracle: the WHOLE developer suite, verified green against P0 (spec §4.4).
    # It labels every candidate, and it is therefore also ARM A -- whose GAR and BRR are
    # both 1.0 by construction. ARM A's rows are still emitted so the table is complete,
    # and `score.report` renders both its figures as `definitional` and excludes it from
    # comparison. The old half-and-half split is superseded (spec §4.4).
    oracle = js_oracle.prepare_oracle(name, ws / "oracle_ws")
    task.hidden_oracle_dir.mkdir(parents=True, exist_ok=True)
    (task.hidden_oracle_dir / "oracle.test.jsx").write_text(oracle["oracle_text"])
    log(f"{name}: oracle admits {len(oracle['admissible_oracle'])} case(s) "
        f"{oracle['reason']}")

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
    forbidden = [oracle["oracle_text"]]
    # Both arms return a LIST of test-file texts, one per surviving draw -- see `assemble`.
    n_files, nrec = naive.generate(task, sources, JS, p0_ws, imp, tag=f"{name}_naive",
                                   n_draws=draws, forbidden=forbidden)
    records["naive"] = nrec
    m_files, mrec = mutation.generate(task, sources, JS, ws / "mut", imp,
                                      tag=f"{name}_mut", n_mutants=draws, target=target,
                                      forbidden=forbidden)
    records["mutation"] = mrec

    # The matched control, asserted rather than trusted. Both arms report what they were
    # ASKED for (not what the model happened to produce), so this fires if a future caller
    # gives the two arms different counts, or if an arm stops honouring the count it was
    # given -- e.g. `naive.generate` reporting n_draws=6 while ARM M was asked for 3.
    n_n, n_m = nrec["n_draws"], mrec["n_mutants_requested"]
    assert n_n == n_m == draws, (
        f"{name}: the arms were not given the same draw count (ARM N n_draws={n_n}, "
        f"ARM M n_mutants={n_m}, requested {draws}) -- the matched control is broken")

    # 2b. MATCHED TEST COUNT (user-directed, 2026-09-15). ARM M keeps only killing tests
    # and ARM N keeps green ones, so their kept counts diverge and a raw comparison also
    # measures suite SIZE. Each arm's contributors are truncated to min(kept_M, kept_N),
    # in order, and re-assembled through the same shared `assemble` -- here, where both
    # arms' results are in scope, so neither arm's own code knows it is being matched and
    # neither can adapt to it. Both variants are scored and reported: the raw arms under
    # their own names, the truncated ones as `*_matched`, which is the headline.
    n_match = min(len(nrec.get("contributors") or []), len(mrec.get("contributors") or []))
    recs = {"naive": nrec, "mutation": mrec}
    matched = {}
    for arm, rec in recs.items():
        contributors = (rec.get("contributors") or [])[:n_match]
        files, kept, res, meta = assemble(contributors, task, JS, ws / f"matched_{arm}",
                                          tag=f"matched_{arm}")
        rec["n_kept_raw"] = len(rec.get("contributors") or [])
        rec["n_kept_matched"] = len(kept)
        # The count of test CASES the suite actually ran on P0 -- the size of the thing
        # that gets executed, which is not the contributor count (one contributor may hold
        # several cases).
        rec["n_cases_p0_raw"] = (rec.get("on_p0") or {}).get("n_cases", 0)
        rec["n_cases_p0_matched"] = len(res.cases) if res else 0
        rec["matched"] = {"n_kept": len(kept), "n_target": n_match,
                          "n_cases_p0": rec["n_cases_p0_matched"], **meta}
        matched[arm] = (files, res.passed if res else [])
    log(f"{name}: kept raw N={nrec['n_kept_raw']} M={mrec['n_kept_raw']}; "
        f"matched to {n_match} -> N={nrec['n_kept_matched']} M={mrec['n_kept_matched']} "
        f"(cases on P0: raw N={nrec['n_cases_p0_raw']} M={mrec['n_cases_p0_raw']}, "
        f"matched N={nrec['n_cases_p0_matched']} M={mrec['n_cases_p0_matched']})")

    # 2c. MATCHED CASE COUNT (user-directed, 2026-09-15). Matching CONTRIBUTORS is not
    # enough. Measured on analytics_dashboard: ARM N kept one contributor holding 39 leaf
    # cases and ARM M kept one holding 1, so the "matched" pair still differed 39x in the
    # number of tests that actually ran -- the same confound as the budget asymmetry, just
    # pointing the other way. The unit that matters is the LEAF CASE, so each arm's
    # contributor-matched composite is additionally capped to the first
    # k = min(cases_M, cases_N) of its admissible-on-P0 cases.
    #
    # The cap SKIP-MASKS the surplus instead of cutting it (`js_split.cap_cases`), so
    # imports, `beforeEach` and shared fixture constants survive for the cases that stay
    # live. Masking can still break a neighbour that depended on a masked case, so the
    # capped suite is RE-RUN on P0 and its admissible set is taken from that run -- never
    # inherited from the uncapped one. An arm that comes back below k, or whose suite
    # `cap_cases` refuses to parse, falls back to its uncapped composite with the reason
    # recorded: a mismatched pair that SAYS it is mismatched is honest, a silently
    # mismatched one is not.
    cases_by_arm = {arm: len(adm) for arm, (_, adm) in matched.items()}
    k_cases = min(cases_by_arm.values())
    matched_skipped = None
    if not k_cases:
        # Capping both arms to zero cases would convert a ONE-ARM failure into a NO-DATA
        # fixture: travel_itinerary_planner (ARM N kept nothing, ARM M kept one case) came
        # out as `matched to 0 -> N=0 M=0`, destroying the surviving arm's real result too.
        # The matched pair simply does not exist for such a fixture, so it is skipped and
        # said so, and the raw arms are still scored.
        empty = [ARM_NAMES[a] for a in sorted(cases_by_arm) if not cases_by_arm[a]]
        matched_skipped = f"{' and '.join(empty)} kept nothing"
        log(f"{name}: matched pair SKIPPED -- {matched_skipped} (admissible cases on P0: "
            f"N={cases_by_arm['naive']} M={cases_by_arm['mutation']}); the raw arms are "
            f"still scored, and no *_matched arm is")
        for rec in recs.values():
            rec["matched"]["cap"] = {"skipped": matched_skipped}
        matched = {}
    else:
        for arm, (files, adm) in sorted(matched.items()):
            capped, n_kept = js_split.cap_cases_files(files, k_cases)
            why = None
            if capped is None:
                why = "cap_cases refused (unparseable call form)"
            elif n_kept < k_cases:
                why = f"only {n_kept} leaf case(s) are parseable, need {k_cases}"
            else:
                res = JS.run_suites(JS.stage(name, ws / f"capped_{arm}"), capped,
                                    f"capped_{arm}")
                new_adm = res.passed
                if len(new_adm) < k_cases:
                    why = (f"capping to {k_cases} left only {len(new_adm)} admissible "
                           f"case(s) on P0 (status {res.status}) -- a masked case was "
                           f"load-bearing for a neighbour")
                else:
                    matched[arm] = (capped, new_adm)
                    recs[arm]["n_cases_p0_matched"] = len(new_adm)
                    recs[arm]["matched"]["n_cases_p0"] = len(new_adm)
                    recs[arm]["matched"]["cap"] = {"n_cases": k_cases, "capped": True}
            if why:
                log(f"{name}: {ARM_NAMES[arm]} NOT capped to {k_cases} case(s) -- {why}; "
                    f"falling back to its uncapped matched suite ({len(adm)} case(s)), so "
                    f"this fixture's matched pair is NOT case-matched")
                recs[arm]["matched"]["cap"] = {"n_cases": k_cases, "capped": False,
                                               "fallback": why}
        log(f"{name}: matched cases -> k={k_cases} from N={cases_by_arm['naive']} "
            f"M={cases_by_arm['mutation']}; scored N={len(matched['naive'][1])} "
            f"M={len(matched['mutation'][1])}")

    # An arm with an empty admissible set is OMITTED, not registered empty. `score_task`
    # refuses such an arm outright, because a suite that can fire no behavioural rejection
    # silently ACCEPTS every candidate -- and that refusal would discard this task's rows
    # and every paid call already spent on its candidates. An arm keeping nothing is a
    # legitimate, informative outcome (no draw survived its gate), not a reason to crash
    # the task and lose the other arm's results with it. `records[arm]["n_kept"]` says why.
    def register(arm, files, adm, why):
        if files and adm:
            # `suite_files` and not `list(...)`: a one-file suite handed over as a bare
            # string would otherwise be exploded into a list of single CHARACTERS.
            suites[arm], admissible[arm] = suite_files(files), list(adm)
        else:
            log(f"{name}: {why} produced no scorable suite (n_admissible="
                f"{len(adm or [])}) -- omitted from scoring")

    register("naive", n_files, nrec.get("admissible"), "ARM N")
    register("mutation", m_files, mrec.get("admissible"), "ARM M")
    # `matched` is emptied above when the pair is skipped, so neither `*_matched` arm is
    # registered -- not even the arm that did keep cases. Half a matched pair is not a
    # matched comparison, and scoring it alone would put an arm's suite up against nothing.
    for arm in sorted(matched):
        register(f"{arm}_matched", *matched[arm], f"{ARM_NAMES[arm]} (matched count)")

    # 3. Freeze. Candidates must not exist yet. ARM A -- the whole holdout, i.e. the
    # oracle itself -- is frozen alongside the generated arms so it passes through the
    # same hash gate they do, even though its verdicts are definitional.
    to_freeze = dict(suites, existing=oracle["oracle_text"])
    hashes = freeze(task, to_freeze)
    log(f"{name}: froze {list(hashes)}")

    # 4. Candidates. Arm N is the primary BAD population, so this loop runs until it has
    # enough of BOTH labels rather than stopping at n_good: a run that collected only GOOD
    # candidates has nothing to measure detection against.
    # THE FIXTURE MUST YIELD BOTH LABELS. It draws until it holds at least one GOOD AND at
    # least one BAD (and then until the n_good/n_bad quotas are met), or until the
    # temperature sweep is exhausted. A single-labelled fixture cannot measure the
    # comparison at all -- GOOD-only gives every arm BRR n/a, BAD-only gives every arm GAR
    # n/a -- and either silently flatters whichever arm rejects more, so the outcome is
    # recorded and the fixture is excluded from the headline table.
    candidates, good_sources = {}, None
    p0_hash = sources_hash(sources)
    draws_used = 0
    for i, (model, temp, budget) in enumerate(CANDIDATE_SWEEP):
        n_g = sum(1 for c, _ in candidates.values() if c.label == "GOOD")
        n_b = sum(1 for c, _ in candidates.values()
                  if c.label == "BAD" and c.bad_arm == "N")
        # Both floors first, then the quotas. With n_good/n_bad >= 1 the quota check
        # already implies the floors; the floors are written out anyway so a pilot run with
        # n_bad=0 cannot stop at a GOOD-only fixture.
        if n_g >= 1 and n_b >= 1 and n_g >= n_good and n_b >= n_bad:
            break
        draws_used = i + 1
        cand = propose_candidate(task, sources, target, model, temp, f"{name}_c{i}",
                                 budget=budget)
        if not cand:
            continue
        # NEVER admit P0 itself as a GOOD candidate. It trivially passes the oracle, and
        # every arm's suite is already gated green-on-P0, so an identity candidate hands
        # every arm GAR 100% and makes the GOOD side of the comparison vacuous. A model
        # that echoes its input back unchanged produces exactly this, and the label step
        # would happily call it GOOD.
        if sources_hash(cand["sources"]) == p0_hash:
            log(f"{name}: draw {i} returned P0 unchanged -- discarded (an identity "
                f"candidate would give every arm GAR 100% for free)")
            continue
        lbl, why = label(JS, task, cand["sources"], oracle["oracle_text"], ws / f"c{i}",
                         admissible=oracle["admissible_oracle"])
        if lbl == "EXCLUDE":
            continue
        if lbl == "GOOD":
            cid = f"good_{sum(1 for c, _ in candidates.values() if c.label == 'GOOD') + 1:02d}"
            good_sources = good_sources or cand["sources"]
        else:
            cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        used_model, gen_params = _candidate_provenance(cand)
        candidates[cid] = (Candidate(cid, name, lbl, None if lbl == "GOOD" else "N",
                                     None, used_model, used_model, gen_params, "ok", why,
                                     candidate_commit=sources_hash(cand["sources"])),
                           cand["sources"])
    n_harvested_good = sum(1 for c, _ in candidates.values() if c.label == "GOOD")
    n_harvested_bad = sum(1 for c, _ in candidates.values() if c.label == "BAD")
    log(f"{name}: sweep used {draws_used}/{len(CANDIDATE_SWEEP)} draws -> "
        f"{n_harvested_good} GOOD, {n_harvested_bad} BAD (harvested, arm N)")

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
    #
    # Arm S is also the FALLBACK that guarantees the BAD side exists: when the sweep ends
    # with no harvested BAD, a seeded fault in a verified-GOOD migration is the only honest
    # way to obtain one, and it is admitted only if the oracle confirms it really fails.
    # It stays tagged bad_arm="S" and is never pooled with harvested BADs, so the report
    # can say how much of the BAD side had to be manufactured. (There is no symmetric
    # fallback for GOOD: a GOOD candidate cannot be manufactured, only drawn.)
    if good_sources is None:
        seeded = []
        log(f"{name}: no verified-GOOD migration -- Arm S skipped (seeding into P0 would "
            f"score faults in the pre-migration program as a migration, and ARM M's tests "
            f"were built against that same P0)")
    else:
        # Arm S is OVER-REQUESTED when it has to carry the BAD side alone. `seed_faults`
        # returns n applied faults, but only a fraction of those are BAD once the oracle
        # rules on them -- the rest are equivalent, landing where the tests do not look
        # (~8% admitted on complex subjects). Asking for exactly n_bad therefore yields
        # ~zero, and the fixture comes back GOOD-only: every arm gets BRR n/a and the
        # comparison measures nothing. When the sweep DID harvest a BAD, Arm S is back to
        # its diagnostic role and n_bad is the right ask.
        n_seed = n_bad if n_harvested_bad else max(6 * n_bad, 12)
        if not n_harvested_bad:
            log(f"{name}: the sweep produced no BAD candidate -- Arm S must carry the BAD "
                f"side of this fixture; seeding {n_seed} faults for {n_bad} wanted")
        seeded = seed_faults(task, good_sources, n_seed, f"{name}_seed")
    seed_model, seed_params = _seed_provenance()
    n_seeded_bad = 0
    for j, f in enumerate(seeded):
        lbl, why = label(JS, task, f["sources"], oracle["oracle_text"], ws / f"s{j}",
                         admissible=oracle["admissible_oracle"])
        if lbl != "BAD":
            continue          # equivalent fault, or it broke the build: excluded (§3.5)
        cid = f"bad_{sum(1 for c, _ in candidates.values() if c.label == 'BAD') + 1:02d}"
        n_seeded_bad += 1
        candidates[cid] = (Candidate(cid, name, "BAD", "S", f["fault_category"],
                                     seed_model, seed_model, seed_params, "ok", why,
                                     candidate_commit=sources_hash(f["sources"])),
                           f["sources"])
    log(f"{name}: BAD candidates -- {n_harvested_bad} harvested (arm N), "
        f"{n_seeded_bad} seeded (arm S) from {len(seeded)} seeding attempt(s)")

    # Is this fixture usable for the comparison at all? Computed over EVERY candidate,
    # harvested and seeded alike, because a seeded BAD gives the fixture a real BAD side.
    labels = {c.label for c, _ in candidates.values()}
    single_label = next(iter(labels)) if len(labels) == 1 else None
    if single_label:
        missing = "BRR" if single_label == "GOOD" else "GAR"
        log(f"{name}: UNUSABLE for the comparison -- every candidate is {single_label}, "
            f"so {missing} is n/a for EVERY arm. Recorded as single_label and excluded "
            f"from the headline table (it is still listed there).")

    for cid, (cand, src) in candidates.items():
        d = cand.dir(task)
        d.mkdir(parents=True, exist_ok=True)
        (d / "sources.json").write_text(json.dumps(src, indent=2))

    write_metadata(task, [c for c, _ in candidates.values()],
                   extra={"suite_hashes": hashes, "oracle": {
                       "reason": oracle["reason"],
                       "n_admissible": len(oracle["admissible_oracle"]),
                       # Recorded so a reader of the artifact cannot mistake ARM A's
                       # perfect figures for a measurement (spec §4.4).
                       "arm_a_is_the_oracle": True},
                       # A fixture with only one label measures nothing; recorded here so
                       # the artifact says so even if nobody reads the log.
                       "single_label": single_label,
                       # Why this fixture has no matched pair, if it has none (None when
                       # it does). Recorded so the artifact distinguishes "the matched
                       # comparison does not exist here" from "both matched arms scored
                       # zero", which is what zeroing both arms used to look like.
                       "matched_skipped": matched_skipped,
                       "matched_cases": {"k": k_cases, "per_arm": cases_by_arm},
                       "candidate_routes": {"harvested_good": n_harvested_good,
                                            "harvested_bad": n_harvested_bad,
                                            "seeded_bad": n_seeded_bad,
                                            "sweep_draws_used": draws_used,
                                            "sweep_draws_available":
                                                len(CANDIDATE_SWEEP)},
                       "draws": draws,
                       # `contributors` holds suite TEXT, not a record: excluded for the
                       # same reason `suite` is.
                       "arms": {k: {x: v[x] for x in v
                                    if x not in ("suite", "contributors")}
                                for k, v in records.items()}})

    rows = score_task(task, candidates, suites, admissible, JS, ws / "score",
                      existing_text=oracle["oracle_text"],
                      existing_admissible=oracle["admissible_oracle"])
    # ARM C is exempt from the freeze: it is handed the GOOD->BAD diff by construction,
    # which is exactly why it is a ceiling and not a technique.
    if good_sources:
        rows += ceiling.score_ceiling(task, candidates, JS, ws / "ceiling", good_sources,
                                      imp)
    else:
        log(f"{name}: no verified-GOOD candidate, ARM C skipped")
    # Carried on every row so `score.report` can exclude this task from the headline table
    # without the single-label finding having to be plumbed through rows.json separately
    # (and without it being lost when rows from several runs are concatenated).
    for r in rows:
        r["single_label"] = single_label
        # Carried per row for the same reason as `single_label`: `score.report` must be able
        # to say which fixtures have no matched comparison even when rows from several runs
        # are concatenated and no metadata.json is at hand.
        r["matched_skipped"] = matched_skipped
    (task.root / "rows.json").write_text(json.dumps(rows, indent=2))
    log(f"{name}: {len(candidates)} candidates, {len(rows)} rows")
    return rows
