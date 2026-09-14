# Relational mutation testing (C ↔ Rust)

Can an LLM write a C test that kills a C mutant, and does that test still discriminate
after being transpiled to Rust?

```
C source P ──1──▶ P′ ──2──▶ C test T ──3──▶ RT
                            (T ✓P, ✗P′)      │
                                             ▼
      reference Rust impl M:              RT must ✓ M  and  ✗ M′
                                                            ▲
                          the same fault, ported from prose ─┘
```

The whole study reduces to one question per mutant: **does RT pass M and fail M′?**
Everything upstream exists to produce a well-formed instance of it.

## Running it

```bash
.venv/bin/python run.py select                # choose and freeze the project set
.venv/bin/python run.py run --mutants 10      # run the funnel
.venv/bin/python run.py report                # funnel table + failure taxonomy

.venv/bin/python run.py naive --run <run>      # the naive baseline, on that run's mutants
.venv/bin/python run.py rustnative --run <run> # the rust-native ceiling, same mutants
.venv/bin/python run.py naive-report --run <run>
```

Projects and mutants both fan out (`--project-workers`, `--mutant-workers`), capped at
16 in-flight API requests (`RMT_MAX_INFLIGHT`).

### Resuming

Kill a run at any point and re-issue the same command. Nothing is recomputed:

| level | artifact | what it saves |
|---|---|---|
| project | `<project>/mutants.json` | the whole project |
| mutant | `<project>/mutants/<id>/verdicts.json` | that mutant's gcc + cargo work |
| stage 1 | `<project>/stage1.json` | re-proposing mutants and re-running the compile gate |
| call | `results/.llm_cache/` | every LLM call that already returned |

Interrupting a 100 s `geofence` run and resuming it takes 0.7 s. `--redo` ignores all of
it and runs clean.

## The stages

| | module | what it does | gate |
|---|---|---|---|
| 1 | `stage1_mutate.py` | LLM proposes mutants as verbatim snippet replacements, each with a **language-agnostic `semantic_change`** description | applies uniquely · P′ compiles · text differs |
| 2 | `stage2_testgen.py` | LLM writes a C test reaching the fault through the public API | **executed**: T exits 0 on P and non-zero on P′ |
| 3 | `stage3_transpile.py` | LLM re-expresses T in Rust against the **interfaces only** | compiles; repair is compile-errors-only, assertions may not be weakened |
| 4 | `stage4_relate.py` | a second LLM ports the *described* fault into the reference Rust, giving M′ | RT on M, RT on M′, plus the RBench GT suite on M′ as a control |

Stage 2's gate is also the value grounding: an asserted literal survives only if the
original C library actually produces it. Stage 4 never shows the C diff to the model —
the Rust code is an independent re-implementation, so the only transferable thing is the
described behaviour.

## Outcomes

| | meaning |
|---|---|
| `KILLED` | the C-derived test discriminates M from M′ — full success |
| `F1_*` | no killing C test: the LLM could not distinguish P from P′ |
| `F2_transpile_no_compile` | RT never compiled against the interfaces |
| `F3_unsound_on_correct_rust` | RT fails *correct* Rust — transpilation changed the meaning |
| `F4_not_discriminating` | RT passes both M and M′ — discriminating power lost in translation |
| `F5_*` | the fault has no Rust counterpart, or an infrastructure error |

## The naive baseline

The funnel's cost is entirely in the mutant: it exists to make Stage 2 write a test that
provably discriminates. So the control worth having is the version with the mutant taken
out. `baseline_naive.py` hands one model the **C source and the Rust interfaces** — no
mutant, no diff, no C test — and asks for a Rust test suite, one per project. It is then
scored against the *same* `M'` set the funnel built, replayed from the recorded Stage 4b
edits, so RT, the naive suite and the RBench ground-truth suite share a denominator.

Two allowances keep it fair rather than flattering:

| | |
|---|---|
| compile repair | the same as Stage 3's — build errors only, assertions may not be weakened |
| pruning to M | only cases that pass the assumed-correct M can score a mutant; a case that already failed M would "kill" every M' |

Pruning mirrors what RT gets implicitly: RT reaches the M' comparison only where it passed
M, and Stage 4 charges the rest as F3. `naive-report` writes
`results/<run>/NAIVE_BASELINE.md`.

## The rust-native arm (the ceiling)

The naive arm removes the mutant; this one removes **C**. For each of the same `M'`,
`baseline_rustnative.py` shows one model the reference implementation `M` and the exact
`M → M'` unified diff — reconstructed by applying the recorded Stage 4 edits, so it is the
same mutant the other three arms were scored on, byte for byte — and asks for one Rust
test that passes `M` and fails `M'`. Same gates in the same order as RT: compile repair
(build errors only), soundness on `M`, then the kill check on `M'`, with the gate's
verdict fed back for up to `--retries` attempts.

On `full100` it kills **768/773 (99.4%)** against the funnel's 97% — so the C detour costs
about two points, and four of the five survivors are mutants the arm itself calls
unreachable through the public API
([write-up](updates/2026-09-04-rustnative-ceiling.md)).

It knows the fault exactly, in the language the test is written in, with no translation
step to lose it in. So it is an **upper bound, not a technique**: whatever it cannot kill
is a mutant that is hard to reach through the public API, not one the pipeline mislaid, and
`1 - rn_kill_rate` is the floor under every other arm's miss rate. Its own outcome
breakdown (`F_UNSOUND_ON_M`, `F_NOT_DISCRIMINATING`, `UNREACHABLE`, …) says which.

```bash
.venv/bin/python run.py rustnative --all --run full100 --max-mutants 2   # cheap pilot
.venv/bin/python run.py rustnative --all --run full100                   # the full arm
.venv/bin/python run.py naive-report --run full100   # folds it in as a fourth column
```

It pays per *mutant*, like the funnel, not per project like the naive arm — budget for it
accordingly, and start with `--max-mutants`. Results are cached per mutant
(`<project>/rustnative/<id>.json`), so an interrupted run resumes for free.

## What plays the role of correct Rust

RBench ships only interfaces and a ground-truth test suite — no reference
implementation. We take the codex transpilation
(`datasets/codex_expts/all/initial_codex_output`) as correct wherever it passes the
ground-truth suite. Phase 0 measured that: 94/100 projects, all of which the C harness can
drive; 86 also clear the selection filters (enough interface functions and enough mutable
C to host ten mutants).

## Layout

```
run.py              the entrypoint
rmt/
  llm.py            Opus 5, adaptive thinking, disk-cached, concurrency-capped
  cbench.py         C build/run harness (gcc-uniform, throwaway copies)
  rbench.py         RBench interfaces, the reference impl, cargo test injection
  stage1..stage4    the funnel
  baseline_naive    the mutant-free control arm
  baseline_rustnative  the C-free oracle arm: a test per M' from the M->M' diff
  baseline_report   RT vs naive vs GT vs rust-native, on one denominator
  pilot.py          project selection
  driver.py         parallel orchestration
  report.py         funnel table and taxonomy
  prompts/*.md      every prompt, versioned separately from the code
phase0/             feasibility probes (panel size, C harness)
results/            pilot.json, <run>/<project>/mutants.json, .llm_cache/
```

Design spec:
[`docs/superpowers/specs/2026-09-02-relational-mutation-testing-baseline-design.md`](../docs/superpowers/specs/2026-09-02-relational-mutation-testing-baseline-design.md)
· Phase 0 results: [`results/phase0/REPORT.md`](results/phase0/REPORT.md)

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install anthropic
```

The API key is read from `/nas/secrets.env` (`anthropic_key`) unless
`ANTHROPIC_API_KEY` is set. The client pins `https://api.anthropic.com` because the
shell exports `ANTHROPIC_BASE_URL` for a proxy that rejects that key.
