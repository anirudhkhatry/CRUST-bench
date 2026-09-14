# Runbook — the 100×10 relational runs

Live state for the full-panel experiment, so that work survives a context reset. Update
it when a phase completes; delete it when both runs are reported.

## Where things stand (2026-09-03)

| | |
|---|---|
| Panel | `results/panel.json` — all 100 harness-drivable projects, **94 GT-green** |
| Not GT-green (M unverified, reported as a separate stratum) | `Genetic_neural_network_for_simple_control`, `bostree`, `hamta`, `impcheck`, `libfor`, `libwecan` |
| Claude run | `results/full100/`, log `results/full100.log` — **in flight** |
| GPT run | `results/full100_gpt54mini/` — **not started**, launch after the Claude run |
| Naive baseline (Claude) | `results/full100/*/naive.json` — **done** (2026-09-04), report `results/full100/NAIVE_BASELINE.md`, write-up `updates/2026-09-04-naive-baseline.md` |
| Rust-native ceiling | `results/full100/*/rustnative.json` — **done** (2026-09-04), 768/773 = 99.4%, $127.31. Folded into `NAIVE_BASELINE.md`; write-up `updates/2026-09-04-rustnative-ceiling.md` |

## Commands

```bash
# freeze the panel (already done; rerunning is idempotent)
.venv/bin/python run.py select --all

# the Claude run, as launched
CARGO_BUILD_JOBS=2 RMT_MAX_INFLIGHT=32 nohup nice -n 10 .venv/bin/python run.py run \
  --all --run full100 --mutants 10 --project-workers 6 --mutant-workers 4 \
  >> results/full100.log 2>&1 &
./scripts/janitor.sh &                    # reaps /tmp/rmt_cargo, logs memory

# the same run again with the smaller model (only RMT_MODEL differs)
RMT_MODEL=gpt-5.4-mini CARGO_BUILD_JOBS=2 RMT_MAX_INFLIGHT=32 nohup nice -n 10 \
  .venv/bin/python run.py run --all --run full100_gpt54mini --mutants 10 \
  --project-workers 6 --mutant-workers 4 >> results/full100_gpt54mini.log 2>&1 &
./scripts/janitor.sh results/full100_gpt54mini.guard.log &

.venv/bin/python run.py report --run full100     # writes results/<run>/REPORT.md

# the naive baseline, scored on that run's M' set (needs the funnel run to exist first)
CARGO_BUILD_JOBS=2 RMT_MAX_INFLIGHT=16 nohup nice -n 10 .venv/bin/python run.py naive \
  --all --run full100 --project-workers 6 --mutant-workers 4 \
  >> results/naive_full100.log 2>&1 &
./scripts/janitor.sh results/naive_full100.guard.log "run.py naive --all" &
.venv/bin/python run.py naive-report --run full100   # writes NAIVE_BASELINE.md

# the rust-native ceiling, on the same M' set. It pays per mutant, so pilot it first.
CARGO_BUILD_JOBS=2 RMT_MAX_INFLIGHT=16 .venv/bin/python run.py rustnative \
  --all --run full100 --max-mutants 2 --project-workers 3 --mutant-workers 2
CARGO_BUILD_JOBS=2 RMT_MAX_INFLIGHT=16 nohup nice -n 10 .venv/bin/python run.py rustnative \
  --all --run full100 --project-workers 3 --mutant-workers 2 \
  >> results/rustnative_full100.log 2>&1 &
./scripts/janitor.sh results/rustnative_full100.guard.log "run.py rustnative --all" &
```

Re-issuing a `run` command **resumes**: finished projects are skipped, unfinished mutants
replay their model calls free from `results/.llm_cache`. So an interruption costs wall
clock, never money, and the right response to a crash is to run the same line again.

## Remaining steps

`scripts/chain.sh` now drives steps 1–2 and 4 unattended (launched 2026-09-03, log
`results/chain.log`). It waits out the in-flight Claude pass, does the Claude resume pass,
writes the Claude report, then runs both GPT passes and its report. `pgrep -af chain.sh`
says whether it is still going; `grep chain results/chain.log` is its progress trace.

1. ~~Wait for `results/full100.log` to print its `done in …s` line.~~ (chain)
2. ~~**Resume pass**~~ (chain). Projects that hit `PROJECT ERROR` left no `mutants.json`,
   so `driver._finished` treats them as unfinished and only they rerun, now with the
   enlarged Stage 1 budget. `morton` is one; check
   `grep "PROJECT ERROR" results/full100.log` for the rest.
3. **Still manual.** After each `REPORT.md` lands, read every non-`KILLED` mutant's
   `verdicts.json` and write the failure analysis: which F-class, and whether the cause is
   the mutant, the transpilation, or the harness.
4. ~~Launch the `gpt-5.4-mini` run~~ (chain). Then put the two models side by side.

The chain restarts safely: rerunning `./scripts/chain.sh` skips whatever already finished.

## Resource guards (both runs)

`CARGO_BUILD_JOBS=2` is the one that matters — cargo otherwise takes a rustc job per core
(56 here) per build, and 24 builds in flight could have exhausted memory.

**The container *is* cgroup-capped: `/sys/fs/cgroup/memory.max` = 64 GiB.** An earlier
version of this note claimed otherwise, and `scripts/janitor.sh` was written against
`/proc/meminfo`, which reports the *host's* 251 GB. So the guard logged `avail=118G` while
the run sat against a 64 GiB ceiling — it could not have warned. The janitor now reads
`memory.current` vs `memory.max` and also prints the cgroup's `oom_kill` counter. The
cgroup files are root-owned and not writable from here, so `memory.high` cannot be set as
a throttle; concurrency remains the only enforceable limit. Worst case is
`project-workers x mutant-workers x CARGO_BUILD_JOBS` rustc processes — at 6x4x2 that is
48 rustc against 64 GiB, which is the configuration that crashed. Keep the product at or
below ~12 (e.g. `--project-workers 3 --mutant-workers 2` with `CARGO_BUILD_JOBS=2`).
`scripts/janitor.sh` reaps `CARGO_TARGET_DIR`s untouched for 40 minutes (nothing else ever
deletes them; they reached 7 GB in the first 20 minutes) and logs headroom every 5
minutes. It only logs on low memory: killing a build would charge a mutant an F2 it did
not earn, and pausing the driver would make an in-flight subprocess timeout fire
spuriously on resume.

## Two bugs of the same shape, both fixed

A truncated reply used to be fatal rather than retryable. Stage 4b (`stage4_relate.py`)
and Stage 1 (`stage1_mutate.py`) both now ask for a larger budget and retry once on an
unparseable reply. Stage 1's version was the worse of the two: its single call carries all
ten mutants, so a truncation took the entire project down.

Note that raising a stage's `max_tokens` changes the cache key, so it re-bills every
not-yet-made call at that stage. That is why the Stage 1 fix was landed *without*
restarting the run in flight — the errored projects pick it up in the resume pass instead.
