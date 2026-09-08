#!/usr/bin/env bash
# Chain the remaining Experiment 1 + 2 stages, so the pipeline finishes unattended.
#
# Stages 1-3 (results/expA_funnel.log) and the static-only candidates for the 132
# non-runnable records (results/expA_base132.log) are already running when this starts.
# It waits for both, then does everything that depends on them, in dependency order.
#
# Each step is skipped if its output already exists, so this is safe to re-run.
set -u
cd "$(dirname "$0")"

export RMT_MODEL=claude-opus-5
export RMT_MIGRATE_PROMPT=published
export RMT_MUTATE_PROMPT=mutate_mig
M=claude-opus-5

say() { echo "=== $(date '+%F %T') $* ==="; }

# The two running jobs. `pgrep -f` rather than a PID file: this script may be started
# after them, and the patterns are unique to this run.
wait_for() {
  local pat="$1" name="$2"
  while pgrep -f "$pat" > /dev/null; do sleep 60; done
  say "$name finished"
}

say "waiting for stages 1-3 over the 158 funnel records"
wait_for "run --run expA" "funnel"
say "waiting for static-only candidates over the 132 non-runnable records"
wait_for "migbaseline --run expA --panel results/panel_rest132" "baseline(132)"

# The candidates for the runnable half. Held back until now so two processes never
# materialise the same repo tree at once.
say "candidates for the 158 runnable records"
python run_pmb.py migbaseline --run expA --panel results/panel_funnel158.json \
  --allow-static --workers 3 >> results/expA_base158.log 2>&1
say "candidates done: $(ls -d results/expA/*/candidates/$M/candidate.json 2>/dev/null | wc -l) total"

# Experiment 1's answer: does T' agree with the repository's own suite?
say "validate (T' judges every candidate)"
python run_pmb.py validate --run expA --model $M --workers 3 \
  >> results/expA_validate.log 2>&1
say "validate done"

# The static oracles, free: the developer's recorded edits, and tier 0.
say "changedev (developer-diff oracle)"
python run_pmb.py changedev --run expA --model $M >> results/expA_changedev.log 2>&1

# Experiment 2: can the tests repair what they rejected?
say "guided vs blind repair"
python run_pmb.py guided --run expA --model $M --workers 2 \
  >> results/expA_guided.log 2>&1
say "ALL DONE"
