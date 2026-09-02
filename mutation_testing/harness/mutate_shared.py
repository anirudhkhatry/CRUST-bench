"""Phase 3: build each mutant once, test it against all four suites.

Phase 2 (``split_crates.py``) gives every project a workspace where the mutated source
sits in ``core/`` and each setting's tests sit in their own crate. It still runs
cargo-mutants four times, so ``core`` is rebuilt once per mutant *per setting*. This
module drives the mutation loop directly instead: apply a mutant to ``core``, build it
ONCE, then run the four test crates against that one build.

Two reasons, one of which is correctness rather than cost:

* **Viability becomes shared by construction.** Under phase 2 each setting discovers
  unviability by failing to build ``cargo test --package=tests-<X>`` -- so a shared
  denominator held empirically but was not guaranteed. Here ONE build invocation spanning
  the mutated crate and all four test crates decides it, and every setting is handed that
  same verdict. (It has to span the test crates: 4 of 34 ``aes128_SIMD`` mutants compile
  as a library and only fail once a dependent crate const-evaluates the mutated
  expression.)
* **Unviable mutants cost one build instead of four builds plus four test attempts.**
  On ``aes128_SIMD`` 15 of 34 mutants in a shard are unviable, so this is not a marginal
  case.

Mutants are applied by splicing ``replacement`` over the byte range named by ``span``
from ``cargo mutants --list --json``, NOT by patching. cargo-mutants' emitted ``diff``
carries a pseudo-header (``+++ replace pad_buffer with ()``) that no patch tool accepts,
whereas the span is exact: verified on all 398 ``aes128_SIMD`` mutants that the spliced
file reproduces every added line of the corresponding diff.

Editing in place is safe here, and specifically does NOT reintroduce the stale-artifact
bug that forced phase 2 to partition its target dirs per setting. That bug came from
cargo-mutants copying the workspace to a fresh path per run, so a copy's mtimes could
land behind a cached artifact's and cargo would call it fresh. Here the workspace path is
fixed and a write moves the mtime forward, so cargo's fingerprinting works: verified by
mutating -> testing -> restoring -> testing twice over, with every suite failing while
mutated and passing while clean.
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path

# cargo-mutants' own vocabulary, so the outcomes.json written here is readable by
# intersection_scores.py and run_mutation._summarize_outcomes unchanged.
CAUGHT, MISSED, TIMEOUT, UNVIABLE = "CaughtMutant", "MissedMutant", "Timeout", "Unviable"

# Same excludes as run_mutation.MUTATE_EXCLUDES: never mutate test bins or interfaces.
EXCLUDES = ["--exclude", "**/src/bin/**", "--exclude", "**/src/interfaces/**"]

# Keys of a --list --json entry that also appear in outcomes.json's scenario.Mutant.
_MUTANT_KEYS = ("name", "package", "file", "function", "span", "replacement", "genre")


def _offset(lines, pos):
    """Byte offset of a cargo-mutants {line, column} position (both 1-based)."""
    return sum(len(l) for l in lines[: pos["line"] - 1]) + (pos["column"] - 1)


def splice(text, span, replacement):
    """Return ``text`` with the region named by ``span`` replaced."""
    lines = text.splitlines(keepends=True)
    return text[: _offset(lines, span["start"])] + replacement + text[_offset(lines, span["end"]):]


def list_mutants(ws, package, env=None, timeout=900):
    """All mutants cargo-mutants would generate for ``package``, in --no-shuffle order."""
    cmd = ["cargo", "mutants", "--list", "--json", "--package", package] + EXCLUDES
    r = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cargo mutants --list failed in {ws}: {r.stderr[-2000:]}")
    return json.loads(r.stdout)


def shard_mutants(mutants, shard):
    """Take one shard of the mutant list, as ``k/n``, by stride over --list order.

    Deliberately NOT cargo-mutants' partition: it shards by hashing the mutant name, so
    neither ``[k-1::n]`` nor a contiguous block reproduces it (checked against a stored
    ``--shard 1/12`` run). That only matters for comparing against phase-2 numbers
    mutant-for-mutant -- pass ``only_names`` for that. Within a phase-3 run all four
    settings score the identical subset, which is the property the study needs.
    """
    if not shard:
        return mutants
    k, n = (int(x) for x in shard.split("/"))
    return mutants[k - 1:: n]


def _run(cmd, cwd, env, timeout):
    """(returncode, seconds, output); returncode None on timeout."""
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.monotonic() - t0, "Timeout"
    return r.returncode, time.monotonic() - t0, r.stdout + "\n" + r.stderr


def _compile_error(output):
    """True if a cargo run failed to BUILD rather than failing its tests."""
    return "could not compile" in output or "error[E" in output


def baseline(ws, test_crates, env, timeout=1800):
    """Build and run every test crate unmutated.

    Returns ``{setting: {"ok", "test_s", "output"}}``. This is the same gate phase 2
    needed (cargo-mutants baselines the package it mutates, not the one named by
    ``--test-package``), and here there is no cargo-mutants baseline at all, so it is the
    only thing standing between a red suite and a 100% score.

    ``--no-run`` first so the reported time is test time, not build time, and so the
    target dir is warm before the mutation loop starts.
    """
    out = {}
    for setting, crate in sorted(test_crates.items()):
        base = ["cargo", "test", "--package", crate]
        rc, _, o = _run(base + ["--no-run"], ws, env, timeout)
        if rc != 0:
            out[setting] = {"ok": False, "test_s": 0.0, "output": o[-4000:]}
            continue
        rc, secs, o = _run(base, ws, env, timeout)
        ok = rc == 0 and "FAILED" not in o
        out[setting] = {"ok": ok, "test_s": secs, "output": "" if ok else o[-4000:]}
    return out


def run_shared_mutation(ws, package, test_crates, out_base, project_name=None,
                        target_dir=None, jobs=None, shard=None, only_names=None,
                        timeout_mult=2.0, min_timeout=20, build_timeout=900, verbose=True):
    """Mutate ``package`` once per mutant and test all of ``test_crates`` against it.

    ``test_crates`` maps setting name -> test crate name (see
    ``split_crates.TEST_CRATE_NAMES``). Writes one cargo-mutants-shaped
    ``<out_base>/<setting_dir>/mutants.out/outcomes.json`` per setting, so
    ``intersection_scores.py`` and the existing summarisers work on the result unchanged.

    Returns ``{"settings": {setting: summary}, "shared": {...}}``.
    """
    from harness.run_mutation import SETTING_DIRS  # local: run_mutation imports the whole pipeline

    ws = Path(ws)
    project_name = project_name or ws.name
    env = os.environ.copy()
    env["CARGO_INCREMENTAL"] = "0"
    if target_dir:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    if jobs:
        env["CARGO_BUILD_JOBS"] = str(jobs)

    mutants = list_mutants(ws, package, env=env)
    n_total = len(mutants)
    if only_names:
        want = set(only_names)
        mutants = [m for m in mutants if m["name"] in want]
    else:
        mutants = shard_mutants(mutants, shard)

    # Disambiguate mutants sharing a name (same file:line:col, same replacement) exactly
    # as intersection_scores.py does, so the two agree on the key.
    seen, keys = {}, []
    for m in mutants:
        seen[m["name"]] = seen.get(m["name"], 0) + 1
        keys.append(m["name"] if seen[m["name"]] == 1 else f"{m['name']}#{seen[m['name']]}")

    base = baseline(ws, test_crates, env)
    live = {s: c for s, c in test_crates.items() if base[s]["ok"]}
    for s in sorted(test_crates):
        if not base[s]["ok"] and verbose:
            print(f"  {project_name}/{s}: unmutated suite is red, excluded from this run")
    if not live:
        raise RuntimeError(f"{project_name}: no setting has a green baseline")

    # Per-setting cap from that setting's own measured suite time. The floor mirrors
    # cargo-mutants' --minimum-test-timeout so a sub-second suite still gets 20s; raise
    # ``min_timeout`` when a sub-second baseline is producing timeouts that are really
    # just slow-but-terminating mutants sitting in the denominator without being kills.
    caps = {s: max(min_timeout, math.ceil(base[s]["test_s"] * timeout_mult)) for s in live}

    # One build invocation covering the mutated crate and every live test crate.
    build_cmd = ["cargo", "test", "--no-run", "--package", package]
    for crate in sorted(live.values()):
        build_cmd += ["--package", crate]

    results = {s: [] for s in live}
    shared = {"unviable": 0, "viable": 0, "test_crate_broke": 0,
              "build_failed_other": 0, "build_s": 0.0, "test_s": 0.0}
    t_start = time.time()

    for key, m in zip(keys, mutants):
        target = ws / m["file"]
        original = target.read_text(encoding="utf-8")
        try:
            target.write_text(splice(original, m["span"], m["replacement"]), encoding="utf-8")

            # ONE build decides viability for every setting -- the whole point of phase 3 --
            # and it builds the test binaries too, so the per-setting steps below only have
            # to RUN them.
            #
            # It has to span every test crate, not just `package`. Measured on
            # aes128_SIMD: 4 of 34 mutants compile fine as a library and then break the
            # test crates, because const-evaluating the mutated expression (e.g. `*` -> `/`
            # yielding a division by zero in an array length) is deferred to the use site.
            # Building `package` alone called those 4 viable and each setting then
            # rediscovered the failure on its own -- exactly the per-setting-denominator
            # disease this plan exists to cure. Here one failure retires the mutant for all
            # four settings, so the shared denominator holds by construction.
            rc, secs, out = _run(build_cmd, ws, env, build_timeout)
            shared["build_s"] += secs
            if rc != 0:
                shared["unviable"] += 1
                if not _compile_error(out):
                    shared["build_failed_other"] += 1
                for s in live:
                    results[s].append((key, m, UNVIABLE))
                continue

            shared["viable"] += 1
            for s, crate in sorted(live.items()):
                rc, secs, out = _run(["cargo", "test", "--package", crate], ws, env, caps[s])
                shared["test_s"] += secs
                if rc is None:
                    summary = TIMEOUT
                elif rc == 0:
                    summary = MISSED
                elif _compile_error(out):
                    # Unreachable in principle: the batched build above already compiled
                    # this crate's tests. Counted rather than folded into caught, so that
                    # if it ever fires we find out instead of reporting a phantom kill.
                    shared["test_crate_broke"] += 1
                    summary = UNVIABLE
                    if verbose:
                        print(f"  {project_name}/{s}: test crate failed to build on "
                              f"{m['name']} despite a green batched build")
                else:
                    summary = CAUGHT
                results[s].append((key, m, summary))
        finally:
            target.write_text(original, encoding="utf-8")

    summaries = {}
    for s, rows in sorted(results.items()):
        # The layout intersection_scores.outcomes_path() looks in first, so phase-3 output
        # is a drop-in for phase-2 output: pass out_base=<output_dir>/_mutants_out.
        summaries[s] = _write_outcomes(
            Path(out_base) / SETTING_DIRS[s] / project_name / "mutants.out", rows, n_total,
            t_start, timeout_s=caps[s], baseline_test_s=base[s]["test_s"])

    if verbose:
        print(f"  {project_name}: {shared['viable']} viable / {shared['unviable']} unviable "
              f"of {len(mutants)}; builds {shared['build_s']:.0f}s, "
              f"tests {shared['test_s']:.0f}s")
        for s, sm in sorted(summaries.items()):
            print(f"    {s:9} killed {sm['killed']}/{sm['viable_mutants']} "
                  f"({sm['mutation_score']:.2%})")

    shared = {**shared, "mutants_run": len(mutants), "mutants_total": n_total,
              "shard": None if only_names else shard,
              "wall_s": round(time.time() - t_start, 1)}
    return {"settings": summaries, "shared": shared}


def _write_outcomes(out_dir, rows, n_total, t_start, timeout_s, baseline_test_s):
    """Write one setting's results as a cargo-mutants outcomes.json, return a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {CAUGHT: 0, MISSED: 0, TIMEOUT: 0, UNVIABLE: 0}
    outcomes = []
    for key, m, summary in rows:
        counts[summary] += 1
        mutant = {k: m[k] for k in _MUTANT_KEYS if k in m}
        outcomes.append({"scenario": {"Mutant": mutant}, "summary": summary,
                         "mutant_key": key})

    killed, survived = counts[CAUGHT], counts[MISSED]
    viable = killed + survived
    doc = {
        # Marked so a reader can tell these outcomes were produced by the shared-build
        # harness rather than by cargo-mutants itself.
        "cargo_mutants_version": "mutate_shared (phase 3)",
        "start_time": t_start, "end_time": time.time(), "success": True,
        "total_mutants": n_total, "caught": killed, "missed": survived,
        "timeout": counts[TIMEOUT], "unviable": counts[UNVIABLE],
        "timeout_s": timeout_s, "baseline_test_s": round(baseline_test_s, 3),
        "outcomes": outcomes,
    }
    (out_dir / "outcomes.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return {"killed": killed, "survived": survived, "timeout": counts[TIMEOUT],
            "unviable": counts[UNVIABLE], "viable_mutants": viable,
            "mutation_score": (killed / viable) if viable else 0.0,
            "mutants_total": n_total, "timeout_s": timeout_s,
            "baseline_test_s": round(baseline_test_s, 3), "baseline_valid": True}


def main():
    import argparse
    from harness.split_crates import TEST_CRATE_NAMES

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("workspace", help="project workspace root built by split_crates.py")
    p.add_argument("--package", required=True, help="the crate to mutate (core)")
    p.add_argument("--output", required=True,
                   help="absolute path OUTSIDE the workspace; results land in "
                        "<output>/<setting_dir>/<project>/mutants.out. Point it at a "
                        "results tree's _mutants_out/ to have intersection_scores.py "
                        "pick them up.")
    p.add_argument("--target-dir", default=None)
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--shard", default=None, help="k/n, by stride over --list order")
    p.add_argument("--timeout-multiplier", type=float, default=2.0)
    p.add_argument("--min-test-timeout", type=float, default=20,
                   help="floor on the per-mutant test timeout, in seconds (default 20, "
                        "matching cargo-mutants' --minimum-test-timeout)")
    p.add_argument("--only-names-from", default=None,
                   help="an existing outcomes.json; run exactly its mutants (for A/B)")
    p.add_argument("--settings", nargs="+", default=sorted(TEST_CRATE_NAMES))
    a = p.parse_args()

    only = None
    if a.only_names_from:
        d = json.loads(Path(a.only_names_from).read_text())
        only = [o["scenario"]["Mutant"]["name"] for o in d["outcomes"]
                if isinstance(o.get("scenario"), dict) and "Mutant" in o["scenario"]]

    r = run_shared_mutation(
        a.workspace, a.package, {s: TEST_CRATE_NAMES[s] for s in a.settings},
        out_base=Path(a.output).resolve(), target_dir=a.target_dir, jobs=a.jobs,
        shard=a.shard, only_names=only, timeout_mult=a.timeout_multiplier,
        min_timeout=a.min_test_timeout)
    print(json.dumps(r["shared"], indent=2))


if __name__ == "__main__":
    main()
