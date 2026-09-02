"""Faster drop-in for ``mutate_shared.run_shared_mutation``.

``mutate_shared`` (phase 3) got the *semantics* right -- one build per mutant, viability
shared across settings by construction -- and its verdicts are the reference this module
is validated against. What it did not do is go fast. Measured on ``ulidgen`` (122
mutants, four suites): the suites themselves run in 0.406 s, so all four settings across
all 122 mutants amount to ~3.3 minutes of actual test execution, yet a full pass takes
well over half an hour. The time goes somewhere else entirely.

It goes into ``cargo``. Phase 3 issues **five** cargo invocations per mutant:

    1 x  cargo test --no-run -p core -p tests-src -p tests-... (the batched build)
    4 x  cargo test -p tests-<setting>                         (one per suite)

That is 610 cargo process launches for ``ulidgen``, each re-walking the dependency graph
and stat-ing the tree to conclude there is nothing to rebuild. ``plan_shared_mutant_set``
already fingered this as the reason phase 3 delivered 1.5-2.1x rather than the hoped 3-4x:
"cargo's per-invocation freshness check, not the `core` build, is the dominant per-mutant
cost, and phase 3 still runs one `cargo test` per setting."

Four changes, each independently switchable so a regression can be bisected:

1. **Parallel mutants** (``workers=N``). Phase 3 is strictly serial on a machine with 56
   cores. Mutants are spliced *in place*, so concurrency needs one workspace clone and one
   target dir per worker; the clone is cheap (a few MB of sources) and the target dir is
   warmed once up front. This is the largest win and is orthogonal to the other three.

2. **Direct test-binary execution** (``runner="direct"``, the default). The batched build
   already produces the test executables, and asking for ``--message-format=json`` hands
   us their paths for free. Running them directly drops the per-mutant cargo count from
   **5 to 1**. This is strictly better than the obvious alternative of collapsing the four
   ``cargo test`` calls into one multi-``-p`` call, which would save the same overhead but
   lose per-setting timeout isolation -- exactly the thing under study here. Direct exec
   keeps every setting on its own clock.

3. **nextest** (``runner="nextest"``). Lower per-binary overhead than ``cargo test`` and,
   more usefully here, a process per test -- so a non-terminating mutant costs one *test*
   rather than one whole *suite*. Kept as an option rather than the default because it is
   an extra dependency and direct exec already removes the overhead it targets.

4. **Timeout floor** (``min_timeout``). Because these suites run in ~0.4 s, the dynamic
   ``2 x baseline`` cap is ~0.8 s and the floor is always what actually binds. It is the
   smallest of the four levers and is plumbed here for completeness.

Fidelity note for ``runner="direct"``: ``cargo test`` runs each binary with the package
directory as cwd and ``CARGO_MANIFEST_DIR`` set, and a test reading a fixture by relative
path would notice if we did not. Both are reproduced below. ``validate_against_shared``
exists to check this empirically rather than on argument.

``mutant_lister`` selects where mutants come from. The default is cargo-mutants, which is
the validated path and is byte-identical to previous runs. Passing
``list_transpile_mutants.list_transpile_mutants`` scores the transpilation-aware catalog
instead; passing a lambda that concatenates both scores them together, in which case the
``Transpile*`` genre prefix keeps the two families attributable in the analysis.
"""

import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.mutate_shared import (
    CAUGHT, MISSED, TIMEOUT, UNVIABLE,
    _compile_error, _run, _write_outcomes, baseline, list_mutants, shard_mutants, splice,
)

try:
    from tqdm import tqdm
except ImportError:  # progress is a convenience, never a dependency
    tqdm = None

# Sources only: `target` is per-worker and rebuilt, `metadata` is results not input.
_CLONE_IGNORE = shutil.ignore_patterns("target", "metadata", ".git", "*.bak")


def _pkg_name(package_id):
    """Package name out of a cargo package_id, across both id formats.

    Modern cargo emits ``path+file:///w/tests-src#0.1.0``; older/registry ids carry the
    name explicitly as ``...#tests-src@0.1.0``.
    """
    head, _, tail = package_id.partition("#")
    if "@" in tail:
        return tail.split("@")[0]
    if tail and not tail[0].isdigit():
        return tail
    return head.rstrip("/").split("/")[-1]


def _clone_workspace(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_CLONE_IGNORE, symlinks=True)
    return dst


def _build(ws, build_cmd, env, timeout):
    """Batched build. Returns (rc, seconds, output, {package: [test executables]}).

    ``rc`` is None on timeout, matching ``mutate_shared._run``. The executables come from
    the same invocation that does the building -- ``--message-format=json`` makes the
    artifact list a by-product rather than an extra cargo call.
    """
    t0 = time.monotonic()
    try:
        r = subprocess.run(build_cmd, cwd=ws, capture_output=True, text=True,
                           env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.monotonic() - t0, "Timeout", {}

    exes = {}
    for line in r.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-artifact" or not msg.get("executable"):
            continue
        # profile.test marks a binary libtest can run; it excludes the plain `bin`
        # artifacts a test crate may also produce.
        if not msg.get("profile", {}).get("test"):
            continue
        exes.setdefault(_pkg_name(msg["package_id"]), []).append(msg["executable"])
    return r.returncode, time.monotonic() - t0, r.stdout + "\n" + r.stderr, exes


def _verdict_direct(exes, pkg_dir, env, cap):
    """Run a setting's test binaries directly. Returns (verdict, seconds).

    cwd and CARGO_MANIFEST_DIR mirror what `cargo test` sets, so a test that opens a
    fixture by relative path behaves the same as it does under cargo.
    """
    e = dict(env)
    e["CARGO_MANIFEST_DIR"] = str(pkg_dir)
    total, verdict = 0.0, MISSED
    for exe in exes:
        rc, secs, _out = _run([exe], pkg_dir, e, cap)
        total += secs
        if rc is None:
            return TIMEOUT, total          # a hang ends the setting immediately
        if rc != 0:
            verdict = CAUGHT               # keep going: another binary may yet time out
    return verdict, total


def _verdict_cargo(ws, crate, env, cap, runner):
    """Reference path: one `cargo test` (or `cargo nextest run`) per setting."""
    cmd = (["cargo", "nextest", "run", "--package", crate] if runner == "nextest"
           else ["cargo", "test", "--package", crate])
    rc, secs, out = _run(cmd, ws, env, cap)
    if rc is None:
        return TIMEOUT, secs, out
    if rc == 0:
        return MISSED, secs, out
    return (UNVIABLE if _compile_error(out) else CAUGHT), secs, out


def run_fast_mutation(ws, package, test_crates, out_base, project_name=None,
                      target_dir=None, jobs=None, shard=None, only_names=None,
                      timeout_mult=2.0, min_timeout=20, build_timeout=900,
                      workers=1, runner="direct", no_debuginfo=True, verbose=True,
                      mutant_lister=None):
    """Mutate ``package`` once per mutant, test every suite, in parallel.

    Same contract as ``mutate_shared.run_shared_mutation`` -- same arguments, same
    ``outcomes.json`` layout, same ``{"settings", "shared"}`` return -- plus ``workers``
    and ``runner``. Verdicts are intended to be identical; ``validate_against_shared``
    checks that they are.
    """
    from harness.run_mutation import SETTING_DIRS  # local: run_mutation imports the whole pipeline

    ws = Path(ws).resolve()
    project_name = project_name or ws.name
    if runner not in ("direct", "cargo", "nextest"):
        raise ValueError(f"unknown runner {runner!r}")
    if runner == "nextest" and shutil.which("cargo-nextest") is None:
        raise RuntimeError("runner='nextest' needs cargo-nextest: cargo install cargo-nextest")

    base_env = os.environ.copy()
    base_env["CARGO_INCREMENTAL"] = "0"
    if jobs:
        base_env["CARGO_BUILD_JOBS"] = str(jobs)
    if no_debuginfo:
        # Measured on ulidgen: rebuild after touching core drops 19.11s -> 14.45s (1.32x).
        # Mutation testing never reads a backtrace, so the debuginfo is generated and
        # thrown away. Done via RUSTFLAGS rather than a [profile] block so no manifest is
        # edited -- one less file that can be left modified by an interrupted run.
        base_env["RUSTFLAGS"] = (base_env.get("RUSTFLAGS", "") + " -Cdebuginfo=0").strip()
    if workers > 1:
        # Each worker already saturates its own build; letting libtest fan out on top of
        # that oversubscribes the box (the failure mode that has crashed this machine).
        base_env.setdefault("RUST_TEST_THREADS", "1")

    env0 = dict(base_env)
    if target_dir:
        env0["CARGO_TARGET_DIR"] = str(Path(target_dir).resolve())

    mutants = (mutant_lister or list_mutants)(ws, package, env=env0)
    n_total = len(mutants)
    if only_names:
        want = set(only_names)
        mutants = [m for m in mutants if m["name"] in want]
    else:
        mutants = shard_mutants(mutants, shard)

    seen, keys = {}, []
    for m in mutants:
        seen[m["name"]] = seen.get(m["name"], 0) + 1
        keys.append(m["name"] if seen[m["name"]] == 1 else f"{m['name']}#{seen[m['name']]}")

    # Baseline gate, unchanged from phase 3 and still the only thing between a red suite
    # and a flattering 100%: a red suite fails on every mutant, so every mutant reads as
    # caught. Run once on the original workspace, before any clone exists.
    base = baseline(ws, test_crates, env0)
    live = {s: c for s, c in test_crates.items() if base[s]["ok"]}
    for s in sorted(test_crates):
        if not base[s]["ok"] and verbose:
            print(f"  {project_name}/{s}: unmutated suite is red, excluded from this run")
    if not live:
        raise RuntimeError(f"{project_name}: no setting has a green baseline")

    caps = {s: max(min_timeout, math.ceil(base[s]["test_s"] * timeout_mult)) for s in live}

    build_cmd = ["cargo", "test", "--no-run", "--package", package]
    for crate in sorted(live.values()):
        build_cmd += ["--package", crate]
    if runner == "direct":
        build_cmd += ["--message-format=json"]

    # ---- worker pool: one workspace clone + one target dir each ----------------------
    td_root = Path(target_dir).resolve() if target_dir else (ws.parent / f".{ws.name}__td")
    pool, clones = queue.Queue(), []
    for i in range(max(1, workers)):
        if workers > 1:
            wdir = _clone_workspace(ws, ws.parent / f".{ws.name}__w{i}")
            clones.append(wdir)
        else:
            wdir = ws
        wenv = dict(base_env)
        wenv["CARGO_TARGET_DIR"] = str(td_root if workers == 1 else td_root / f"w{i}")
        pool.put((wdir, wenv))

    if workers > 1 and verbose:
        print(f"  {project_name}: warming {workers} worker target dirs...", flush=True)
        def _warm(item):
            wdir, wenv = item
            _run(build_cmd, wdir, wenv, build_timeout)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_warm, list(pool.queue)))

    results = {s: [None] * len(mutants) for s in live}
    shared = {"unviable": 0, "viable": 0, "test_crate_broke": 0,
              "build_failed_other": 0, "build_s": 0.0, "test_s": 0.0}
    lock = threading.Lock()
    done = [0]
    t_start = time.time()

    # Phase 3 printed nothing until a whole project finished, which on a 30-minute
    # project is indistinguishable from a hang. The bar carries the counts that actually
    # tell you whether the run is healthy: viable/unviable split and timeouts so far.
    bar = tqdm(total=len(mutants), desc=f"{project_name} [{runner}, w{workers}]",
               unit="mut", dynamic_ncols=True, leave=True) if (tqdm and verbose) else None
    tallies = {"viable": 0, "unviable": 0, "timeout": 0, "killed": 0}

    def _one(idx):
        key, m = keys[idx], mutants[idx]
        wdir, wenv = pool.get()
        try:
            target = wdir / m["file"]
            original = target.read_text(encoding="utf-8")
            try:
                target.write_text(splice(original, m["span"], m["replacement"]),
                                  encoding="utf-8")

                # ONE build decides viability for every setting, and it spans the test
                # crates on purpose: some mutants compile as a library and only fail once
                # a dependent crate const-evaluates the mutated expression. One failure
                # retires the mutant for all settings, which is what keeps the denominator
                # shared by construction rather than by luck.
                rc, secs, out, exes = _build(wdir, build_cmd, wenv, build_timeout)
                with lock:
                    shared["build_s"] += secs
                if rc != 0:
                    with lock:
                        shared["unviable"] += 1
                        tallies["unviable"] += 1
                        if not _compile_error(out):
                            shared["build_failed_other"] += 1
                    for s in live:
                        results[s][idx] = (key, m, UNVIABLE)
                    return
                with lock:
                    shared["viable"] += 1
                    tallies["viable"] += 1

                for s, crate in sorted(live.items()):
                    if runner == "direct":
                        verdict, secs = _verdict_direct(
                            exes.get(crate, []), wdir / crate, wenv, caps[s])
                    else:
                        verdict, secs, out = _verdict_cargo(wdir, crate, wenv, caps[s], runner)
                        if verdict is UNVIABLE:
                            with lock:
                                shared["test_crate_broke"] += 1
                    with lock:
                        shared["test_s"] += secs
                        if verdict is TIMEOUT:
                            tallies["timeout"] += 1
                        elif verdict is CAUGHT:
                            tallies["killed"] += 1
                    results[s][idx] = (key, m, verdict)
            finally:
                target.write_text(original, encoding="utf-8")
        finally:
            pool.put((wdir, wenv))
            with lock:
                done[0] += 1
                if bar is not None:
                    bar.set_postfix(viable=tallies["viable"], unviable=tallies["unviable"],
                                    killed=tallies["killed"], timeout=tallies["timeout"],
                                    refresh=False)
                    bar.update(1)
                elif verbose and done[0] % 25 == 0:
                    el = time.time() - t_start
                    print(f"  {project_name}: {done[0]}/{len(mutants)} mutants, "
                          f"{el:.0f}s elapsed, {el / done[0]:.1f}s/mutant", flush=True)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, range(len(mutants))))
    else:
        for idx in range(len(mutants)):
            _one(idx)

    if bar is not None:
        bar.close()
    if workers > 1:
        for c in clones:
            shutil.rmtree(c, ignore_errors=True)

    summaries = {}
    for s, rows in sorted(results.items()):
        summaries[s] = _write_outcomes(
            Path(out_base) / SETTING_DIRS[s] / project_name / "mutants.out",
            [r for r in rows if r is not None], n_total, t_start,
            timeout_s=caps[s], baseline_test_s=base[s]["test_s"])

    if verbose:
        print(f"  {project_name}: {shared['viable']} viable / {shared['unviable']} unviable "
              f"of {len(mutants)}; builds {shared['build_s']:.0f}s, "
              f"tests {shared['test_s']:.0f}s")
        for s, sm in sorted(summaries.items()):
            print(f"    {s:9} killed {sm['killed']}/{sm['viable_mutants']} "
                  f"({sm['mutation_score']:.2%}) timeout {sm['timeout']}")

    shared = {**shared, "mutants_run": len(mutants), "mutants_total": n_total,
              "shard": None if only_names else shard, "workers": workers,
              "runner": runner, "min_timeout": min_timeout,
              "wall_s": round(time.time() - t_start, 1)}
    return {"settings": summaries, "shared": shared}


def validate_against_shared(a_dir, b_dir, project):
    """Compare two output trees mutant-for-mutant. Returns (n_compared, [mismatches])."""
    from harness.run_mutation import SETTING_DIRS

    mismatches, n = [], 0
    for setting, sd in SETTING_DIRS.items():
        pa = Path(a_dir) / sd / project / "mutants.out" / "outcomes.json"
        pb = Path(b_dir) / sd / project / "mutants.out" / "outcomes.json"
        if not (pa.exists() and pb.exists()):
            continue
        A = {o["mutant_key"]: o["summary"] for o in json.loads(pa.read_text())["outcomes"]}
        B = {o["mutant_key"]: o["summary"] for o in json.loads(pb.read_text())["outcomes"]}
        for k in sorted(set(A) & set(B)):
            n += 1
            if A[k] != B[k]:
                mismatches.append((setting, k, A[k], B[k]))
        for k in sorted(set(A) ^ set(B)):
            mismatches.append((setting, k, A.get(k, "<absent>"), B.get(k, "<absent>")))
    return n, mismatches


def main():
    import argparse
    from harness.split_crates import TEST_CRATE_NAMES

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("workspace")
    p.add_argument("--package", required=True)
    p.add_argument("--output", required=True, help="absolute path OUTSIDE the workspace")
    p.add_argument("--target-dir", default=None)
    p.add_argument("--jobs", type=int, default=None, help="CARGO_BUILD_JOBS per worker")
    p.add_argument("--workers", type=int, default=1, help="mutants tested concurrently")
    p.add_argument("--runner", choices=["direct", "cargo", "nextest"], default="direct")
    p.add_argument("--keep-debuginfo", action="store_true",
                   help="keep debuginfo (default drops it: 1.32x on the rebuild, and "
                        "mutation testing never reads a backtrace)")
    p.add_argument("--shard", default=None)
    p.add_argument("--timeout-multiplier", type=float, default=2.0)
    p.add_argument("--min-test-timeout", type=float, default=20)
    p.add_argument("--only-names-from", default=None)
    p.add_argument("--settings", nargs="+", default=sorted(TEST_CRATE_NAMES))
    p.add_argument("--mutants", choices=["cargo", "transpile", "both"], default="cargo",
                   help="which operator set to score (default: cargo-mutants, the "
                        "validated path)")
    a = p.parse_args()

    from harness.list_transpile_mutants import list_transpile_mutants
    listers = {
        "cargo": None,
        "transpile": list_transpile_mutants,
        "both": lambda ws, package, env=None, timeout=900: (
            list_mutants(ws, package, env=env, timeout=timeout)
            + list_transpile_mutants(ws, package, env=env, timeout=timeout)),
    }

    only = None
    if a.only_names_from:
        d = json.loads(Path(a.only_names_from).read_text())
        only = [o["scenario"]["Mutant"]["name"] for o in d["outcomes"]
                if isinstance(o.get("scenario"), dict) and "Mutant" in o["scenario"]]

    r = run_fast_mutation(
        a.workspace, a.package, {s: TEST_CRATE_NAMES[s] for s in a.settings},
        out_base=Path(a.output).resolve(), target_dir=a.target_dir, jobs=a.jobs,
        shard=a.shard, only_names=only, timeout_mult=a.timeout_multiplier,
        min_timeout=a.min_test_timeout, workers=a.workers, runner=a.runner,
        no_debuginfo=not a.keep_debuginfo, mutant_lister=listers[a.mutants])
    print(json.dumps(r["shared"], indent=2))


if __name__ == "__main__":
    main()
