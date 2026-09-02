import os
import sys
import json
import shutil
import argparse
import csv
import math
import stat
import time
import queue
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

SRC_DIR = Path("/nas/CRUST-bench-COLM/CRUST-bench/src")
sys.path.append(str(SRC_DIR))
# This module is both a library (every driver imports SETTING_DIRS from it) and a CLI run
# as ``python harness/run_mutation.py``. In the latter case sys.path[0] is ``harness/``,
# so the study root has to be added for its own ``harness.*`` imports to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the existing CRUST-bench pipeline from src/
from benchmark import Benchmark, load_benchmarks
from prompter import Prompter, RepairPrompter
from transpiler import Transpiler
from repairer import Repairer
from endpoint_config import endpoint_resolver
from endpoints.call_endpoint import get_result
from prompters.prompter_utils import find_best_parser
from utils.rust_project_builder import write_rust_files, create_top_level_cargo_toml
from harness.split_manifests import make_standalone, rewrite_manifest, _FALLBACK_VERSIONS
from harness.split_crates import build_workspace, TEST_CRATE_NAMES
from harness.mutate_shared import run_shared_mutation
from compile_projects import compile as compile_projects

# Default prompts shipped with the benchmark (markdown variants).
TRANSPILE_PROMPT = (
    SRC_DIR / "prompts/transpilation_prompts/markdown/safe_rust_ffi_system_interface.prompt"
)
REPAIR_PROMPT = (
    SRC_DIR / "prompts/repair_prompts/markdown/safe_rust_ffi_repair_system.prompt"
)
# Repairs the generated TEST files (never the source) so the mutation baseline is green
# while the mutated program stays byte-identical across settings.
TEST_REPAIR_PROMPT = SRC_DIR / "prompts/test_repair_prompts/repair_tests_only.prompt"
# Each setting uses a prompt whose declared inputs match what is actually fed to
# the model, so the system prompt never references files that aren't present.
UNIT_TEST_PROMPTS = {
    # C source + Rust interfaces (no C tests).
    "src": SRC_DIR / "prompts/unit_test_prompts/unit_test_from_c_source.prompt",
    # C source + C tests + Rust source (no interfaces).
    "src+test": SRC_DIR / "prompts/unit_test_prompts/unit_test_from_c_test_and_rust_source.prompt",
    # Rust source + Rust interfaces only (decoupled agent, no C).
    "src/test": SRC_DIR / "prompts/unit_test_prompts/unit_test_from_rust_source.prompt",
    # C source + Rust source + Rust interfaces.
    "c+rust": SRC_DIR / "prompts/unit_test_prompts/unit_test_from_c_and_rust_source.prompt",
}

PROMPT_FORMAT = "markdown_with_system_instructions"
PROMPT_STRATEGY = "all"

# The test-generation modalities under comparison. All mutate the SAME frozen Rust
# source, so any difference in mutation score is attributable to the test-generation
# modality alone (not to a different program under test).
SETTINGS = ["src", "src+test", "src/test", "c+rust"]

# Filesystem-safe directory name per setting ("/" is illegal in a path component).
SETTING_DIRS = {
    "src": "src",
    "src+test": "src+test",
    "src/test": "src_over_test",
    "c+rust": "c_and_rust",
}

# Projects that break the harness for environmental reasons (e.g. `impcheck` creates
# named pipes on the NFS mount that can't be copied/deleted). Skipped by name.
EXCLUDE_PROJECTS = {"impcheck"}


# --------------------------------------------------------------------------- #
# Collecting C test files (mirrors src/codex_expts/generate_unit_tests.py so
# the same test-selection heuristic is used everywhere).
# --------------------------------------------------------------------------- #
def _sanitize_project_name(name):
    name = name.replace("-", "_")
    if name and name[0].isdigit():
        name = "proj_" + name
    if "." in name:
        name = name.split(".")[0]
    return name


def _rbench_path(rust_dir, project_name):
    """Resolve a project's RBench directory, trying the capitalised variant too."""
    if not rust_dir:
        return None
    candidates = [
        Path(rust_dir) / project_name,
        Path(rust_dir) / (project_name[0].upper() + project_name[1:]),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def select_projects(benchmark_dir, rust_dir, single_benchmark, limit):
    """Pick the project directory names to run BEFORE constructing Benchmarks.

    Selecting first avoids `load_benchmarks` creating/seeding all 100 projects when we
    only want a handful. When ``rust_dir`` is given we keep only projects that have a
    matching RBench project (they provide the Rust interfaces the pipeline needs).
    """
    projs = sorted(
        p for p in os.listdir(benchmark_dir)
        if (Path(benchmark_dir) / p).is_dir()
        and _sanitize_project_name(p) not in EXCLUDE_PROJECTS
        and p not in EXCLUDE_PROJECTS
    )
    if rust_dir:
        projs = [p for p in projs if _rbench_path(rust_dir, _sanitize_project_name(p))]
    if single_benchmark:
        # accept a comma-separated list so several named projects run in ONE invocation
        # (one shared _mut_target pool -> safe concurrent mutation; separate invocations
        #  would collide on the same target dirs).
        targets = {_sanitize_project_name(t) for t in single_benchmark.split(",") if t.strip()}
        projs = [p for p in projs if _sanitize_project_name(p) in targets]
    if limit:
        projs = projs[:limit]
    return projs


def get_c_test_files(c_path):
    """Collect C test files from test/ or tests/ dirs or files with 'test' in the name."""
    test_files = []
    for fpath in Path(c_path).rglob("**/*"):
        if fpath.is_dir():
            continue
        if not (str(fpath).endswith(".c") or str(fpath).endswith(".h")):
            continue
        rel = str(fpath)
        is_test = "/test/" in rel or "/tests/" in rel or "test" in fpath.stem.lower()
        if not is_test:
            continue
        if "unity" in fpath.stem.lower():
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                test_files.append({"file_name": fpath.name, "content": f.read()})
        except Exception:
            pass
    return sorted(test_files, key=lambda x: x["file_name"])


def _render_files(header, files, lang):
    if not files:
        return ""
    block = f"{header}\n"
    for entry in files:
        block += f"{{{{{entry['file_name']}}}}}\n```{lang}\n{entry['content']}```\n\n"
    return block


def build_unit_test_prompt(benchmark, setting):
    """Build the messages for the unit-test-generation call for a given setting.

    The system prompt and the rendered files are kept in sync so the model is only
    told about inputs that are actually present:

    - ``src``       : C source + Rust interfaces (no C tests).
    - ``src+test``  : C source + C tests + Rust source (no interfaces).
    - ``src/test``  : Rust source + Rust interfaces only (a separate agent that only
                      sees the Rust side).
    - ``c+rust``    : C source + Rust source only (no interfaces).
    """
    if setting not in UNIT_TEST_PROMPTS:
        raise ValueError(f"Unknown setting: {setting}")
    with open(UNIT_TEST_PROMPTS[setting], "r", encoding="utf-8") as f:
        system_prompt = f.read()

    user = ""
    if setting == "src":
        user += _render_files("## The C Source Files are:", benchmark.c_files, "c")
        user += _render_files("## The Rust Interface Files are:", benchmark.rust_headers, "rust")
    elif setting == "src+test":
        # C source + C tests + Rust source -- no interfaces.
        user += _render_files("## The C Source Files are:", benchmark.c_files, "c")
        user += _render_files(
            "## The C Test Files are:", get_c_test_files(benchmark.c_path), "c"
        )
        user += _render_files("## The Rust Source Files are:", benchmark.rust_files, "rust")
    elif setting == "src/test":
        user += _render_files("## The Rust Source Files are:", benchmark.rust_files, "rust")
        user += _render_files("## The Rust Interface Files are:", benchmark.rust_headers, "rust")
    elif setting == "c+rust":
        # C source + Rust source only -- no interfaces.
        user += _render_files("## The C Source Files are:", benchmark.c_files, "c")
        user += _render_files("## The Rust Source Files are:", benchmark.rust_files, "rust")

    # settings that include the interface files vs. those that only show the Rust source.
    closing = ("Please generate Rust unit tests that conform to the Rust interfaces above."
               if setting in ("src", "src/test")
               else "Please generate Rust unit tests that exercise the Rust source above.")
    user += f"\nThe Rust crate name is `{benchmark.project_name}`. {closing}\n"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def write_test_files(benchmark, parsed_files):
    """Write generated test files into the project's src/bin directory."""
    bin_dir = benchmark.rust_path / "src" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for pf in parsed_files:
        content = pf["content"]
        # each bin target needs a main(); tests run via #[test] with `cargo test`.
        if "fn main()" not in content:
            content = content + "\nfn main() {}\n"
        with open(bin_dir / pf["file_name"].split("/")[-1], "w", encoding="utf-8") as f:
            f.write(content)


def _fallback_parse_tests(response, project_name):
    """Salvage tests when the model omits the ``{{filename.rs}}`` header.

    Some models emit a bare ```rust code block with the filename only in a leading
    comment (e.g. ``// foo_tests.rs``). Extract those blocks and name each file from
    that comment, falling back to a generated name.
    """
    import re
    blocks = re.findall(r"```(?:rust)?\n(.*?)```", response, flags=re.DOTALL)
    files = []
    for idx, body in enumerate(blocks):
        if "#[test]" not in body and "fn main" not in body:
            continue  # not a test/code block
        m = re.search(r"//\s*([\w./-]+\.rs)", body)
        name = Path(m.group(1)).name if m else f"{project_name}_tests_{idx}.rs"
        files.append({"file_name": name, "content": body})
    return files


def generate_rust_tests(benchmark, setting, endpoint, config, lock):
    """Ask the model to generate Rust unit tests for the benchmark and write them out."""
    prompt = build_unit_test_prompt(benchmark, setting)

    metadata_dir = benchmark.rust_path / "metadata" / "output"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "unit_test_prompt.json", "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2)

    response = get_result(prompt, lock, endpoint, config)["response"]
    with open(metadata_dir / "unit_test_response.txt", "w", encoding="utf-8") as f:
        f.write(response)

    parsed_files = find_best_parser(response)
    if not parsed_files:
        parsed_files = _fallback_parse_tests(response, benchmark.project_name)
    if not parsed_files:
        print(f"  WARNING: no test files parsed for {benchmark.project_name}")
        return
    write_test_files(benchmark, parsed_files)
    print(f"  Wrote {len(parsed_files)} test file(s) for {benchmark.project_name}")


# --------------------------------------------------------------------------- #
# Source translation (reuses Transpiler + Repairer from src/).
# --------------------------------------------------------------------------- #
def _normalize_rust_files(rust_files):
    """Collapse model-supplied filenames to bare basenames and de-duplicate.

    Models sometimes emit path-prefixed names like ``src/circular_buffer.rs``. The
    downstream ``write_rust_files`` builds lib.rs as ``pub mod {name.split('.')[0]};``,
    which turns such a name into the invalid line ``pub mod src/circular_buffer;`` and
    breaks the build. Reducing to the basename (keeping the longest content per name)
    prevents that.
    """
    by_name = {}
    for r in rust_files:
        name = Path(r["file_name"]).name
        if name not in by_name or len(r["content"]) >= len(by_name[name]["content"]):
            by_name[name] = {"file_name": name, "content": r["content"]}
    return list(by_name.values())


def translate_all_sources(benchmarks, endpoint, config, iterations, lock):
    """Transpile + repair the C->Rust source for ALL benchmarks in one batch.

    Reuses the exact Transpiler/Repairer machinery from src/run.py. Passing the full
    benchmark list (rather than one at a time) lets their built-in multiprocessing
    Pool issue the transpilation/repair LLM calls **in parallel** across benchmarks.
    The resulting source is frozen and shared by every downstream setting.
    """
    if not benchmarks:
        return []
    # all benchmarks share the same frozen output dir.
    output_dir = benchmarks[0].rust_path.parent

    prompter = Prompter(TRANSPILE_PROMPT, PROMPT_FORMAT, PROMPT_STRATEGY, include_headers=True)
    transpiler = Transpiler(benchmarks, prompter, endpoint=endpoint, config=config)
    results, benchmarks = transpiler.run()          # parallel over benchmarks
    transpiler.log_benchmarks(results)
    for b in benchmarks:
        b.rust_files = _normalize_rust_files(b.rust_files)
        write_rust_files(b)
    compile_projects(output_dir, 0)

    for i in range(iterations):
        repair_prompter = RepairPrompter(REPAIR_PROMPT, PROMPT_FORMAT, PROMPT_STRATEGY)
        repairer = Repairer(benchmarks, repair_prompter, i, endpoint, config=config)
        results, benchmarks = repairer.run()        # parallel over benchmarks needing repair
        repairer.log_results(results)
        for b in benchmarks:
            b.rust_files = _normalize_rust_files(b.rust_files)
            write_rust_files(b)
        compile_projects(output_dir, i + 1)

    # keep a stable order for the downstream per-project loop.
    return sorted(benchmarks, key=lambda b: b.project_name)


# --------------------------------------------------------------------------- #
# Per-setting project materialisation.
#
# The Rust *source* is generated once (translate_all_sources) and frozen. For each
# setting we copy that frozen project and drop in only that setting's generated
# tests, so all three settings are mutation-tested against byte-identical source.
# --------------------------------------------------------------------------- #
def _ignore_uncopyable(dirpath, names):
    """copytree ignore fn: skip build artefacts and files copytree can't handle.

    Some C projects (e.g. impcheck) leave named pipes / sockets / NFS temp files in
    the tree; shutil.copytree raises on FIFOs, so we drop them (and target/mutants.out).
    """
    skip = {"target", "mutants.out"}
    for n in names:
        if n.startswith(".nfs"):
            skip.add(n)
            continue
        try:
            mode = os.lstat(os.path.join(dirpath, n)).st_mode
        except OSError:
            skip.add(n)
            continue
        if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
            skip.add(n)
    return skip


def materialize_setting_project(frozen_benchmark, setting, output_dir):
    """Copy the frozen source into the setting's workspace and clear old tests.

    Does no LLM work and does NOT write the workspace manifest (that is done once
    per setting dir, after all copies, to avoid a race when parallelising).
    """
    if setting not in SETTING_DIRS:
        raise ValueError(f"Unknown setting: {setting}")
    setting_dir = Path(output_dir) / SETTING_DIRS[setting]
    setting_dir.mkdir(parents=True, exist_ok=True)
    proj_dst = setting_dir / frozen_benchmark.project_name
    if proj_dst.exists():
        shutil.rmtree(proj_dst)

    # copy the frozen source project; skip build artefacts and uncopyable special files.
    shutil.copytree(frozen_benchmark.rust_path, proj_dst, ignore=_ignore_uncopyable)
    # start from a clean test dir so only this setting's tests are present.
    bin_dir = proj_dst / "src" / "bin"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)

    # reload as a Benchmark rooted at the copy (its rust_files == the frozen source).
    return Benchmark(str(frozen_benchmark.c_path), str(proj_dst))


def run_translation(frozen_benchmark, setting, output_dir, endpoint, config, lock):
    """Materialise the project for ``setting`` and generate that setting's tests.

    The C->Rust *source* translation is done once, up front, in ``translate_all_sources``
    and shared by every setting. Here we only vary how the Rust tests are produced:

    - ``src``      : tests generated from the C source + Rust interfaces (no C tests).
    - ``src+test`` : tests generated from the C source + C tests + Rust interfaces.
    - ``src/test`` : a separate agent generates tests from the Rust source + interfaces.
    """
    benchmark = materialize_setting_project(frozen_benchmark, setting, output_dir)
    generate_rust_tests(benchmark, setting, endpoint, config, lock)
    # Re-prune this project's manifest now that its tests exist: the generated tests
    # may reference a crate the source alone does not. Touches only this project's
    # Cargo.toml, so it is safe to run inside the concurrent test-gen phase.
    rewrite_manifest(benchmark.rust_path, _FALLBACK_VERSIONS)
    return benchmark


# --------------------------------------------------------------------------- #
# Green baseline: repair the generated tests (never the source), then prune.
#
# cargo-mutants refuses to run unless the unmutated test suite is green. We first
# try to REPAIR failing tests against the frozen source, then drop whatever still
# fails. Repairing/pruning only ever touches the test files in src/bin, so the
# mutated program stays byte-identical across settings.
# --------------------------------------------------------------------------- #
class TestSuite:
    """Where one setting's test files live and how to run a single one of them.

    Two layouts exist. Legacy: the tests are ``src/bin/*.rs`` inside the mutated crate,
    run as ``cargo test --bin <name>``. Split (see ``split_crates.py``): they are
    ``tests/*.rs`` in a separate crate of the project workspace, run as
    ``cargo test -p <crate> --test <name>`` from the workspace root.
    """

    def __init__(self, dir, cwd, package=None, kind="bin"):
        self.dir = Path(dir)
        self.cwd = Path(cwd)
        self.package = package
        self.kind = kind

    def files(self):
        return sorted(self.dir.glob("*.rs")) if self.dir.exists() else []

    def cmd(self, name):
        pkg = ["-p", self.package] if self.package else []
        return ["cargo", "test", *pkg, f"--{self.kind}", name, "--", "--test-threads=1"]


def _legacy_suite(benchmark):
    """The in-crate ``src/bin`` suite used when the source and tests share a crate."""
    return TestSuite(benchmark.rust_path / "src" / "bin", benchmark.rust_path)


def _cargo_test_bin(suite, name, timeout=300, target_dir=None, jobs=None):
    """Run one test target of ``suite``. Returns (passed, combined_output).

    ``target_dir`` isolates this build in its own CARGO_TARGET_DIR. This is REQUIRED
    when baselines run concurrently: several settings of the same crate building into
    one shared target emit the same lib<crate>-<hash>.rlib and clobber each other,
    producing spurious test failures (which the prune step then deletes). One target
    dir per concurrent worker eliminates the race.
    """
    env = os.environ.copy()
    if target_dir:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    if jobs:
        env["CARGO_BUILD_JOBS"] = str(jobs)
    try:
        r = subprocess.run(
            suite.cmd(name),
            cwd=suite.cwd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    passed = r.returncode == 0 and "FAILED" not in r.stdout
    return passed, (r.stdout + "\n" + r.stderr)


def _read_test_file(bin_path):
    with open(bin_path, "r", encoding="utf-8") as f:
        return {"file_name": bin_path.name, "content": f.read()}


def build_test_repair_prompt(benchmark, test_file, error_output):
    """Prompt to repair a single failing test file against the frozen Rust source."""
    with open(TEST_REPAIR_PROMPT, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    user = ""
    user += _render_files("## The Rust Source Files are:", benchmark.rust_files, "rust")
    user += _render_files("## The Rust Interface Files are:", benchmark.rust_headers, "rust")
    user += _render_files("## The failing Rust Test File is:", [test_file], "rust")
    user += "## The compiler / test output is:\n```\n" + error_output.strip() + "\n```\n"
    user += (
        f"\nThe Rust crate name is `{benchmark.project_name}`. "
        "Please return the corrected test file only.\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def repair_tests_loop(benchmark, endpoint, config, lock, iterations, target_dir=None, jobs=None,
                      suite=None):
    """Iteratively repair failing test files against the frozen source.

    Only the test files are rewritten; the source is never touched. Stops early once
    every test target passes.
    """
    suite = suite or _legacy_suite(benchmark)
    if not suite.dir.exists() or iterations <= 0:
        return

    for it in range(iterations):
        failing = []
        for f in suite.files():
            passed, output = _cargo_test_bin(suite, f.stem, target_dir=target_dir, jobs=jobs)
            if not passed:
                failing.append((f, output))
        if not failing:
            return  # baseline is already green

        print(f"  [{benchmark.project_name}] test-repair round {it}: {len(failing)} failing")
        for f, output in failing:
            prompt = build_test_repair_prompt(benchmark, _read_test_file(f), output)
            try:
                response = get_result(prompt, lock, endpoint, config)["response"]
            except Exception as e:  # keep repairing the other files
                print(f"    repair call failed for {f.name}: {e}")
                continue
            parsed = find_best_parser(response)
            for pf in parsed:
                # only write back the file we asked to repair.
                if pf["file_name"].split("/")[-1] != f.name:
                    continue
                content = pf["content"]
                if "fn main()" not in content:
                    content = content + "\nfn main() {}\n"
                with open(f, "w", encoding="utf-8") as out:
                    out.write(content)


def _sync_tests_back(src_dir, dst_dir):
    """Mirror a test crate's ``tests/`` back onto the setting copy's ``src/bin``.

    Under the split layout the workspace is where tests get repaired and pruned, but
    ``--reuse_tests`` (and anything reading the setting trees) still looks in
    ``src/bin``. Keeping the two in step means a later run reuses the *green* tests
    rather than the originals, and that pruned tests stay pruned.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    keep = set()
    for f in sorted(src_dir.glob("*.rs")):
        shutil.copy2(f, dst_dir / f.name)
        keep.add(f.name)
    for f in sorted(dst_dir.glob("*.rs")):
        if f.name not in keep:
            f.unlink()


def ensure_green_baseline(benchmark, endpoint, config, lock, repair_iterations, target_dir=None,
                          jobs=None, suite=None):
    """Make the test suite green: repair what we can, drop what we cannot.

    Returns (kept, total): test files remaining vs. originally generated.

    ``target_dir`` isolates every cargo build here in its own CARGO_TARGET_DIR so
    concurrent baselines (across projects/settings) don't clobber a shared target.
    ``suite`` selects the layout; it must be given per *test crate* under the split
    layout, where a test that fails to compile breaks its own crate rather than
    marking mutants unviable.
    """
    suite = suite or _legacy_suite(benchmark)
    if not suite.dir.exists():
        return 0, 0
    total = len(suite.files())

    repair_tests_loop(benchmark, endpoint, config, lock, repair_iterations,
                      target_dir=target_dir, jobs=jobs, suite=suite)

    # prune anything still failing so cargo-mutants has a green baseline.
    kept = 0
    for f in suite.files():
        passed, _ = _cargo_test_bin(suite, f.stem, target_dir=target_dir, jobs=jobs)
        if passed:
            kept += 1
        else:
            f.unlink()
    return kept, total


# --------------------------------------------------------------------------- #
# Mutation testing via cargo-mutants.
# --------------------------------------------------------------------------- #
def _summarize_outcomes(outcomes_path):
    """Turn cargo-mutants' outcomes.json into a mutation-score summary."""
    with open(outcomes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = {"caught": 0, "missed": 0, "timeout": 0, "unviable": 0}
    for outcome in data.get("outcomes", []):
        summary = outcome.get("summary", "")
        if summary == "CaughtMutant":
            counts["caught"] += 1
        elif summary == "MissedMutant":
            counts["missed"] += 1
        elif summary == "Timeout":
            counts["timeout"] += 1
        elif summary == "Unviable":
            counts["unviable"] += 1

    # fall back to the top-level totals cargo-mutants also writes.
    for key in counts:
        if counts[key] == 0 and key in data:
            counts[key] = data[key]

    killed = counts["caught"]
    survived = counts["missed"]
    viable = killed + survived
    mutation_score = (killed / viable) if viable else 0.0
    return {
        "killed": killed,
        "survived": survived,
        "timeout": counts["timeout"],
        "unviable": counts["unviable"],
        "viable_mutants": viable,
        "mutation_score": mutation_score,
    }


def run_mutation(benchmark, endpoint, config, lock, repair_iterations=3, timeout=3600, shard=None, jobs=16):
    """Green-baseline (repair + prune) then mutation-test a materialised project.

    Convenience wrapper used by the non-phased path; the phased ``run_experiment``
    calls ``ensure_green_baseline`` and ``mutate_with_cargo`` separately so the
    (network-bound) baseline repair can be parallelised apart from cargo-mutants.
    """
    kept, total = ensure_green_baseline(benchmark, endpoint, config, lock, repair_iterations)
    return mutate_with_cargo(benchmark, kept, total, shard=shard, jobs=jobs, timeout=timeout)


MUTATE_EXCLUDES = ["--exclude", "**/src/bin/**", "--exclude", "**/src/interfaces/**"]


def _strip_dangling_targets(proj):
    """Remove [[test]]/[[bin]] blocks from a project's Cargo.toml whose `path` file is
    missing. RBench seeds an explicit `[[test]] path="src/bin/test.rs"` pointing at the
    ground-truth test; once we swap in our own (differently-named) generated tests, that
    path dangles and cargo-mutants' whole-package baseline build fails ("can't find
    integration-test"). cargo auto-discovers src/bin/*.rs anyway, so dropping the dead
    block is safe. Operates ONLY on the given project copy -- never on datasets/RBench.
    """
    cargo = Path(proj) / "Cargo.toml"
    if not cargo.exists():
        return
    lines = cargo.read_text(encoding="utf-8").splitlines()
    out, i, changed = [], 0, False
    while i < len(lines):
        if lines[i].strip() in ("[[test]]", "[[bin]]"):
            block, j = [lines[i]], i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("["):
                block.append(lines[j]); j += 1
            path = None
            for b in block:
                if b.strip().startswith("path"):
                    path = b.split("=", 1)[1].strip().strip('"').strip("'")
            if path is not None and not (Path(proj) / path).exists():
                changed = True; i = j; continue      # drop the dangling block
            out.extend(block); i = j
        else:
            out.append(lines[i]); i += 1
    if changed:
        cargo.write_text("\n".join(out) + "\n", encoding="utf-8")


def _count_mutants(proj, env=None, package=None):
    """Return how many mutants cargo-mutants would generate (implementation only).

    Under the split layout ``package`` scopes generation to the source crate, so the
    count is a property of the source alone and is shared by all four settings.
    """
    pkg = ["--package", package] if package else []
    try:
        r = subprocess.run(["cargo", "mutants", "--list"] + pkg + MUTATE_EXCLUDES,
                           cwd=proj, capture_output=True, text=True, timeout=900, env=env)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return sum(1 for line in r.stdout.splitlines() if line.strip())


def _decide_shard(n_mutants, shard, big_threshold, big_shard_target, project_name=None):
    """(applied_shard, sharded): the full mutant set, or a moderate shard for big projects.

    Shared by ``mutate_with_cargo`` and the shared-mutant path so a project shards the
    same way whichever runs it.
    """
    if n_mutants and big_threshold and n_mutants > big_threshold:
        applied = f"1/{max(2, math.ceil(n_mutants / big_shard_target))}"
        if project_name:
            print(f"  {project_name}: BIG ({n_mutants} mutants) -> sharding {applied}")
        return applied, True
    return shard, False


def _test_package_baseline(proj, test_package, env=None, timeout=1800):
    """Run one test crate unmutated. Returns (passed, test_seconds, output).

    Needed only under the split layout, and it is not redundant with cargo-mutants'
    own baseline: cargo-mutants baselines the package it MUTATES, not the one named by
    ``--test-package``. Verified from a split run's outcomes.json, where the Baseline
    scenario ran ``cargo test --package=<core>`` while every mutant ran
    ``cargo test --package=tests-<setting>``. Since ``core`` carries no tests of its
    own, that baseline is vacuous: a setting whose suite is red or does not compile
    sails past it and comes back as all-caught or all-unviable with no warning. In the
    legacy layout the suite lived in the mutated crate, so cargo-mutants' baseline
    covered it for free; splitting the crates dropped that guard, and this restores it.

    Also returns the suite's test time, which is the input cargo-mutants would have
    derived ``--timeout-multiplier`` from. Off a 0.05s vacuous baseline the multiplier
    is meaningless and every mutant falls back to the 20s floor -- long enough to be a
    spurious-Timeout hazard on slow suites, and timeouts sit in the denominator without
    counting as kills.

    Builds first (``--no-run``) so the returned time is test time rather than build
    time, and so the target dir is warm before cargo-mutants starts.
    """
    base = ["cargo", "test", "--package", test_package]
    try:
        b = subprocess.run(base + ["--no-run"], cwd=proj, capture_output=True, text=True,
                           timeout=timeout, env=env)
        if b.returncode != 0:
            return False, 0.0, b.stdout + "\n" + b.stderr
        t0 = time.monotonic()
        r = subprocess.run(base, cwd=proj, capture_output=True, text=True,
                           timeout=timeout, env=env)
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return False, float(timeout), "Timeout"
    passed = r.returncode == 0 and "FAILED" not in r.stdout
    return passed, elapsed, r.stdout + "\n" + r.stderr


def mutate_with_cargo(benchmark, kept, total, shard=None, jobs=16, timeout=None,
                      big_threshold=300, big_shard_target=150, timeout_mult=2.0, target_dir=None,
                      out_base=None, proj_dir=None, package=None, test_package=None,
                      metadata_dir=None, n_mutants=None):
    """Run cargo-mutants on a project whose green baseline is already established.

    Under the split layout (``split_crates.py``) pass ``proj_dir`` = the project's
    workspace root, ``package`` = the source crate and ``test_package`` = the setting's
    test crate. Mutant generation is then scoped to the source crate, so all four
    settings see the same mutants and the same viable count; pass the project's shared
    ``n_mutants`` too, so the sharding decision (and hence the shard's mutant subset) is
    also identical across settings.

    Scoped to the implementation only. Runs the FULL mutant set for normal-sized
    projects; for "big" projects (> ``big_threshold`` mutants) it applies a moderate
    shard so ~``big_shard_target`` mutants run, and records that it did so.

    Two timeouts matter and were the source of earlier stalls:
      * ``mutant_timeout`` -- per-mutant test cap passed to cargo-mutants ``--timeout``.
        Without it, cargo-mutants derives a huge per-mutant timeout from a slow baseline
        test, so a single slow/hanging generated test made every mutant crawl.
      * ``timeout`` -- overall wall cap for this project's cargo-mutants run. If None it
        is computed DYNAMICALLY from how many mutants will actually run, so big (sharded)
        projects get proportionally more time and small ones aren't over-allotted.

    Always writes metadata/mutation.json (even on timeout) so a resume can tell
    attempted-but-failed from not-yet-run. CPU-heavy (uses ``jobs`` cores) -> call
    sequentially, not in a thread pool.
    """
    proj = Path(proj_dir) if proj_dir else benchmark.rust_path
    metadata_dir = Path(metadata_dir) if metadata_dir else proj / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    def _write(summary):
        with open(metadata_dir / "mutation.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    if kept == 0:
        print(f"  {benchmark.project_name}: no passing tests, skipping mutation")
        return _write({"project": benchmark.project_name, "baseline_valid": False,
                       "reason": "no_passing_tests", "tests_kept": 0, "tests_total": total})

    # Drop dangling [[test]]/[[bin]] target declarations (RBench-seeded, now missing)
    # so cargo-mutants' baseline build doesn't fail on a target-resolution error.
    _strip_dangling_targets(proj)

    # A dedicated cargo target dir (when running projects in parallel) keeps each
    # concurrent cargo-mutants run isolated. This is REQUIRED for correctness, not just
    # speed: two cargo-mutants processes (or cargo-mutants' own -j jobs) writing the same
    # target concurrently emit the same `lib<crate>-<hash>.rlib` and clobber each other,
    # so builds fail spuriously and get recorded as *Unviable*. Combined with -j 1 below
    # (serial mutant builds) and CARGO_INCREMENTAL=0, no two builds ever share a target.
    env = os.environ.copy()
    if target_dir:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    # cargo-mutants runs serially (-j 1); `jobs` now caps each individual cargo *build*'s
    # core use so N concurrent projects (the worker pool) use ~N*jobs cores total instead
    # of oversubscribing (every build defaulting to all cores).
    if jobs:
        env["CARGO_BUILD_JOBS"] = str(jobs)

    # Decide sharding: full set for normal projects, a moderate shard for big ones.
    if n_mutants is None:
        n_mutants = _count_mutants(proj, env=env, package=package)
    applied_shard, sharded = _decide_shard(n_mutants, shard, big_threshold, big_shard_target,
                                           project_name=benchmark.project_name)

    # Write cargo-mutants' output OUTSIDE the workspace tree when out_base is given.
    # cargo-mutants copies the whole enclosing workspace before each run; if its output
    # (mutants.out) lives inside that tree, a concurrent run's rmtree/create of a sibling
    # mutants.out makes another run's tree-walk hit a vanishing path and abort. An external
    # out_base keeps the copied tree static, so concurrent runs are safe.
    # absolute so cargo-mutants (run with cwd=proj) doesn't resolve it inside the tree.
    mo_parent = Path(out_base).resolve() if out_base else proj
    mo_parent.mkdir(parents=True, exist_ok=True)
    out_dir = mo_parent / "mutants.out"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    # NB: cargo-mutants reports package-prefixed paths, so excludes are anchored `**/`.
    # We run inside the member dir (no --package, which mis-resolves and yields 0).
    # DYNAMIC per-mutant timeout: in the LEGACY layout --timeout-multiplier sets each
    # mutant's test cap to (baseline_test_time x mult). We deliberately do NOT set an
    # absolute --timeout there, which also caps the baseline run -- a fixed 20s killed
    # the baselines of slow-but-valid test suites, yielding 0 mutants tested. Under the
    # split layout we run the baseline ourselves and pass an absolute --timeout; see below.
    # -j 1: run cargo-mutants' mutant builds SERIALLY. With a forced CARGO_TARGET_DIR its
    # internal parallel jobs build the SAME crate concurrently into one target and clobber
    # each other's artifacts -> spurious "unviable" + repeated full rebuilds. Throughput
    # comes from the cross-project worker pool (many projects at once, each -j1) instead.
    # Within a run, mutants are still cheap: only the mutated crate recompiles, deps are
    # reused as cached .rlibs (build-from-baseline), so serial is not a bottleneck.
    cmd = ["cargo", "mutants", "--no-shuffle", "-j", "1",
           "--output", str(mo_parent)] + MUTATE_EXCLUDES
    # Split layout: mutate only the source crate, test only this setting's crate.
    if package:
        cmd += ["--package", package]

    baseline_s = None
    if test_package:
        cmd += ["--test-package", test_package]
        # cargo-mutants would baseline `package` (which has no tests), so gate on the
        # test crate ourselves -- see _test_package_baseline -- and then tell
        # cargo-mutants to skip its own vacuous baseline. That also frees us to pass an
        # absolute --timeout, which is otherwise unsafe (it caps the baseline run too),
        # computed from the suite time we just measured instead of from a 0.05s no-op.
        ok, baseline_s, out = _test_package_baseline(proj, test_package, env=env)
        if not ok:
            print(f"  {benchmark.project_name}/{test_package}: unmutated suite is red, "
                  f"skipping mutation")
            return _write({"project": benchmark.project_name, "baseline_valid": False,
                           "reason": "test_package_baseline_failed", "tests_kept": kept,
                           "tests_total": total, "package": package,
                           "test_package": test_package, "baseline_output": out[-4000:]})
        mutant_timeout = max(20, math.ceil(baseline_s * timeout_mult))
        cmd += ["--baseline", "skip", "--timeout", str(mutant_timeout)]
    else:
        cmd += ["--timeout-multiplier", str(timeout_mult)]

    if applied_shard:
        cmd += ["--shard", applied_shard]

    # Dynamic overall wall cap: scale with the number of mutants that will actually run.
    # cargo-mutants runs serially (-j 1), so wall time ~= baseline build + run_est mutants
    # each costing (incremental build + a capped test). Budget a generous ~45s/mutant plus
    # a fixed baseline-build allowance; floor 1500s, cap 2h.
    if timeout is None:
        shard_n = int(applied_shard.split("/")[1]) if applied_shard else 1
        run_est = max(1, math.ceil((n_mutants or big_shard_target) / shard_n))
        timeout = min(7200, max(1500, 300 + run_est * 45))

    meta = {"project": benchmark.project_name, "tests_kept": kept, "tests_total": total,
            "mutants_total": n_mutants, "sharded": sharded, "shard": applied_shard,
            "timeout_multiplier": timeout_mult, "project_timeout_s": timeout,
            "package": package, "test_package": test_package,
            "baseline_test_s": round(baseline_s, 3) if baseline_s is not None else None}
    try:
        subprocess.run(cmd, cwd=proj, capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError:
        print("cargo-mutants not found. Install it with: cargo install cargo-mutants")
        return None
    except subprocess.TimeoutExpired:
        print(f"  {benchmark.project_name}: cargo-mutants timed out")
        return _write({**meta, "baseline_valid": False, "reason": "timeout"})

    outcomes_path = out_dir / "outcomes.json"
    if not outcomes_path.exists():
        print(f"  {benchmark.project_name}: no outcomes.json (baseline may still fail)")
        return _write({**meta, "baseline_valid": False, "reason": "no_outcomes"})

    summary = {**meta, **_summarize_outcomes(outcomes_path), "baseline_valid": True}
    print(
        f"  {benchmark.project_name}: killed {summary['killed']}/{summary['viable_mutants']} "
        f"(score {summary['mutation_score']:.2%}, {kept}/{total} tests"
        f"{', sharded '+applied_shard if sharded else ''})"
    )
    return _write(summary)


def _aggregate(per_project, settings, output_dir):
    """Compare settings only over projects with a valid baseline in EVERY setting."""
    comparable = [
        proj for proj, by_setting in per_project.items()
        if all(
            by_setting.get(s) and by_setting[s].get("baseline_valid")
            and by_setting[s].get("viable_mutants", 0) > 0
            for s in settings
        )
    ]

    averages = {}
    for s in settings:
        scores = [per_project[p][s]["mutation_score"] for p in comparable]
        tests = [per_project[p][s]["tests_kept"] for p in comparable]
        averages[s] = {
            "avg_mutation_score": (sum(scores) / len(scores)) if scores else None,
            "avg_tests_kept": (sum(tests) / len(tests)) if tests else None,
        }

    # Note which (project, setting) runs used a moderate shard because they were big.
    sharded = sorted({
        f"{p}/{s} ({r.get('mutants_total')}mut -> {r.get('shard')})"
        for p, by in per_project.items() for s, r in by.items()
        if isinstance(r, dict) and r.get("sharded")
    })

    # Projects that yielded ZERO viable mutants in EVERY setting -- the frozen source
    # has nothing (usefully) mutatable, so they carry no signal and can't be compared.
    def _zero_all(by):
        vals = [by.get(s) for s in settings]
        return all(
            isinstance(r, dict) and r.get("baseline_valid") and r.get("viable_mutants", 0) == 0
            for r in vals
        ) and any(isinstance(r, dict) and r.get("baseline_valid") for r in vals)
    zero_mutant_projects = sorted(p for p, by in per_project.items() if _zero_all(by))

    report = {
        "settings": settings,
        "n_projects_attempted": len(per_project),
        "n_comparable": len(comparable),
        "comparable_projects": comparable,
        "zero_mutant_projects": zero_mutant_projects,
        "averages_over_comparable": averages,
        "sharded_runs": sharded,
        "per_project": per_project,
    }
    with open(Path(output_dir) / "mutation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_results_table(per_project, settings, comparable, averages, output_dir, zero_mutant_projects)
    if zero_mutant_projects:
        print(f"\n{len(zero_mutant_projects)} project(s) have ZERO mutants in every setting "
              f"(no signal, excluded from comparison):")
        print("  " + ", ".join(zero_mutant_projects))
    if sharded:
        print(f"\n{len(sharded)} run(s) used a moderate shard (big projects).")
    return report


def _result_cell(r):
    """Render one (project, setting) result as a compact table cell ('*' = sharded)."""
    if r and r.get("baseline_valid") and r.get("viable_mutants", 0) > 0:
        star = "*" if r.get("sharded") else ""
        return f"{r['mutation_score']*100:.0f}% ({r['killed']}/{r['viable_mutants']}){star}"
    reasons = {
        "source_build_failed": "no-build",
        "no_passing_tests": "no-tests",
        "no_outcomes": "no-outcomes",
        "timeout": "timeout",
        "tool_unavailable": "no-tool",
    }
    if r and r.get("baseline_valid") and r.get("viable_mutants", 0) == 0:
        return "0-mutants"
    return reasons.get((r or {}).get("reason"), "-")


def _write_results_table(per_project, settings, comparable, averages, output_dir,
                         zero_mutant_projects=()):
    """Print an aligned results table and write Markdown + CSV copies to output_dir.

    A ``note`` column flags projects that are ``comparable``, ``ZERO-MUTANTS`` (no
    mutants in any setting), or ``no`` (partial coverage).
    """
    output_dir = Path(output_dir)
    zero_set = set(zero_mutant_projects)
    projects = list(per_project.keys())
    headers = ["project", *settings, "note"]

    rows = []
    for p in projects:
        row = [p]
        for s in settings:
            row.append(_result_cell(per_project[p].get(s, {})))
        row.append("comparable" if p in comparable else ("ZERO-MUTANTS" if p in zero_set else "no"))
        rows.append(row)

    # AVERAGE row over comparable projects (mutation score + avg tests).
    avg_row = ["AVERAGE (comparable)"]
    for s in settings:
        a = averages[s]
        if a["avg_mutation_score"] is None:
            avg_row.append("n/a")
        else:
            avg_row.append(f"{a['avg_mutation_score']*100:.1f}% [~{a['avg_tests_kept']:.1f} tests]")
    avg_row.append(f"n={len(comparable)}")

    # ---- aligned console table ----
    all_rows = rows + [avg_row]
    widths = [max(len(str(r[i])) for r in ([headers] + all_rows)) for i in range(len(headers))]
    def fmt(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    sep = "  ".join("-" * w for w in widths)
    print("\n=== Mutation comparison (cell = score (killed/viable); "
          f"averaged over {len(comparable)} comparable project(s)) ===")
    print(fmt(headers))
    print(sep)
    for r in rows:
        print(fmt(r))
    print(sep)
    print(fmt(avg_row))
    if not comparable:
        print("\n  No project had a valid baseline in ALL settings; averages are n/a.")

    # ---- Markdown ----
    md = ["| " + " | ".join(headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows + [avg_row]:
        md.append("| " + " | ".join(str(c) for c in r) + " |")
    (output_dir / "mutation_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- CSV ----
    with open(output_dir / "mutation_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
        w.writerow(avg_row)

    print(f"\nWrote table to {output_dir/'mutation_table.md'} and {output_dir/'mutation_table.csv'}")


def _mutation_json_path(output_dir, project_name, setting, split_crates=False):
    """Where a (project, setting) run records its summary, per layout."""
    output_dir = Path(output_dir)
    if split_crates:
        # one workspace per project, one metadata dir per setting inside it.
        return (output_dir / "_workspaces" / project_name / "metadata"
                / SETTING_DIRS[setting] / "mutation.json")
    return output_dir / SETTING_DIRS[setting] / project_name / "metadata" / "mutation.json"


def _load_done_result(output_dir, project_name, setting, resume, split_crates=False):
    """If resuming and a definitive result already exists on disk, return it.

    A result counts as done only if it is a valid mutation run or a definitive
    non-transient failure (no passing tests / source didn't build). Transient
    outcomes (timeout / no_outcomes / errors) return None so they are re-run.
    """
    if not resume:
        return None
    m = _mutation_json_path(output_dir, project_name, setting, split_crates)
    if not m.exists():
        return None
    try:
        d = json.load(open(m, encoding="utf-8"))
    except Exception:
        return None
    # source_build_failed is definitive (the frozen source itself won't compile).
    # NOTE: no_passing_tests is NOT treated as done -- a prior workspace-manifest bug
    # (a member dir missing its Cargo.toml poisoning the whole workspace) produced many
    # spurious no_passing_tests, so we recompute them.
    if d.get("reason") == "source_build_failed":
        return d
    # A valid mutation run is done ONLY if it actually tested mutants, OR the project
    # genuinely has no mutants (mutants_total == 0). A baseline_valid result with
    # viable==0 but mutants existing/unknown is stale/broken (e.g. baseline timed out
    # in an earlier run) -> recompute it.
    if d.get("baseline_valid"):
        if d.get("viable_mutants", 0) > 0 or d.get("mutants_total") == 0:
            return d
    return None


def run_experiment(benchmarks, output_dir, endpoint, config, iterations, settings,
                   repair_iterations=3, shard=None, jobs=16, test_workers=8,
                   resume=False, big_threshold=300, big_shard_target=150, timeout_mult=2.0,
                   skip_translate=False, mutation_workers=4, reuse_tests=False,
                   split_crates=False, shared_mutants=False, min_test_timeout=20):
    """Freeze every project's source once (parallel), then mutation-test all settings.

    LLM-bound phases (source translation, test generation, test-repair baseline) run
    concurrently; the CPU-bound cargo-mutants phase runs sequentially since each run
    already saturates ``jobs`` cores. With ``resume=True``, (project, setting) pairs
    that already have a definitive mutation.json are loaded and skipped. With
    ``skip_translate=True``, Phase 1 is skipped entirely and the already-frozen source
    on disk (from a prior run) is used as-is -- the ``benchmarks`` passed in must
    already be rooted at the frozen source dir.

    With ``split_crates=True`` the generated tests are moved out of the mutated crate
    into one test crate per setting (see ``split_crates.py``), so mutant generation and
    viability depend on the source alone and all four settings score over the same
    denominator. Test generation is unchanged: it still runs against the per-setting
    project copies, whose ``src/bin`` remains the store of record for ``--reuse_tests``.

    ``shared_mutants=True`` (requires ``split_crates``) additionally replaces the four
    cargo-mutants invocations per project with a single ``mutate_shared`` run that applies
    each mutant once and tests every suite against that one build. Beyond being faster it
    is what makes the denominator shared *by construction*: one batched build decides
    viability for all four settings at once, instead of each setting rediscovering it.
    ``--resume`` then works at project granularity -- a project with any setting still
    outstanding re-runs the settings that are outstanding.
    """
    if shared_mutants and not split_crates:
        raise ValueError("shared_mutants requires split_crates: the mutant and the test "
                         "suites have to be in separate crates before one build can serve "
                         "all four settings")
    lock = multiprocessing.Manager().Lock()
    # Absolute: cargo/cargo-mutants run with cwd=<project>, so any relative CARGO_TARGET_DIR
    # or --output would resolve INSIDE the project tree (breaking isolation + the external
    # output that keeps concurrent tree-copies race-free).
    output_dir = Path(output_dir).resolve()
    per_project = {}

    timings = {}
    t0 = time.time()

    # 1. Transpile + repair the source for ALL benchmarks (unless reusing frozen source).
    if skip_translate:
        print(f"Phase 1: SKIPPED -- reusing existing frozen source for {len(benchmarks)} benchmark(s).")
        frozen_benchmarks = benchmarks
    else:
        print(f"Phase 1: translating source for {len(benchmarks)} benchmark(s) in parallel...")
        frozen_benchmarks = translate_all_sources(benchmarks, endpoint, config, iterations, lock)
    timings["1_translate"] = time.time() - t0; t0 = time.time()

    # 2. Keep only projects whose frozen source builds; record the rest as skipped.
    buildable = []
    for frozen in frozen_benchmarks:
        if frozen.compile_rust():
            print(f"  {frozen.project_name}: frozen source does not build; skipping")
            per_project[frozen.project_name] = {
                s: {"baseline_valid": False, "reason": "source_build_failed"} for s in settings
            }
        else:
            per_project[frozen.project_name] = {}
            buildable.append(frozen)

    # 3. Build the work list. On resume, load already-computed results and skip them.
    #    With reuse_tests, a not-done project that ALREADY has generated tests on disk
    #    skips materialise + test-gen + test-repair and goes straight to (prune + Phase 4)
    #    -- the 0-mutant issue was always in Phase 4, never in test-gen/baseline.
    fresh_tasks = []   # need materialise + test-gen + full baseline
    reuse_tasks = []   # already have tests -> prune-only baseline, straight to mutation
    n_resumed = 0
    for frozen in buildable:
        for setting in settings:
            done = _load_done_result(output_dir, frozen.project_name, setting, resume,
                                     split_crates=split_crates)
            if done is not None:
                per_project[frozen.project_name][setting] = done
                n_resumed += 1
                continue
            proj_dst = output_dir / SETTING_DIRS[setting] / frozen.project_name
            bin_dir = proj_dst / "src" / "bin"
            has_tests = bin_dir.exists() and any(bin_dir.glob("*.rs"))
            if reuse_tests and has_tests:
                try:
                    reuse_tasks.append((frozen, setting, Benchmark(str(frozen.c_path), str(proj_dst))))
                    continue
                except Exception as e:
                    print(f"  {frozen.project_name}/{setting}: reuse load failed ({e}); regenerating")
            try:
                sb = materialize_setting_project(frozen, setting, output_dir)
                fresh_tasks.append((frozen, setting, sb))
            except Exception as e:
                print(f"  {frozen.project_name}/{setting}: materialize failed: {e}")
                per_project[frozen.project_name][setting] = {
                    "baseline_valid": False, "reason": "materialize_failed"}
    if resume:
        print(f"Resume: loaded {n_resumed} done; reuse-tests for {len(reuse_tasks)}; "
              f"fresh for {len(fresh_tasks)}.")
    if not fresh_tasks and not reuse_tasks:
        return _aggregate(per_project, settings, output_dir)

    # 4. PARALLEL test generation -- only for FRESH tasks (reuse tasks keep on-disk tests).
    print(f"Phase 2: generating tests for {len(fresh_tasks)} fresh (project,setting) pairs "
          f"({test_workers} workers)...")
    def _gen(t):
        try:
            generate_rust_tests(t[2], t[1], endpoint, config, lock)
        except Exception as e:
            print(f"  test-gen failed {t[0].project_name}/{t[1]}: {e}")
    with ThreadPoolExecutor(max_workers=test_workers) as ex:
        list(ex.map(_gen, fresh_tasks))
    timings["2_testgen"] = time.time() - t0; t0 = time.time()

    # 5. Make every project its own workspace root, carrying only the dependencies it
    #    actually references (serial -> no write race). cargo-mutants copies the whole
    #    enclosing workspace per run, so an umbrella workspace would make each run copy
    #    all ~93 projects and build the union of their deps. See split_manifests.py.
    for setting in settings:
        make_standalone(output_dir / SETTING_DIRS[setting])

    all_tasks = fresh_tasks + reuse_tasks
    # 5b. Split layout: rebuild each project as core + one test crate per setting, so the
    #     mutated crate no longer contains any test code. Serial: it rewrites manifests.
    workspaces, suites = {}, {}
    if split_crates:
        by_project = {}
        for frozen, setting, sb in all_tasks:
            by_project.setdefault(frozen.project_name, (frozen, {}))[1][setting] = sb
        print(f"Phase 2b: assembling {len(by_project)} split workspace(s) "
              f"(core + one test crate per setting)...")
        offenders = 0
        for name, (frozen, sbs) in sorted(by_project.items()):
            ws = output_dir / "_workspaces" / name
            try:
                info = build_workspace(
                    name, frozen.rust_path,
                    {s: sb.rust_path / "src" / "bin" for s, sb in sbs.items()}, ws)
            except Exception as e:
                print(f"  {name}: workspace assembly failed: {e}")
                continue
            workspaces[name] = ws
            for setting, crate in info["tests"].items():
                suites[(name, setting)] = TestSuite(
                    crate / "tests", ws, package=TEST_CRATE_NAMES[setting], kind="test")
            for setting, path, where in info["api_offenders"]:
                print(f"  {name}/{setting}: {Path(path).name} reaches past the public API "
                      f"({where}); it will be repaired or pruned by the baseline")
                offenders += 1
        if offenders:
            print(f"  {offenders} test file(s) use crate internals and may not survive the split.")

    # Shared pool of isolated cargo target dirs, reused across Phase 3 (baselines) and
    # Phase 4 (mutation). One dir per concurrent worker so concurrent same-crate builds
    # never clobber each other's rlib. Sized to the max concurrency of either phase.
    #
    # Under the split layout the pool is partitioned PER SETTING, because all four
    # settings of a project now mutate the very same workspace path. Two settings sharing
    # a target dir then look identical to cargo, which reuses the previous setting's test
    # binaries instead of relinking them against the mutated core -- every mutant survives
    # and the setting scores 0%. Measured on aes128_SIMD (shard 1/12): a shared target
    # gives 17/19 for the first setting and 0/23 for each of the next three; one target
    # per setting gives 17/19 for all four, with the same viable count. Different projects
    # may still share a dir -- their package names differ, so their artifacts do not alias.
    # ``shared_mutants`` makes this moot: there is one mutation run per PROJECT rather than
    # one per (project, setting), so no two settings can share a target dir in the first
    # place and the pool goes back to being flat.
    n_mw = max(1, mutation_workers)
    pool_size = max(test_workers, n_mw)
    partition_pool = split_crates and not shared_mutants
    per_setting = max(2, math.ceil(pool_size / len(settings))) if partition_pool else pool_size

    def _new_pool(suffix, size):
        pool = queue.Queue()
        for i in range(size):
            td = output_dir / f"_mut_target_{suffix}{i}"
            td.mkdir(parents=True, exist_ok=True)
            pool.put(str(td))
        return pool

    if partition_pool:
        target_pools = {s: _new_pool(f"{SETTING_DIRS[s]}_", per_setting) for s in settings}
        _pool = lambda setting: target_pools[setting]
    else:
        target_pool = _new_pool("", pool_size)
        _pool = lambda setting: target_pool
    bl_job = max(2, jobs // max(1, test_workers))  # per-build CPU cap for baselines

    # 6. PARALLEL green baseline. Fresh tasks: repair (LLM) + prune. Reuse tasks:
    #    prune-only (repair_iterations=0), since their tests were already made green.
    print(f"Phase 3: baselines -- {len(fresh_tasks)} with repair, {len(reuse_tasks)} prune-only "
          f"({test_workers} workers, own target dirs, CARGO_BUILD_JOBS={bl_job})...")
    def _baseline(t, iters):
        # Under the split layout the baseline runs per TEST CRATE: a test that fails to
        # compile now breaks its own crate instead of marking every mutant unviable.
        suite = suites.get((t[0].project_name, t[1])) if split_crates else None
        if split_crates and suite is None:
            return (0, 0)  # workspace assembly failed or the setting had no tests
        pool = _pool(t[1])
        td = pool.get()
        try:
            kept, total = ensure_green_baseline(t[2], endpoint, config, lock, iters,
                                                target_dir=td, jobs=bl_job, suite=suite)
            if suite is not None:
                # Mirror the surviving tests back to the setting copy's src/bin, which
                # stays the store of record for --reuse_tests on later runs.
                _sync_tests_back(suite.dir, t[2].rust_path / "src" / "bin")
            return kept, total
        except Exception as e:
            print(f"  baseline failed {t[0].project_name}/{t[1]}: {e}")
            return (0, 0)
        finally:
            pool.put(td)
    with ThreadPoolExecutor(max_workers=test_workers) as ex:
        fresh_bl = list(ex.map(lambda t: _baseline(t, repair_iterations), fresh_tasks))
        reuse_bl = list(ex.map(lambda t: _baseline(t, 0), reuse_tasks))
    timings["3_baseline"] = time.time() - t0; t0 = time.time()

    # 7. PARALLEL mutation testing: run `mutation_workers` projects at once, each in its
    #    OWN cargo target dir (a small reused pool) so concurrent cargo-mutants runs don't
    #    fight over one target's build lock. Smallest projects first.
    def _src_size(sb):
        return sum(len(f.get("content", "")) for f in sb.rust_files)
    ordered = sorted(list(zip(fresh_tasks, fresh_bl)) + list(zip(reuse_tasks, reuse_bl)),
                     key=lambda tb: _src_size(tb[0][2]))

    per_job = max(2, jobs // n_mw)  # split cores across the concurrent runs
    # reuse the same isolated target-dir pool built before Phase 3 (already warm).

    # Pre-strip dangling Cargo.toml targets SERIALLY so no manifest is rewritten during
    # the concurrent phase. cargo-mutants copies the whole enclosing workspace per run;
    # a manifest rewritten mid-copy corrupts a sibling run's tree copy. _strip is
    # idempotent, so the (no-op) call inside mutate_with_cargo stays harmless afterwards.
    # Under the split layout the manifests are generated fresh (no seeded [[test]] blocks)
    # and the test files live in the test crates, so it applies to those instead.
    for (fr, st, s), _ in ordered:
        _strip_dangling_targets(s.rust_path)
        if split_crates and (fr.project_name, st) in suites:
            _strip_dangling_targets(suites[(fr.project_name, st)].dir.parent)

    # Count the mutants ONCE per project, serially, and reuse that count for all four
    # settings. Sharding is derived from it, so an identical count means an identical
    # shard -- the four settings then run byte-identical mutant subsets.
    shared_counts = {}
    if split_crates:
        env = os.environ.copy()
        for name, ws in sorted(workspaces.items()):
            shared_counts[name] = _count_mutants(ws, env=env, package=name)

    how = ("one shared build per mutant, all suites against it (mutate_shared)"
           if shared_mutants else "cargo-mutants")
    print(f"Phase 4: running {how} ({n_mw} projects concurrent, each -j1 serial "
          f"with CARGO_BUILD_JOBS={per_job}, own target dirs + external output, small "
          f"projects first, per-mutant timeout = {timeout_mult}x baseline)...")

    def _mutate(item):
        (frozen, setting, sb), (kept, total) = item
        pool = _pool(setting)
        td = pool.get()
        # cargo-mutants output goes OUTSIDE the workspace tree (see mutate_with_cargo).
        out_base = output_dir / "_mutants_out" / SETTING_DIRS[setting] / frozen.project_name
        split = {}
        if split_crates:
            ws = workspaces.get(frozen.project_name)
            suite = suites.get((frozen.project_name, setting))
            if ws is None or suite is None:
                pool.put(td)
                return (frozen.project_name, setting,
                        {"baseline_valid": False, "reason": "no_split_workspace"})
            split = {
                "proj_dir": ws,
                "package": frozen.project_name,          # the `core` crate keeps this name
                "test_package": suite.package,
                "metadata_dir": ws / "metadata" / SETTING_DIRS[setting],
                "n_mutants": shared_counts.get(frozen.project_name),
            }
        try:
            summary = mutate_with_cargo(sb, kept, total, shard=shard, jobs=per_job,
                                        big_threshold=big_threshold, big_shard_target=big_shard_target,
                                        timeout_mult=timeout_mult, target_dir=td, out_base=out_base,
                                        **split)
        except Exception as e:
            print(f"  cargo-mutants failed {frozen.project_name}/{setting}: {e}")
            summary = {"baseline_valid": False, "reason": "mutation_error"}
        finally:
            pool.put(td)
        return (frozen.project_name, setting, summary)

    def _mutate_shared(item):
        """One mutation run for a whole PROJECT: apply each mutant once, test every suite.

        Returns the same (project, setting, summary) triples as ``_mutate`` so the two
        paths are interchangeable from here on, and writes the same per-setting
        ``metadata/<setting_dir>/mutation.json`` so ``--resume`` keeps working.
        """
        pname, group = item
        ws = workspaces.get(pname)
        if ws is None:
            return [(pname, s, {"baseline_valid": False, "reason": "no_split_workspace"})
                    for (_, s, _), _ in group]

        # Only settings that kept a passing test can run; the rest are recorded as such.
        crates, meta, out = {}, {}, []
        for (frozen, setting, sb), (kept, total) in group:
            suite = suites.get((pname, setting))
            if suite is None or kept == 0:
                out.append((pname, setting, {
                    "project": pname, "baseline_valid": False,
                    "reason": "no_passing_tests" if suite is not None else "no_split_workspace",
                    "tests_kept": kept, "tests_total": total}))
                continue
            crates[setting] = suite.package
            meta[setting] = (kept, total)
        if not crates:
            return out

        n_mutants = shared_counts.get(pname)
        applied_shard, sharded = _decide_shard(n_mutants, shard, big_threshold,
                                               big_shard_target, project_name=pname)
        td = target_pool.get()
        try:
            res = run_shared_mutation(
                ws, package=pname, test_crates=crates,
                out_base=output_dir / "_mutants_out", project_name=pname,
                target_dir=td, jobs=per_job, shard=applied_shard, timeout_mult=timeout_mult,
                min_timeout=min_test_timeout)
        except Exception as e:
            print(f"  shared mutation failed {pname}: {e}")
            return out + [(pname, s, {"baseline_valid": False, "reason": "mutation_error"})
                          for s in crates]
        finally:
            target_pool.put(td)

        for setting, summary in res["settings"].items():
            kept, total = meta[setting]
            summary = {**summary, "project": pname, "tests_kept": kept, "tests_total": total,
                       "sharded": sharded, "shard": applied_shard, "package": pname,
                       "test_package": crates[setting], "shared_mutants": True}
            md = ws / "metadata" / SETTING_DIRS[setting]
            md.mkdir(parents=True, exist_ok=True)
            (md / "mutation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            out.append((pname, setting, summary))
        # A setting whose baseline went red is dropped by run_shared_mutation, not scored.
        for setting in crates:
            if setting not in res["settings"]:
                out.append((pname, setting, {
                    "project": pname, "baseline_valid": False,
                    "reason": "test_package_baseline_failed",
                    "tests_kept": meta[setting][0], "tests_total": meta[setting][1]}))
        return out

    if shared_mutants:
        # Regroup by project, keeping the smallest-source-first order of `ordered`.
        grouped = {}
        for item in ordered:
            grouped.setdefault(item[0][0].project_name, []).append(item)
        with ThreadPoolExecutor(max_workers=n_mw) as ex:
            for rows in ex.map(_mutate_shared, list(grouped.items())):
                for pname, setting, summary in rows:
                    per_project[pname][setting] = summary
    else:
        with ThreadPoolExecutor(max_workers=n_mw) as ex:
            for pname, setting, summary in ex.map(_mutate, ordered):
                per_project[pname][setting] = summary or {"baseline_valid": False, "reason": "tool_unavailable"}
    timings["4_mutation"] = time.time() - t0

    print("\n=== phase timings (s) ===")
    for k, v in timings.items():
        print(f"  {k:14} {v:8.1f}")
    print(f"  {'TOTAL':14} {sum(timings.values()):8.1f}")

    return _aggregate(per_project, settings, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compare test-generation modalities via mutation testing on a frozen Rust source."
    )
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Path to the CBench directory")
    parser.add_argument("--rust_dir", type=str, required=False, help="Path to the RBench directory (provides Rust interfaces)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument(
        "--setting", type=str, default="all",
        choices=["all", "src", "src+test", "src/test", "c+rust"],
        help="Which modality to run ('all' compares every setting; default: all)",
    )
    parser.add_argument("--endpoint", type=str, required=True, help="Model endpoint (e.g. claude, gpt-4o, gpt-5.4)")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to the endpoint config JSON")
    parser.add_argument("--iterations", type=int, default=5, help="Number of source-repair iterations")
    parser.add_argument("--test_repair_iterations", type=int, default=3, help="Number of test-repair rounds for the baseline")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N benchmarks")
    parser.add_argument("--shard", type=str, default=None, help="fixed cargo-mutants shard 'k/n' applied to ALL projects (overrides adaptive sharding)")
    parser.add_argument("--jobs", type=int, default=16, help="cargo-mutants parallel jobs (mutants tested concurrently)")
    parser.add_argument("--test_workers", type=int, default=8, help="concurrent workers for test-gen and test-repair LLM calls")
    parser.add_argument("--big_threshold", type=int, default=300, help="projects with more than this many mutants get a moderate shard (0 = never shard)")
    parser.add_argument("--big_shard_target", type=int, default=150, help="target number of mutants to run for a 'big' project")
    parser.add_argument("--timeout_mult", type=float, default=2.0, help="per-mutant test timeout = this multiple of the baseline test time (cargo-mutants --timeout-multiplier; dynamic, never caps the baseline)")
    parser.add_argument("--min_test_timeout", type=float, default=20, help="floor on the per-mutant test timeout in seconds (shared-mutant path only); raise it when a sub-second baseline drives the 20s floor and slow-but-terminating mutants are being recorded as timeouts")
    parser.add_argument("--resume", action="store_true", help="skip (project,setting) pairs that already have a definitive mutation.json")
    parser.add_argument("--skip_translate", action="store_true", help="reuse the frozen source already on disk (skip re-seeding + re-transpiling Phase 1)")
    parser.add_argument("--mutation_workers", type=int, default=4, help="number of projects to mutation-test concurrently (each in its own target dir)")
    parser.add_argument("--reuse_tests", action="store_true", help="reuse tests already on disk (skip re-materialise + test-gen + test-repair) and go straight to mutation")
    parser.add_argument("--split_crates", action="store_true", help="move each setting's tests out of the mutated crate into their own crate (one workspace per project: core + tests-<setting>), so every setting scores the same mutant set over the same denominator")
    parser.add_argument("--shared_mutants", action="store_true", help="requires --split_crates: apply each mutant once and run all four suites against that single build, instead of one cargo-mutants run per setting (faster, and makes viability -- hence the denominator -- shared by construction)")
    parser.add_argument("--single_benchmark", type=str, default=None, help="Run a single benchmark only")
    args = parser.parse_args()

    config = endpoint_resolver(args.config, args.endpoint)
    settings = SETTINGS if args.setting == "all" else [args.setting]

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Share ONE cargo target dir across every workspace/project (frozen + all settings +
    # cargo-mutants). Otherwise each of the ~28 heavy shared deps (openssl, nix, magic,
    # regex, ...) gets recompiled from scratch per workspace. With a shared target the
    # dependency tree builds once and everything reuses the cached artifacts.
    shared_target = (output_dir / "_shared_target").resolve()
    shared_target.mkdir(parents=True, exist_ok=True)
    os.environ["CARGO_TARGET_DIR"] = str(shared_target)
    # Disable rustc incremental compilation everywhere. When several cargo builds reuse a
    # target dir, the per-crate incremental cache races and rustc fails with
    # "failed to create encoded metadata from file: No such file or directory" -- which
    # cargo-mutants then mis-reports as an *Unviable* mutant (a type-filtered mutant),
    # inflating the unviable count with build-infra failures. Incremental gives no benefit
    # in mutation testing (each mutant is a one-shot crate state); dependency .rlib reuse
    # -- the real speedup -- is a separate mechanism and is unaffected.
    os.environ["CARGO_INCREMENTAL"] = "0"
    print(f"Using shared CARGO_TARGET_DIR: {shared_target} (CARGO_INCREMENTAL=0)")

    # The frozen source lives in its own subtree; per-setting copies live under output_dir/<setting>.
    frozen_dir = output_dir / "_frozen_source"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    # Select only the projects we intend to run, so we never create/seed/compile all 100.
    selected = select_projects(benchmark_dir, args.rust_dir, args.single_benchmark, args.limit)
    if not selected:
        print("No matching benchmarks found.")
        return

    def _build_benchmarks(names):
        """Construct Benchmarks, skipping any project that fails (never fatal)."""
        out = []
        for p in names:
            try:
                out.append(Benchmark(str(benchmark_dir / p), str(frozen_dir / p)))
            except Exception as e:
                print(f"  skipping {p}: cannot construct benchmark ({e})")
        return out

    # Construct Benchmarks only for the selected projects. With --skip_translate we load
    # the frozen source already on disk (from a prior run) and do NOT re-seed/re-cargo-new.
    benchmarks = _build_benchmarks(selected)

    # Seed the selected projects with the Rust interfaces from RBench (same as src/run.py).
    if args.rust_dir and not args.skip_translate:
        seeded = []
        for benchmark in benchmarks:
            src = _rbench_path(args.rust_dir, benchmark.project_name)
            if src is None:
                continue
            try:
                if benchmark.rust_path.exists():
                    shutil.rmtree(benchmark.rust_path)
                shutil.copytree(src, benchmark.rust_path, ignore=_ignore_uncopyable)
                seeded.append(benchmark.c_path.name)
            except Exception as e:
                print(f"  skipping {benchmark.project_name}: seeding failed ({e})")
        # reload so rust_files / rust_headers reflect the seeded interfaces.
        benchmarks = _build_benchmarks(seeded)

    print(f"Comparing settings {settings} over {len(benchmarks)} benchmark(s): "
          f"{[b.project_name for b in benchmarks]}")
    run_experiment(
        benchmarks, output_dir, args.endpoint, config, args.iterations,
        settings, args.test_repair_iterations, args.shard, args.jobs, args.test_workers,
        resume=args.resume, big_threshold=args.big_threshold, big_shard_target=args.big_shard_target,
        timeout_mult=args.timeout_mult, skip_translate=args.skip_translate,
        mutation_workers=args.mutation_workers, reuse_tests=args.reuse_tests,
        split_crates=args.split_crates,
        shared_mutants=args.shared_mutants,
        min_test_timeout=args.min_test_timeout,
    )


if __name__ == "__main__":
    main()
