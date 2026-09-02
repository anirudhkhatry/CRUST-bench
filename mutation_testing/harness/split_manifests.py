"""Turn each benchmark project into its own standalone cargo workspace.

The pipeline used to write ONE umbrella workspace manifest per setting dir listing
all ~93 projects as members, with every project inheriting every dependency in the
benchmark via ``foo = { workspace = true }``. That costs us twice:

  * cargo-mutants copies the whole *enclosing workspace* into a scratch dir before
    each run, so mutating one 3.4 MB project copied 524 MB of sources (~36 s/run,
    ~350 runs for a full 4-setting sweep).
  * every project compiled the union of all projects' dependencies (openssl, magic,
    nix, syslog, ...) whether or not a single line of its source mentioned them.

This pass rewrites each project's Cargo.toml so it is its own workspace root and
carries only the dependencies its Rust sources actually reference, then drops the
umbrella manifest (a member declaring its own ``[workspace]`` while still listed as
a member of the parent is a hard cargo error).

Dependency detection is deliberately over-approximate: a crate is kept if its ident
(``openssl-sys`` -> ``openssl_sys``) appears as a word anywhere in the project's
.rs files. A false keep costs a little build time; a false drop breaks the build.
Dependencies written with an explicit version rather than ``workspace = true`` are
project-specific and are always preserved verbatim.

Run standalone to (re)apply or undo::

    python split_manifests.py <setting_dir> [...]
    python split_manifests.py --restore <setting_dir> [...]
"""

import re
import shutil
import sys
from pathlib import Path

BACKUP_SUFFIX = ".workspace.bak"

def _load_fallback_versions():
    """The benchmark's shared dependency table, used once no umbrella manifest is left.

    Read straight from the repo's dependencies.json so this module works whether or
    not the CRUST-bench ``src`` package happens to be importable.
    """
    import json
    # harness/ -> mutation_testing/ -> CRUST-bench/
    deps_file = Path(__file__).resolve().parents[2] / "src/resources/cache/dependencies.json"
    try:
        return {k: f'"{v}"' for k, v in json.loads(deps_file.read_text(encoding="utf-8")).items()}
    except Exception:
        try:
            from utils.rust_project_builder import DEPENDENCIES
            return {k: f'"{v}"' for k, v in DEPENDENCIES.items()}
        except Exception:
            return {}


_FALLBACK_VERSIONS = _load_fallback_versions()

# `name = { workspace = true }` -- the inherited deps, the ones we prune.
_INHERITED = re.compile(r'^\s*([A-Za-z0-9_-]+)\s*=\s*\{\s*workspace\s*=\s*true\s*\}\s*$')
# `name = "1.2"` in the umbrella [workspace.dependencies] table.
_VERSIONED = re.compile(r'^\s*([A-Za-z0-9_-]+)\s*=\s*(.+?)\s*$')


def workspace_versions(umbrella):
    """Parse ``[workspace.dependencies]`` out of an umbrella manifest -> {name: rhs}."""
    versions = {}
    in_deps = False
    for line in Path(umbrella).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("["):
            in_deps = s == "[workspace.dependencies]"
            continue
        if not in_deps or not s or s.startswith("#"):
            continue
        m = _VERSIONED.match(s)
        if m:
            versions[m.group(1)] = m.group(2)
    return versions


def used_crates(proj):
    """Set of crate idents (underscored) mentioned anywhere in the project's .rs files."""
    words = set()
    for rs in Path(proj).rglob("*.rs"):
        if "target" in rs.parts or "mutants.out" in rs.parts:
            continue
        try:
            text = rs.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return words


def rewrite_manifest(proj, versions):
    """Make ``proj``'s Cargo.toml a standalone workspace root with pruned deps.

    Returns (kept, dropped) dependency-name lists. The untouched manifest is kept
    alongside as ``Cargo.toml.workspace.bak`` and is the source of truth on every
    later call, so this is idempotent *and* safe to re-run after test generation
    has introduced new crate usage.
    """
    proj = Path(proj)
    manifest = proj / "Cargo.toml"
    backup = manifest.with_suffix(".toml" + BACKUP_SUFFIX)
    if backup.exists():
        source = backup
    elif any(l.strip() == "[workspace]" for l in manifest.read_text(encoding="utf-8").splitlines()):
        return None, None  # already standalone with no backup to re-prune from
    else:
        source = manifest
    lines = source.read_text(encoding="utf-8").splitlines()

    words = used_crates(proj)
    out, kept, dropped = [], [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != "[dependencies]":
            out.append(line)
            i += 1
            continue

        # rewrite the whole [dependencies] section, up to the next top-level table.
        out.append(line)
        i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("["):
            body = lines[i]
            m = _INHERITED.match(body)
            if m is None:
                # explicitly versioned dep (`rand = "0.8"`, `openssl = { ... }`): prune
                # it on the same used/unused test, but keep its line verbatim when kept
                # so pins and feature flags survive. Non-dep lines are dropped.
                explicit = _VERSIONED.match(body) if body.strip() else None
                if explicit:
                    name = explicit.group(1)
                    if name.replace("-", "_") in words:
                        out.append(body)
                        kept.append(name)
                    else:
                        dropped.append(name)
                i += 1
                continue
            name = m.group(1)
            if name.replace("-", "_") in words:
                # inline the version: `workspace = true` is a hard error once this
                # manifest is its own root rather than a member.
                rhs = versions.get(name) or _FALLBACK_VERSIONS.get(name)
                if not rhs:
                    raise KeyError(f"{proj}: no version known for kept dependency {name!r}")
                out.append(f"{name} = {rhs}")
                kept.append(name)
            else:
                dropped.append(name)
            i += 1

    # trailing [workspace] makes this manifest its own workspace root, so cargo (and
    # cargo-mutants' tree copy) stop at the project boundary.
    if out and out[-1].strip():
        out.append("")
    out.append("[workspace]")

    if not backup.exists():
        manifest.replace(backup)
    manifest.write_text("\n".join(out) + "\n", encoding="utf-8")
    return kept, dropped


def make_standalone(setting_dir, verbose=True):
    """Split every project under ``setting_dir`` out of the umbrella workspace."""
    setting_dir = Path(setting_dir)
    umbrella = setting_dir / "Cargo.toml"
    # after the first pass the umbrella is gone; fall back to the benchmark's own
    # dependency table so re-running (e.g. after test generation) still works.
    versions = workspace_versions(umbrella) if umbrella.exists() else dict(_FALLBACK_VERSIONS)

    stats = []
    for proj in sorted(p for p in setting_dir.iterdir() if (p / "Cargo.toml").exists()):
        kept, dropped = rewrite_manifest(proj, versions)
        if kept is None:
            continue
        stats.append((proj.name, len(kept), len(dropped)))
    # the umbrella must go: cargo refuses a member that is also a workspace root.
    if umbrella.exists():
        umbrella.replace(umbrella.with_suffix(".toml" + BACKUP_SUFFIX))

    if verbose and stats:
        total_kept = sum(k for _, k, _ in stats)
        total_drop = sum(d for _, _, d in stats)
        print(f"{setting_dir}: {len(stats)} projects standalone, "
              f"kept {total_kept} deps, dropped {total_drop} "
              f"(avg {total_kept / len(stats):.1f} deps/project, was "
              f"{(total_kept + total_drop) / len(stats):.1f})")
    return stats


def make_independent_packages(src_dir, dst_dir, prune=False, verbose=True):
    """Write a non-workspace copy of a benchmark tree: every project its own root.

    Unlike :func:`make_standalone`, which prunes an *already translated* tree down to
    the crates its sources reference, this builds the pristine benchmark where no
    implementation exists yet. Dependencies are therefore *inlined* rather than
    dropped -- the full set stays available to whatever translation is written later,
    it just no longer arrives via a shared workspace. Pass ``prune=True`` for the lean
    variant that keeps only crates the current sources mention.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    projects = sorted(p for p in src_dir.iterdir() if (p / "Cargo.toml").exists())
    if not projects:
        raise FileNotFoundError(f"no cargo projects under {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for proj in projects:
        dst = dst_dir / proj.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(proj, dst, ignore=shutil.ignore_patterns("target", "mutants.out*"))

        if prune:
            rewrite_manifest(dst, _FALLBACK_VERSIONS)
        else:
            lines = (dst / "Cargo.toml").read_text(encoding="utf-8").splitlines()
            out, seen_deps = [], False
            for line in lines:
                out.append(line)
                if line.strip() == "[dependencies]":
                    seen_deps = True
                    body = "\n".join(lines)
                    out.extend(f"{d} = {v}" for d, v in _FALLBACK_VERSIONS.items()
                               if not re.search(rf'^\s*{re.escape(d)}\s*=', body, re.M))
            if not seen_deps:
                out += ["", "[dependencies]"] + [f"{d} = {v}" for d, v in _FALLBACK_VERSIONS.items()]
            out += ["", "[workspace]"]
            (dst / "Cargo.toml").write_text("\n".join(out) + "\n", encoding="utf-8")
        written += 1

    if verbose:
        print(f"{dst_dir}: {written} independent package(s) from {src_dir} "
              f"({'pruned deps' if prune else 'full deps inlined'})")
    return written


def restore(setting_dir):
    """Undo :func:`make_standalone` from the ``.workspace.bak`` copies."""
    setting_dir = Path(setting_dir)
    n = 0
    for bak in sorted(setting_dir.rglob("Cargo.toml" + BACKUP_SUFFIX)):
        bak.replace(bak.with_name("Cargo.toml"))
        n += 1
    print(f"{setting_dir}: restored {n} manifest(s)")
    return n


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--restore":
        for d in args[1:]:
            restore(d)
    elif args and args[0] == "--independent":
        prune = "--prune" in args
        src, dst = [a for a in args[1:] if not a.startswith("--")][:2]
        make_independent_packages(src, dst, prune=prune)
    else:
        for d in args:
            make_standalone(d)
