"""Score the opus5_sample mutation run and break the results down by mutation type.

Two things this reports that a raw ``outcomes.json`` does not.

Comparable denominators
    Each setting's ``mutation_score`` is computed over that setting's own viable set, and
    viability is setting-dependent (a mutant that fails to compile against one test crate
    may compile against another), so the four per-setting scores are not strictly over the
    same population. Everything cross-config here is therefore also reported on the
    *intersection*: the mutants that are viable and non-timeout in every setting the
    project ran. That is the only apples-to-apples number.

Timeouts
    A timeout is semantically a detection -- the mutant changed behaviour enough to hang
    the suite -- but cargo-mutants reports it as neither caught nor missed, so it sits
    outside both numerator and denominator. Both readings are printed: ``strict``
    (timeouts excluded, matching the stored numbers) and ``folded`` (timeouts counted as
    kills). The gap between them is the size of the interpretive question.

Mutation types
    Grouped by ``genre`` (BinaryOperator / UnaryOperator / FnValue) and, more usefully, by
    the concrete substitution -- for operators the ``op -> op'`` pair recovered from the
    original source span, and for FnValue the replacement value the body was swapped for.
    This is what answers "which mutation types does each modality systematically miss".
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Scripts live one level down; put the study root on sys.path so ``harness`` imports
# work when this file is run directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.paths import RESULTS


# Which run to score. Every run writes the same outcomes.json shape, so two positional
# arguments -- the output directory and, optionally, the results root it sits in --
# switch the whole analysis over:
#     python analyse_opus5_sample.py                                   # cargo-mutants, sample
#     python analyse_opus5_sample.py _transpile_out                    # transpile, sample
#     python analyse_opus5_sample.py _transpile_out results_opus5_4settings   # full set
_ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]

# --valid-only drops the substitutions listed in transpilation_operators.RETIRED_LABELS:
# mutants that no suite passing a green baseline can kill, so their presence in the
# denominator only dilutes the score. The generator no longer emits them; this flag is how
# a run recorded before that change is scored on the same population as one recorded after.
VALID_ONLY = "--valid-only" in sys.argv[1:]

ROOT = RESULTS / "claude" / (_ARGS[1] if len(_ARGS) > 1 else "opus5_sample")
OUT = ROOT / (_ARGS[0] if _ARGS else "_mutants_out")
WS = ROOT / "_workspaces"

SETTINGS = [("src", "src"), ("src+test", "src+test"),
            ("src/test", "src_over_test"), ("c+rust", "c_and_rust")]

# The 5-project sample, in the order its README ranks them. Kept as the display order
# when the run holds exactly those projects; any other run is listed alphabetically.
_SAMPLE = ["ulidgen", "clog", "emlang", "clhash", "cset"]


def _discover():
    found = set()
    for _, sdir in SETTINGS:
        d = OUT / sdir
        if d.is_dir():
            found |= {p.name for p in d.iterdir()
                      if (p / "mutants.out" / "outcomes.json").is_file()}
    if not found:
        return _SAMPLE
    return _SAMPLE if found == set(_SAMPLE) else sorted(found)


PROJECTS = _discover()

CAUGHT, MISSED, TIMEOUT, UNVIABLE = "CaughtMutant", "MissedMutant", "Timeout", "Unviable"


def load(setting_dir, project):
    f = OUT / setting_dir / project / "mutants.out" / "outcomes.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())["outcomes"]


NAME_RE = re.compile(r": replace (.+?) with (.+?)(?: in [A-Za-z0-9_:<>&' ]+)?$")


def mutation_type(project, mut):
    """(genre, label, return_type) where label names the concrete substitution.

    cargo-mutants already spells the substitution out in ``name`` ("replace < with =="),
    which is more reliable than re-reading the span out of the source -- the span columns
    are byte offsets into a file that has since been restored, and for FnValue they cover
    the whole function.
    """
    genre, repl = mut["genre"], (mut["replacement"] or "").strip()
    ret = ((mut.get("function") or {}).get("return_type") or "").strip()
    if genre.startswith("Transpile"):
        # The generator already writes the substitution as the label, after the position:
        # "core/src/cset.rs:224:2: replace wrapping_add with saturating_add". Not every
        # transpilation label is an "X with Y" pair ("transpose arguments 1 and 2"), so
        # take the label whole rather than trying to parse a shape it may not have.
        return genre, mut["name"].split(": ", 1)[-1], ret
    if genre == "FnValue":
        return genre, f"body -> {repl or '<deleted>'}", ret
    m = NAME_RE.search(mut["name"])
    if m:
        orig, new = m.group(1).strip(), m.group(2).strip()
        return genre, f"{orig} -> {new}", ret
    return genre, f"-> {repl[:40] or '<deleted>'}", ret


def main():
    # project -> setting -> {key: (verdict, mut)}
    data = defaultdict(dict)
    for setting, sdir in SETTINGS:
        for p in PROJECTS:
            outs = load(sdir, p)
            if outs is None:
                continue
            data[p][setting] = {
                o["mutant_key"]: (o["summary"], o["scenario"]["Mutant"]) for o in outs}

    if VALID_ONLY:
        # Imported here rather than at module scope so the scorer still runs on an
        # interpreter without tree-sitter when the flag is not used.
        from harness.transpilation_operators import is_retired_label
        dropped = 0
        for p in data:
            for setting, d in data[p].items():
                for key, (_, mut) in list(d.items()):
                    if is_retired_label(mutation_type(p, mut)[1]):
                        del d[key]
                        dropped += 1
        print(f"--valid-only: dropped {dropped} retired mutant-outcomes "
              f"(summed over settings) before scoring\n")

    # ---------------- 1. per-setting scores, as run ----------------
    print("=" * 92)
    print("1. KILL SCORE PER CONFIG (each over its own viable set, as cargo-mutants reports)")
    print("=" * 92)
    print(f"{'project':9} {'config':9} {'caught':>7} {'missed':>7} {'timeout':>8} "
          f"{'unviable':>9} {'strict':>8} {'folded':>8}")
    agg = defaultdict(lambda: defaultdict(int))
    for p in PROJECTS:
        for setting, _ in SETTINGS:
            d = data[p].get(setting)
            if d is None:
                print(f"{p:9} {setting:9}     --  (suite red / not run)")
                continue
            c = sum(1 for v, _ in d.values() if v == CAUGHT)
            m = sum(1 for v, _ in d.values() if v == MISSED)
            t = sum(1 for v, _ in d.values() if v == TIMEOUT)
            u = sum(1 for v, _ in d.values() if v == UNVIABLE)
            strict = c / (c + m) if c + m else 0.0
            folded = (c + t) / (c + m + t) if c + m + t else 0.0
            for k, val in (("c", c), ("m", m), ("t", t), ("u", u)):
                agg[setting][k] += val
            print(f"{p:9} {setting:9} {c:7} {m:7} {t:8} {u:9} "
                  f"{strict:7.1%} {folded:7.1%}")
        print()

    print("-" * 92)
    print(f"POOLED ACROSS ALL {len(PROJECTS)} PROJECTS")
    for setting, _ in SETTINGS:
        a = agg[setting]
        strict = a["c"] / (a["c"] + a["m"]) if a["c"] + a["m"] else 0.0
        folded = (a["c"] + a["t"]) / (a["c"] + a["m"] + a["t"]) if a["c"] + a["m"] + a["t"] else 0
        print(f"{'ALL':9} {setting:9} {a['c']:7} {a['m']:7} {a['t']:8} {a['u']:9} "
              f"{strict:7.1%} {folded:7.1%}")

    # ---------------- 2. intersection: identical denominator ----------------
    print()
    print("=" * 92)
    print("2. KILL SCORE ON THE COMMON MUTANT SET (viable + non-timeout in EVERY config)")
    print("=" * 92)
    inter = {}
    for p in PROJECTS:
        settings = list(data[p])
        keys = None
        for s in settings:
            ok = {k for k, (v, _) in data[p][s].items() if v in (CAUGHT, MISSED)}
            keys = ok if keys is None else (keys & ok)
        inter[p] = (settings, keys or set())

    print(f"{'project':9} {'n':>5}  " + "  ".join(f"{s:>9}" for s, _ in SETTINGS))
    pooled = defaultdict(lambda: [0, 0])
    for p in PROJECTS:
        settings, keys = inter[p]
        row = []
        for s, _ in SETTINGS:
            if s not in settings:
                row.append(f"{'--':>9}")
                continue
            k = sum(1 for key in keys if data[p][s][key][0] == CAUGHT)
            pooled[s][0] += k
            pooled[s][1] += len(keys)
            row.append(f"{k / len(keys):8.1%}" if keys else f"{'n/a':>9}")
        note = "" if len(settings) == 4 else f"  ({len(settings)} configs only)"
        print(f"{p:9} {len(keys):5}  " + "  ".join(row) + note)
    print(f"{'POOLED':9} {'':5}  " + "  ".join(
        f"{pooled[s][0] / pooled[s][1]:8.1%}" if pooled[s][1] else f"{'--':>9}"
        for s, _ in SETTINGS))

    # ---------------- 3. mutation types ----------------
    print()
    print("=" * 92)
    print("3. MUTATION TYPES: kill rate per genre, per config (common mutant set)")
    print("=" * 92)
    genre_tot = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    label_tot = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    label_genre = {}
    for p in PROJECTS:
        settings, keys = inter[p]
        for key in keys:
            _, mut = data[p][settings[0]][key]
            genre, label, _ = mutation_type(p, mut)
            label_genre[label] = genre
            for s in settings:
                k = 1 if data[p][s][key][0] == CAUGHT else 0
                genre_tot[genre][s][0] += k
                genre_tot[genre][s][1] += 1
                label_tot[label][s][0] += k
                label_tot[label][s][1] += 1

    print(f"{'genre':16} {'n':>6}  " + "  ".join(f"{s:>9}" for s, _ in SETTINGS))
    for genre in sorted(genre_tot, key=lambda g: -max(v[1] for v in genre_tot[g].values())):
        n = max(v[1] for v in genre_tot[genre].values())
        row = []
        for s, _ in SETTINGS:
            k, t = genre_tot[genre][s]
            row.append(f"{k / t:8.1%}" if t else f"{'--':>9}")
        print(f"{genre:16} {n:6}  " + "  ".join(row))

    print()
    print("=" * 92)
    print("4. MUTATION TYPES: kill rate per concrete substitution (n >= 8)")
    print("=" * 92)
    print(f"{'substitution':28} {'genre':16} {'n':>5}  " + "  ".join(f"{s:>9}" for s, _ in SETTINGS))
    rows = [(l, max(v[1] for v in d.values())) for l, d in label_tot.items()]
    for label, n in sorted(rows, key=lambda r: -r[1]):
        if n < 8:
            continue
        row = []
        for s, _ in SETTINGS:
            k, t = label_tot[label][s]
            row.append(f"{k / t:8.1%}" if t else f"{'--':>9}")
        print(f"{label[:28]:28} {label_genre[label][:16]:16} {n:5}  " + "  ".join(row))

    print()
    print("=" * 92)
    print("5. BLIND SPOTS: substitutions surviving in EVERY config (n >= 3), common set")
    print("=" * 92)
    blind = []
    for label, d in label_tot.items():
        n = max(v[1] for v in d.values())
        if n < 3:
            continue
        if all(k == 0 for k, t in d.values() if t):
            blind.append((n, label, label_genre[label]))
    for n, label, genre in sorted(blind, reverse=True):
        print(f"  {label[:40]:40} {genre:16} n={n}")
    if not blind:
        print("  (none -- every substitution class is killed by at least one config)")

    # ---------------- 6. what src+test specifically misses ----------------
    print()
    print("=" * 92)
    print("6. THE src+test DEFICIT: mutants killed by src/test but surviving src+test")
    print("=" * 92)
    deficit = defaultdict(int)
    tot = defaultdict(int)
    for p in PROJECTS:
        settings, keys = inter[p]
        if "src+test" not in settings or "src/test" not in settings:
            continue
        for key in keys:
            _, mut = data[p][settings[0]][key]
            _, label, _ = mutation_type(p, mut)
            a = data[p]["src/test"][key][0] == CAUGHT
            b = data[p]["src+test"][key][0] == CAUGHT
            tot[label] += 1
            if a and not b:
                deficit[label] += 1
    print(f"{'substitution':30} {'lost':>6} {'of':>5} {'share':>7}")
    for label, n in sorted(deficit.items(), key=lambda x: -x[1])[:20]:
        print(f"{label[:30]:30} {n:6} {tot[label]:5} {n / tot[label]:6.1%}")
    print(f"{'TOTAL':30} {sum(deficit.values()):6} {sum(tot.values()):5}")


if __name__ == "__main__":
    main()
