# C harness fixes — the 11 stragglers (+2 alias losses)

**Scripts:** `rmt/cbench.py` (the harness), `phase0/recheck.py` (the regression probe).
**Data:** `results/phase0/recheck_before.json|csv`, `recheck_after.json|csv`.

`phase0/recheck.py` drives the *live* harness (`rmt.cbench.Project.run`) with a real test
main that must compile, link and exit 0 — i.e. exactly what Stages 1/2 need — rather than
re-implementing the build the way `charness.py` did.

## Headline

| | before | after |
|---|---|---|
| CBench dirs usable (live harness) | **90 / 100** | **100 / 100** |
| RBench projects usable (`eligible.csv`, after alias fix) | 87 / 100 | **100 / 100** |
| eligible (usable ∧ panel n ≥ 3) | 60 | **69** |
| regressions | — | **0** |

The live-harness baseline is 90, not the 89 in `REPORT.md` §2.3: `XOpt` already passed
because `snprintf.c` carries a `#if TEST_SNPRINTF` `main()`, which the harness treats as a
driver and excludes (`charness.py` compiled mains, so it saw the failure).

## Generic fixes (no per-project data was needed for any of the 11)

1. **Amalgamation sources are not translation units** (`_included_c_files`). A `.c` that
   another library TU `#include`s is dropped from the compile set. Only includes written
   by *non-main* sources and headers count, so a test driver that includes the library
   source (mdb's `testmdb_int.c` → `mdb.c`) cannot disqualify the library.
2. **Compile-flag fallback chain** (`CFLAG_SETS`): plain, then `-march=native`, then
   `+ -std=gnu99`. Tried in order, and the first set that compiles everything is cached
   per project — the 90 already-working projects keep the plain flags.
3. **`-fcommon` in the base flags.** Restores the pre-gcc-10 treatment of tentative
   definitions, which projects shipping several programs from one source tree rely on.
4. **Include shims** (`_write_include_shims`). An `#include` that resolves nowhere on the
   include path but whose name matches a project header — exactly, or after normalising
   punctuation — gets a one-line forwarding header in a scratch `_rmt_shims/` dir inside
   the build copy. Covers install-layout (`<gfc/gfc.h>`) and generated-name
   (`tibs.tsp.h` ← `tibs_tsp.h`) includes. The dataset is never touched.
5. **`--wrap` autodetection** (`_wrap_flags`): a source mentioning `__wrap_X`/`__real_X`
   gets `-Wl,--wrap=X`.
6. **Second-chance link** (`_missing_symbol_sources`). On `undefined reference`, sources
   *outside* the library set that actually **define** one of the missing symbols (and have
   no `main`) are compiled in, then the link is retried (≤ 4 rounds); the discovered set
   is cached. Because a file must define a reported symbol, this cannot pull a demo or
   benchmark TU in wholesale — the definition of "library code" is unchanged for everyone
   else.
7. **Name resolution** (`resolve_name`, `PROJECT_ALIASES`): `Project("proj_2DPartInt")`
   now finds `datasets/CBench/2DPartInt`.

A declarative `PROJECT_FLAGS = {"<cbench dir>": {"cflags", "includes", "ldflags", "libs",
"skip", "sources"}}` override table is in place for future stragglers. **It is currently
empty** — all 11 fell to the generic rules.

## Per-project

| project | failure | root cause | fix |
|---|---|---|---|
| `XOpt` | (already passing) | `xopt.c` does `#include "./snprintf.c"`; the K&R `va_dcl` branch only compiles standalone | fix 1 — `snprintf.c` is no longer a TU |
| `gfc` | compile | `#include <gfc/gfc.h>`, real path `src/gfc.h` (install layout) | fix 4 (shim) |
| `tisp` | compile | `tisp.c` includes `tib/*.c`; `main.c`/`test.c` want the Makefile-generated `tibs.tsp.h` | fixes 1 + 4 (`tibs_tsp.h` shim) |
| `libfor` | compile | `for.c` includes the generated `for-gen.c`, which references file-static symbols | fix 1 |
| `roaring-bitmap` | compile | `_mm_popcnt_u32` `always_inline` target mismatch | fix 2 (`-march=native`) |
| `clhash` | compile | carry-less-multiply intrinsics, same shape | fix 2 |
| `matrix_multiplication` | compile | `_mm256_loadu_ps`, same shape | fix 2 |
| `impcheck` | link | `vectors.c` includes `vec.c` once per type (duplicate symbols); plus tentative `nb_vars` in two TUs | fixes 1 + 3 |
| `jccc` | link | library files (`util/hashmap.c`) call the in-repo assertion framework in `src/testing/`, a `SKIP_DIRS` name | fix 6 |
| `lambda-calculus-eval` | link | **not incomplete** — `search`/`insert`/`table_exists` live in `hash-table/`, a sibling of `src/`, which `_lib_root` excluded | fix 6 |
| `ljmm` | link | `__real_mmap64` needs `-Wl,--wrap=mmap64` | fix 5 |
| `proj_2DPartInt` | alias | CBench dir is `2DPartInt` | fix 7 |
| `proj_42_Kocaeli_Printf` | alias | CBench dir is `42-Kocaeli-Printf` | fix 7 |

All 11 were re-verified with a test main that **calls real library functions** (e.g.
`gfc_encrypt`/`gfc_decrypt` round-trip, `insert`+`table_exists`, `for_compress_sorted`,
`tisp_env_init`, `clhash`), not just the trivial probe.

## Source-set changes

Only three projects' compile sets changed, all amalgamations:
`impcheck` 12 → 11, `libfor` 2 → 1, `tisp` 6 → 1. No project's `header_only` flag flipped,
so Stage 1's mutation targets are unchanged elsewhere.

## Regression check

```
.venv/bin/python phase0/recheck.py --jobs 16 --out recheck_after \
    --baseline results/phase0/recheck_before.json
usable: 100/100
recovered (10): clhash gfc impcheck jccc lambda-calculus-eval libfor ljmm
                matrix_multiplication roaring-bitmap tisp
REGRESSED (0): []
```

`results/phase0/eligible.csv` was regenerated from `recheck_after.json` (previous copy kept
as `eligible_before_alias.csv`): the `cbench` column now carries the resolved directory for
every RBench name, and `c_usable` is True for all 100.

**Caveat.** `-march=native` makes the three SIMD projects' objects host-specific; a mutation
run must stay on one machine class, which it already does.
