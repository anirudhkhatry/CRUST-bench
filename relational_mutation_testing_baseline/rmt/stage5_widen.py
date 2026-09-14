"""Stage 5 -- widen each killing test with a differential oracle from the C original.

A mutation-killing test is narrow by construction: it pins the one value that separates
P from P'. Independent re-implementations fail the benchmark suite on *other* things --
aggregate properties, whole-workflow scenarios, behaviour that only appears past a few
dozen elements. This stage makes every accepted test also carry a wide check:

  1. the model writes a C *probe* that drives a realistic workload through the public
     API and records observations (`obs_ll/obs_dbl/obs_str`);
  2. the probe runs on the original C library and its output is frozen into a table;
  3. the model translates the probe into a Rust twin that records the same labels;
  4. the twin is checked against the frozen table on the reference M, mechanically
     dropping observations that pin C-incidental behaviour (they mismatch on M or on
     other ground-truth-passing implementations), within a budget;
  5. the twin is appended to the mutant's `test.rs` as `mod rmt_wide`.

The oracle is the C program, not the model: every expected value was *observed*, and
the model never sees an implementation. The narrow kill is untouched, so a mutant is
never lost here -- when widening fails the test stays as it was.
"""
import json
import re

from . import rbench as R
from .llm import ask, fenced, Refused
from .prompts import load as load_prompt

SYSTEM_C = ("You are a precise C programmer writing characterisation tests. You "
            "reason about concrete values and follow output formats exactly.")
SYSTEM_RS = ("You are a precise Rust programmer translating C tests to Rust. You preserve "
             "every record label and use only APIs you have been shown.")
# Paraphrases tried in turn when the API's safety layer refuses a request. The refusal
# is a classifier verdict on the whole (system, user) pair and is partly stochastic, so a
# reworded system line is retried once before the attempt is written off.
SYSTEM_RS_ALTS = ("You are a senior Rust engineer.",
                  "You are a helpful assistant.",
                  "You are a precise Rust programmer translating C tests to Rust. You preserve "
                  "every record label and use only APIs you have been shown.")
SYSTEM_C_ALTS = ("You are a senior C engineer.",
                 "You are a helpful assistant.",
                 "You are a precise C programmer writing characterisation tests. You "
                 "reason about concrete values and follow output formats exactly.")
MAX_REFUSALS = 4

MIN_OBS, MAX_OBS = 20, 600
LABEL_RE = re.compile(r"^[ -~]+$")          # printable ASCII, no tabs
OBS_LINE = re.compile(r"^RMTOBS\t([^\t]*)\t([LDS])\t(.*)$")
FAIL_RE = re.compile(r"^OBS_(MISMATCH|KIND|UNKNOWN|MISSING) (.+?)(?::| got|$)", re.M)
SKIP_FRACTION = 0.20

C_PRELUDE = r'''
/* --- rmt observation prelude (record mode) --- */
#include <stdio.h>
static void obs_ll(const char *label, long long v) { printf("RMTOBS\t%s\tL\t%lld\n", label, v); }
static void obs_dbl(const char *label, double v) { printf("RMTOBS\t%s\tD\t%.17g\n", label, v); }
static void obs_str(const char *label, const char *s) {
    printf("RMTOBS\t%s\tS\t", label);
    if (!s) { printf("<NULL>\n"); return; }
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c >= 32 && c < 127 && c != '\\') putchar(c); else printf("\\x%02x", c);
    }
    putchar('\n');
}
/* --- end prelude --- */
'''

RS_CHECKER = r'''
    // --- rmt observation checker (appended mechanically; expected values were recorded
    //     by running the C probe above on the original C library) ---
    #[allow(dead_code)]
    #[derive(Debug)]
    enum E { L(i64), D(f64), S(&'static str) }
    static EXPECTED: &[(&str, E)] = &[
<<TABLE>>
    ];
    static SKIP: &[&str] = &[<<SKIP>>];
    thread_local! {
        static SEEN: std::cell::RefCell<Vec<&'static str>> = std::cell::RefCell::new(Vec::new());
        static FAILS: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    }
    fn esc(s: &str) -> String {
        let mut o = String::new();
        for b in s.bytes() {
            if b >= 32 && b < 127 && b != b'\\' { o.push(b as char); }
            else { o.push_str(&format!("\\x{:02x}", b)); }
        }
        o
    }
    fn find(label: &str) -> Option<&'static (&'static str, E)> {
        EXPECTED.iter().find(|(l, _)| *l == label)
    }
    fn fail(msg: String) { FAILS.with(|f| f.borrow_mut().push(msg)); }
    fn seen(label: &str) -> bool {
        match find(label) {
            Some((l, _)) => { SEEN.with(|s| s.borrow_mut().push(*l)); true }
            None => { if !SKIP.contains(&label) { fail(format!("OBS_UNKNOWN {}", label)); } false }
        }
    }
    #[allow(dead_code)]
    pub fn obs_ll(label: &str, v: i64) {
        if !seen(label) || SKIP.contains(&label) { return; }
        match find(label) {
            Some((_, E::L(e))) if *e == v => {}
            Some((_, E::L(e))) => fail(format!("OBS_MISMATCH {}: expected {}, got {}", label, e, v)),
            Some((_, other)) => fail(format!("OBS_KIND {}: expected {:?}, got L({})", label, other, v)),
            None => {}
        }
    }
    #[allow(dead_code)]
    pub fn obs_dbl(label: &str, v: f64) {
        if !seen(label) || SKIP.contains(&label) { return; }
        match find(label) {
            Some((_, E::D(e))) => {
                let ok = (e.is_nan() && v.is_nan()) || *e == v
                    || (v - e).abs() <= 1e-6 * e.abs().max(1.0);
                if !ok { fail(format!("OBS_MISMATCH {}: expected {:?}, got {:?}", label, e, v)); }
            }
            Some((_, E::L(e))) if (*e as f64) == v => {}
            Some((_, other)) => fail(format!("OBS_KIND {}: expected {:?}, got D({:?})", label, other, v)),
            None => {}
        }
    }
    #[allow(dead_code)]
    pub fn obs_str(label: &str, s: &str) {
        if !seen(label) || SKIP.contains(&label) { return; }
        let got = esc(s);
        match find(label) {
            Some((_, E::S(e))) if *e == got => {}
            Some((_, E::S(e))) => fail(format!("OBS_MISMATCH {}: expected {:?}, got {:?}", label, e, got)),
            Some((_, other)) => fail(format!("OBS_KIND {}: expected {:?}, got S({:?})", label, other, got)),
            None => {}
        }
    }
    pub fn obs_done() {
        let seen = SEEN.with(|s| s.borrow().clone());
        for (l, _) in EXPECTED.iter() {
            if !SKIP.contains(l) && !seen.contains(l) { fail(format!("OBS_MISSING {}", l)); }
        }
        let fails = FAILS.with(|f| f.borrow().clone());
        if !fails.is_empty() {
            panic!("{} observation(s) differ from the C original:\n{}", fails.len(),
                   fails.join("\n"));
        }
    }
'''


# ----------------------------------------------------------------------------- C side

def parse_observations(stdout):
    """RMTOBS lines -> [(label, kind, raw)] or raise ValueError."""
    obs, labels = [], set()
    for line in stdout.splitlines():
        m = OBS_LINE.match(line)
        if not m:
            continue
        label, kind, raw = m.groups()
        if not LABEL_RE.match(label):
            raise ValueError(f"label {label!r} is not printable ASCII")
        if label in labels:
            raise ValueError(f"label {label!r} recorded twice; labels must be unique")
        labels.add(label)
        obs.append((label, kind, raw))
    return obs


FOCUS_RE = re.compile(r"## Where to focus\n.*?(?=\n## Hard constraints)", re.S)


def probe(project, mutant_rec, iface_text, source_bundle, kill_test="", retries=2, tag="p"):
    """Write and run the C probe on the original. Returns {probe_c, obs, attempts}.

    The focus section shows the Stage 2 killing test rather than the mutant's semantic
    description: the latter is a prose account of a memory bug, and next to "record what
    the library does" the API's safety layer read the whole request as exploit-writing
    and refused it outright (0 output tokens). If a request is still refused, the focus
    section is dropped for the next attempt; a second refusal ends the probe.
    """
    base = load_prompt("probe", project=project.name, sources=source_bundle,
                       iface=iface_text, function=mutant_rec.get("function", "?"),
                       fault_class=mutant_rec.get("fault_class", "?"),
                       kill_test=(kill_test or "").strip() or "(not available)")
    attempts, prev, last_fb, focus_dropped = [], None, None, False
    systems = [SYSTEM_C, *SYSTEM_C_ALTS]

    def compose():
        if prev is None:
            return base
        return base + "\n\n---\n\n" + load_prompt("probe_repair", feedback=last_fb, prev=prev)

    # Effort drops to medium after a reply with no code: with adaptive thinking the
    # budget is shared between reasoning and text, and a probe that thinks for the
    # whole 32k returns nothing at all (seen on dict and emlang, four attempts each).
    effort, i, refusals = "high", 0, 0
    while i <= retries and refusals <= MAX_REFUSALS:
        try:
            reply = ask(systems[0], compose(), effort=effort, max_tokens=32000,
                        tag=f"prb_{tag}_{i}" + (f"r{refusals}" if refusals else ""))
        except Refused as e:
            refusals += 1
            attempts.append({"attempt": i, "outcome": "refused", "feedback": str(e)[:200]})
            if not focus_dropped:
                focus_dropped = True
                base = FOCUS_RE.sub("## Where to focus\n\nExercise the whole public API, "
                                    f"with particular attention to `{mutant_rec.get('function', '?')}`.\n",
                                    base)
            elif len(systems) > 1:
                systems.pop(0)
            continue
        i += 1
        probe_c = fenced(reply, "c")
        fb = None
        if not probe_c:
            fb = "You returned no fenced C code block."
            effort = "medium"
        elif re.search(r'#\s*include\s*"[^"]*\.c"', probe_c):
            fb = "The probe #includes a .c file; link against the library instead."
        elif re.search(r"\bvoid\s+obs_(ll|dbl|str)\s*\(", probe_c):
            fb = "Do not declare or define obs_ll/obs_dbl/obs_str; they are provided."
        if fb is None:
            r = project.run(C_PRELUDE + "\n" + probe_c, timeout=30, stdout_limit=400_000)
            if not r.ok:
                fb = f"The probe did not {r.stage.upper()}:\n```\n{r.detail[-2500:]}\n```"
            elif r.returncode == 124:
                fb = "The probe TIMED OUT (must run in under 5 seconds)."
            elif r.returncode != 0:
                fb = (f"The probe exited {r.returncode} on the ORIGINAL library; it must "
                      f"exit 0.\n```\n{(r.stdout[-600:] + r.stderr)[-1500:]}\n```")
            else:
                try:
                    obs = parse_observations(r.stdout)
                except ValueError as e:
                    fb = str(e)
                else:
                    if len(obs) < MIN_OBS:
                        fb = (f"Only {len(obs)} observations were recorded; record at "
                              f"least 40 (several scales, boundaries, error paths).")
                    elif len(obs) > MAX_OBS:
                        fb = (f"{len(obs)} observations is too many; keep it under 300 "
                              f"(observe per scenario, not per element of a big loop).")
        attempts.append({"attempt": i, "outcome": "ok" if fb is None else "rejected",
                         "feedback": (fb or "")[:800]})
        if fb is None:
            return {"probe_c": probe_c, "obs": obs, "attempts": attempts}
        prev, last_fb = probe_c or "(no code returned)", fb
    return {"probe_c": None, "obs": [], "attempts": attempts}


# -------------------------------------------------------------------------- Rust side

def _rs_lit(kind, raw):
    if kind == "L":
        return f"E::L({int(raw)})"
    if kind == "D":
        v = float(raw)
        if v != v:
            return "E::D(f64::NAN)"
        if v == float("inf"):
            return "E::D(f64::INFINITY)"
        if v == float("-inf"):
            return "E::D(f64::NEG_INFINITY)"
        s = repr(v)
        if "e" not in s and "." not in s:
            s += ".0"
        return f"E::D({s})"
    return f"E::S({json.dumps(raw)})"


def assemble(twin_rs, obs, skip):
    """Wrap the model's twin and the frozen table into `mod rmt_wide { ... }`."""
    table = "\n".join(f"        ({json.dumps(l)}, {_rs_lit(k, raw)})," for l, k, raw in obs)
    checker = RS_CHECKER.replace("<<TABLE>>", table).replace(
        "<<SKIP>>", ", ".join(json.dumps(s) for s in sorted(skip)))
    body = "\n".join("    " + ln if ln.strip() else ln for ln in twin_rs.splitlines())
    return ("\n\n// ---- differential oracle: the C probe's observations, replayed ----\n"
            "#[allow(unused_imports, unused_variables, unused_mut, dead_code)]\n"
            f"mod rmt_wide {{\n{body}\n{checker}}}\n")


def failed_labels(out):
    return {m.group(2).strip() for m in FAIL_RE.finditer(out)}


UNTRANSLATABLE_RE = re.compile(r"//\s*UNTRANSLATABLE:")


def transpile_twin(project, probe_c, obs, narrow_rs="", retries=3, tag="w"):
    """Model writes the Rust twin; compile-repair loop against the reference.

    Returns {twin_rs, compiles, attempts, n_omitted}. Compiling is checked here, plus
    one push-back if the twin omits more records than `settle` may skip: a twin that
    declares a third of the probe untranslatable would fail settlement for certain, and
    on kd3 the first twins did exactly that for facts the narrow test does reach. The
    value check itself is `settle`'s job.
    """
    mods = R.lib_modules(project) or ["<module>"]
    max_omit = int(SKIP_FRACTION * len(obs))
    base = load_prompt("transpile_wide", probe_c=probe_c, iface=R.iface_source(project),
                       crate=R.crate_name(project), modules=", ".join(mods), module0=mods[0],
                       narrow_rs=(narrow_rs or "").strip() or "(none available)",
                       max_omit=max_omit, n_obs=len(obs))
    attempts, user, twin, systems = [], base, None, [SYSTEM_RS, *SYSTEM_RS_ALTS]
    pushed_back, i, refusals, effort = False, 0, 0, "high"
    # A 300-400 record twin runs to ~10k tokens of Rust on top of the reasoning, and
    # on kd3 four replies in a row came back with no code at 24k. Budget 40k, and
    # after a truncated reply reason less and ask for a tighter file.
    tight = ("\n\nYour previous reply was cut off before the end of the file. Return a "
             "complete file: use helper functions and loops so it is shorter, and put "
             "`obs_done();` last.")
    while i <= retries and refusals <= MAX_REFUSALS:
        try:
            reply = ask(systems[0], user, effort=effort, max_tokens=40000,
                        tag=f"tw_{tag}_{i}" + (f"r{refusals}" if refusals else ""))
        except Refused as e:
            refusals += 1
            attempts.append({"attempt": i, "outcome": "refused", "detail": str(e)[:200]})
            if len(systems) > 1:
                systems.pop(0)
            continue
        i += 1
        twin = fenced(reply, "rust")
        if not twin:
            attempts.append({"attempt": i, "outcome": "no_code"})
            user, effort = base + tight, "medium"
            continue
        if "obs_done()" not in twin:
            attempts.append({"attempt": i, "outcome": "no_obs_done"})
            user, effort = base + tight, "medium"
            continue
        if re.search(r"\bmod\s+rmt_wide\b|\bfn\s+obs_(ll|dbl|str|done)\b", twin):
            attempts.append({"attempt": i, "outcome": "defined_provided_items"})
            user = base + ("\n\n---\n\nDo not define obs_* functions or a `mod rmt_wide` "
                           "wrapper; both are provided. Return just the file body.")
            continue
        res = R.compile_check(project, assemble(twin, obs, set()), tag=f"tw{i}")
        n_omit = len(UNTRANSLATABLE_RE.findall(twin))
        attempts.append({"attempt": i, "outcome": res["status"], "n_omitted": n_omit,
                         "passed": res["passed"], "failed": res["failed"]})
        if res["status"] != "build_error":
            if n_omit > max_omit and not pushed_back:
                pushed_back = True
                user = base + ("\n\n---\n\nYour translation omitted "
                               f"{n_omit} records as UNTRANSLATABLE; at most {max_omit} "
                               "may be omitted. Re-read the working Rust test above for "
                               "how results are reached through this API and translate "
                               "as many of the omitted records as it makes possible. "
                               "Return the complete file.\n\n```rust\n" + twin + "\n```")
                continue
            return {"twin_rs": twin, "compiles": True, "attempts": attempts,
                    "n_omitted": n_omit}
        errs = "\n".join(l for l in res["out"].splitlines()
                         if l.startswith(("error", "  -->")))[-4000:]
        user = base + "\n\n---\n\n" + load_prompt(
            "transpile_repair", errors=errs or res["out"][-3000:], prev=twin)
    return {"twin_rs": twin, "compiles": False, "attempts": attempts,
            "n_omitted": len(UNTRANSLATABLE_RE.findall(twin or ""))}


def panic_excerpt(out, before=100, after=1500):
    """The panic message from a cargo test run, not the cargo chatter after it."""
    i = out.find("panicked at")
    if i < 0:
        return out[-after:]
    return out[max(0, i - before):i + after]


def repair_panic(project, twin_rs, obs, panic_text, retries=1, tag="w"):
    """One model round to stop the twin panicking inside the library.

    On libwecan the reference port panics ("signal length must be in 1..=64") where
    the C library returns a status code, and the twin called it unguarded. The model is
    shown the panic and asked to guard that call (recording the failure the C test
    recorded) or to drop the affected records as UNTRANSLATABLE. Compile-checked; the
    value check is `settle`'s.
    """
    attempts, user, twin = [], None, twin_rs
    base = load_prompt("twin_panic", panic=panic_text, prev=twin_rs)
    user = base
    for i in range(retries + 1):
        try:
            reply = ask(SYSTEM_RS, user, effort="medium", max_tokens=40000,
                        tag=f"twp_{tag}_{i}")
        except Refused as e:
            attempts.append({"attempt": i, "outcome": "refused", "detail": str(e)[:200]})
            break
        cand = fenced(reply, "rust")
        if not cand or "obs_done()" not in cand:
            attempts.append({"attempt": i, "outcome": "no_code"})
            continue
        res = R.compile_check(project, assemble(cand, obs, set()), tag=f"twp{i}")
        attempts.append({"attempt": i, "outcome": res["status"]})
        if res["status"] != "build_error":
            return {"twin_rs": cand, "compiles": True, "attempts": attempts,
                    "n_omitted": len(UNTRANSLATABLE_RE.findall(cand))}
        errs = "\n".join(l for l in res["out"].splitlines()
                         if l.startswith(("error", "  -->")))[-4000:]
        user = base + "\n\n---\n\n" + load_prompt(
            "transpile_repair", errors=errs or res["out"][-3000:], prev=cand)
    return {"twin_rs": twin_rs, "compiles": False, "attempts": attempts,
            "n_omitted": len(UNTRANSLATABLE_RE.findall(twin_rs))}


def settle(project, twin_rs, obs, alternates=(), tag="s"):
    """Decide which observations survive: mechanical, LLM-free.

    Runs the twin on the reference M (and on `alternates`, if any are passed). An
    observation that disagrees pins something the C original does but the Rust
    contract does not promise (iteration order, a capacity, an error code the Rust API
    folds into `Err`), so it is dropped. Dropping is bounded: past SKIP_FRACTION of the
    table the twin is judged mistranslated and the widening is abandoned rather than
    hollowed out.

    The pilot settled against the other suite-passing implementations too, and that
    was wrong: on emlang the only such alternate parses nothing (every program comes
    back as a single `exit`) yet passes the benchmark's tests, and voting with it
    threw away 295 of 399 records. Those are precisely the records this oracle exists
    to keep, so `widen` now passes no alternates.

    Returns {status, skip, budget, votes, wide_rs}.
    """
    budget = max(3, int(len(obs) * SKIP_FRACTION))
    skip, votes = set(), []
    impls = [("reference", R.reference(project))] + [(a.name, a) for a in alternates]
    for rnd in range(3):
        wide = assemble(twin_rs, obs, skip)
        new = set()
        for name, d in impls:
            r = R.run_test(d, wide, tag=f"{tag}_{name[:12]}_{rnd}", keep=400_000)
            lab = failed_labels(r["out"]) if r["status"] == "test_failed" else set()
            votes.append({"round": rnd, "impl": name, "status": r["status"],
                          "n_fail": len(lab), "labels": sorted(lab)[:20]})
            if r["status"] == "test_failed" and not lab:
                # A panic that is not an observation mismatch: the twin itself is broken
                # (an unwrap on None, an index out of range) on this implementation.
                if name == "reference":
                    return {"status": "twin_panics_on_reference", "skip": sorted(skip),
                            "budget": budget, "votes": votes, "wide_rs": None,
                            "out": panic_excerpt(r["out"])}
                continue
            new |= lab
        new -= skip
        if not new:
            ok = all(v["status"] == "green" or v["impl"] != "reference"
                     for v in votes if v["round"] == rnd)
            return {"status": "settled" if ok else "reference_not_green",
                    "skip": sorted(skip), "budget": budget, "votes": votes,
                    "wide_rs": wide if ok else None}
        skip |= new
        if len(skip) > budget:
            return {"status": "too_many_mismatches", "skip": sorted(skip),
                    "budget": budget, "votes": votes, "wide_rs": None}
    return {"status": "did_not_converge", "skip": sorted(skip), "budget": budget,
            "votes": votes, "wide_rs": None}
