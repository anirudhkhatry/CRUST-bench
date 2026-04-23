# Objective

Generate a Rust guardrail harness at `rust/src/bin/generated_guardrails.rs` that captures the public contract of the C project in `c/` and is self-consistent against an oracle built from the C code itself. This file will be a fixed contract for a later translation agent; it must be correct before you terminate.

# Inputs available to you

- `c/` — the C source and headers you are building a guardrail for.
- `rust/src/interfaces/*.rs` — the authoritative entry-point signatures (pub fn / pub struct / pub enum / pub const). These are the contract surface.
- `rust/Cargo.toml` — the crate definition. Do not modify.
- You may write scratch files anywhere under `stage_a_workspace/` at the benchmark root. A downstream step will delete that directory wholesale.

# Do not modify

- `c/` — the C source is ground truth.
- `rust/src/interfaces/` — interface definitions are the fixed surface.
- `rust/Cargo.toml` — crate configuration is out of scope.

# Required process (follow in order)

## 1. Interface anchoring

Read every `*.rs` file under `rust/src/interfaces/`. Extract every `pub fn`, `pub struct`, `pub enum`, and `pub const`. This set is the non-negotiable `INTERFACE_FNS` / `INTERFACE_TYPES` for your guardrail. You do not get to decide which interface symbols are "important" — the interfaces file decides for you. Every `pub fn` in an interfaces file must either be covered by a `validate_semantics_*` probe OR be listed in an explicit exclusion set with a one-line reason.

## 2. C contract reading

For each interface `pub fn`, locate the corresponding C implementation in `c/`. Record its observable contract: return value, output buffers that cross the public API boundary (parameters the caller passes in and inspects afterward), and caller-visible state mutations.

You may assert on:
- return values,
- the contents of output buffers that the public C API writes to,
- state the caller can observe through subsequent public API calls.

You may NOT assert on:
- byte layouts of buffers consumed only internally within the library,
- intermediate tables, key schedules, or other plumbing not exposed by the C header,
- ordering of writes to internal scratch memory.

Test internal structures transitively through their consumers. If the C header does not expose a getter for a piece of state, the guardrail does not get to make claims about its layout.

## 3. Build the C and capture authoritative fixtures

Build the C project. Try in order:
1. `make` at the benchmark root (if a Makefile exists).
2. `cc c/*.c -o stage_a_workspace/oracle_bin` (direct compile, for single-file projects).
3. A one-off build script you write at `stage_a_workspace/build.sh` — use this for projects with custom layouts or multiple TUs.

Then write a small C driver at `stage_a_workspace/driver.c` (or similar) that:
- calls each interface function over a deterministic seed list (hardcoded inputs — NOT random per run, so reruns are reproducible),
- prints each `input → output` pair, with byte buffers hex-encoded.

Run the driver. Capture its stdout. These `(input, output)` pairs are the **authoritative fixtures**. Bake them as literals into your Rust guardrail's `validate_semantics_*` probes — compare crate outputs directly against these captured values. Do NOT write a synthesised Rust reference implementation and compare the crate against it; the C is the reference.

**If the C cannot be built** after trying the three approaches above (missing system deps, exotic toolchain, non-portable code), skip the oracle-derived fixtures, add a leading comment inside the guardrail file documenting the limitation (one line), and still satisfy gates A, B, D below. Do not fabricate fixtures.

## 4. Write the guardrail file

Write `rust/src/bin/generated_guardrails.rs` containing:

- `pub const BEHAVIORAL_GUARDRAILS: &[&str] = &[...]` — human-readable behavioral rules.
- `pub const SYNTACTIC_GUARDRAILS: &[&str] = &[...]` — human-readable syntactic rules.
- Authoritative fixtures baked in as `const` byte arrays / literals.
- One `pub fn validate_semantics_<name>()` per covered interface fn; each probe calls the crate's public API with a fixture input and `assert_eq!`s the result against the fixture output.
- `pub fn header_required_functions() -> &'static [&'static str]` — returns the list of every `pub fn` declared in `rust/src/interfaces/`.
- `pub fn semantics_covered_public_functions() -> &'static [&'static str]` — returns the list of fns that have a `validate_semantics_*` probe.
- `pub const UNREACHABLE_FROM_PUBLIC: &[&str]` and/or `pub const DISPATCH_ONLY: &[&str]` — explicit exclusion lists. Every excluded fn must have a one-line justification comment on the line above.
- `pub fn validate_guardrail_completeness()` — asserts `INTERFACE_FNS ⊆ covered ∪ DISPATCH_ONLY ∪ UNREACHABLE_FROM_PUBLIC`.
- `#[cfg(test)]` tests that call the `validate_*` functions so `cargo test --bin generated_guardrails` exercises them.
- `fn main() {}` — harmless, keeps the bin compilable standalone.

## 5. Self-check (hard gates — do not terminate until all pass)

**Gate A — cargo check compiles.**
Run from `rust/`:
```
cargo check --bin generated_guardrails
```
If this fails, fix the guardrail file until it compiles. Do not terminate with a non-compiling guardrail.

**Gate B — coherence.**
Verify: `INTERFACE_FNS ⊆ semantics_covered ∪ DISPATCH_ONLY ∪ UNREACHABLE_FROM_PUBLIC`. The `validate_guardrail_completeness()` function encodes this, but you should check it by eye too: if `header_required_functions()` lists `ft_printf`, then either `semantics_covered_public_functions()` contains `ft_printf`, or `DISPATCH_ONLY` / `UNREACHABLE_FROM_PUBLIC` contains it. No silent drops.

**Gate C — harness not at fault.**
Run from `rust/`:
```
cargo test --bin generated_guardrails -- --nocapture
```
Interpret the results:
- All tests pass → continue.
- A test panics with the frame pointing at `src/bin/generated_guardrails.rs` AND both sides of the failing `assert_eq!` can be reduced without calling into the crate (e.g., `ref_foo(input)` compared against a literal constant, both of which live in the guardrail file) → the harness itself is buggy. Fix the guardrail and re-run. Do NOT terminate with a buggy harness.
- A test panics with the panic originating from `unimplemented!()` or `todo!()` inside the crate under test (i.e., the unimplemented interface stubs) → expected at this stage. Log which probes hit unimplemented stubs and continue.
- Any other panic → inspect. If the guardrail is asking the crate for something the crate's stubs can't yet provide, that's OK. If the guardrail is asking for something the C code does NOT produce, the guardrail is wrong — fix it.

**Gate D — no synthesised-reference traps.**
If any `validate_semantics_*` probe compares the output of a Rust helper `ref_*` against a literal fixture, and that Rust helper was written by you rather than produced from C-oracle output, delete the helper. The C oracle output IS the reference; a hand-written Rust reference is a second point of failure.

## 6. Terminate

Once gates A–D pass, terminate. The guardrail file is ready for the translation agent.

If you have exhausted reasonable effort and cannot satisfy one of the gates (e.g., C cannot be built and gate C still fails for non-stub reasons), leave a leading comment in the guardrail file documenting which gates could not be satisfied and why, and terminate anyway. A human will review.

# Completion criteria

- `rust/src/bin/generated_guardrails.rs` exists and is non-empty.
- Gate A: `cargo check --bin generated_guardrails` passes.
- Gate B: coherence holds (`INTERFACE_FNS ⊆ covered ∪ excluded`).
- Gate C: no harness-internal panics from `cargo test --bin generated_guardrails`.
- Gate D: no synthesised `ref_*` comparisons against literal fixtures.
- `c/`, `rust/src/interfaces/`, and `rust/Cargo.toml` are unmodified.
- All intermediate build artifacts and scratch files live under `stage_a_workspace/`.
