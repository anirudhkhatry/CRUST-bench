# Objective

Translate the C project in `c/` into an equivalent safe Rust implementation in `rust/`, satisfying the fixed guardrail contract in `rust/src/bin/generated_guardrails.rs`.

# Required process

1. Inspect and understand the C source files in `c/` before making changes.
2. Read the Rust project structure and the entry point interface definitions in:
   - `rust/`
   - `rust/src/interfaces/`
3. Read the guardrail contract at `rust/src/bin/generated_guardrails.rs`. This file is the fixed contract you must satisfy. You MUST NOT modify it.
4. The interface files are the **entry points** — the functions, structs, enums, and types that the guardrails exercise. Treat them as the primary contract your implementation must satisfy.
5. Implement the required functionality in safe Rust inside `rust/src/`, strictly conforming to the entry point definitions provided for each file in `rust/src/interfaces/`.
6. Each file in `rust/src/interfaces/` should be implemented in `rust/src/` with the same name but without the `unimplemented!()` macro, and instead with a concrete implementation that reproduces the behavior of the C code AND satisfies the guardrail assertions.
7. You may define additional internal types, helper functions, or modules — but the entry point signatures and types must be preserved exactly, and the guardrail file must not be modified.

# Requirements

- Preserve the existing Rust entry point interfaces exactly.
- Implement concrete Rust code in `rust/src/` based on the behavior of the C code.
- Reproduce the C implementation's behavior faithfully.
- Prefer safe, idiomatic Rust abstractions where possible, while preserving correctness.
- Every assertion in `rust/src/bin/generated_guardrails.rs` must pass against your implementation.

# Constraints

- Use only safe Rust.
- Do not use `unsafe`.
- Do not modify:
  - `rust/src/interfaces/` (interface definitions)
  - `rust/src/bin/generated_guardrails.rs` (guardrail contract)
  - source files in `c/`
  - `Cargo.toml`

# Dispute release valve

If, while translating, you become convinced that a specific guardrail assertion contradicts the C semantics — for example, the guardrail asserts on internal plumbing that the C code legitimately leaves unspecified, or the guardrail's own expected value disagrees with what the C code demonstrably produces — do NOT modify the guardrail file to work around it. Instead:

1. Continue implementing the rest of the contract as faithfully as you can.
2. Write a file at the benchmark root named `.guardrail_dispute.md` containing:
   - The specific assertion you believe is wrong (file name + line range).
   - The C semantics you observed.
   - The value the C code produces vs. what the guardrail expects.
3. Terminate the task with that dispute written; a human will review.

This valve exists because the guardrail is machine-generated and occasionally wrong. Silent workarounds (editing the guardrail, weakening assertions, stubbing the failing probe) defeat the purpose of guardrail-guided translation and are not acceptable.

# Validation

After implementation, run:

1. `cargo build` — must succeed without errors.
2. `cargo test --bin generated_guardrails` — must succeed (all guardrail assertions pass).

Iterate on your implementation until both succeed, or a `.guardrail_dispute.md` has been written per the dispute valve above.

# Repair

If `cargo build` fails, debug and fix the implementation until build succeeds without errors.

If `cargo test --bin generated_guardrails` fails, the failure is (in almost all cases) a real semantic divergence between your Rust implementation and the C code. Re-read the C, locate the behavior the failing assertion is probing, and correct the Rust. Do NOT modify the guardrail file.

# Completion criteria

The task is complete only when:

- The Rust implementation matches the expected behavior of the C code.
- All required entry point interfaces are implemented without modification.
- The guardrail file at `rust/src/bin/generated_guardrails.rs` has not been modified.
- `cargo build` succeeds.
- `cargo test --bin generated_guardrails` succeeds — OR a `.guardrail_dispute.md` has been written at the benchmark root documenting why a specific guardrail contradicts the observed C semantics.
- No disallowed files were changed.
