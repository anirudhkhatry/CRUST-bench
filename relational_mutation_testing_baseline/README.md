# Relational mutation testing baseline (C ↔ Rust)

Can an LLM generate a C test that kills a C mutant, and does that test still discriminate
after being transpiled to Rust?

```
C source P ──mutate──▶ P′ ──LLM──▶ C test T (passes P, fails P′) ──transpile──▶ Rust test RT
```

**Design spec:**
[`docs/superpowers/specs/2026-09-02-relational-mutation-testing-baseline-design.md`](../docs/superpowers/specs/2026-09-02-relational-mutation-testing-baseline-design.md)

Nothing is implemented yet. Phase 0 (feasibility: panel size + C build harness, no LLM
calls) is the first step.
