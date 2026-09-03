"""Relational mutation testing between C and Rust.

    C source P --mutate--> P'  --LLM--> C test T (passes P, fails P')
                                          |
                                     transpile (RBench interfaces only)
                                          v
    reference Rust impl M:             RT must pass M and fail M'
                                       where M' carries the same described fault

See ../README.md for the entrypoint and docs/superpowers/specs/ for the design.
"""
