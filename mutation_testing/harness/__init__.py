"""The mutation-testing engine: pipeline, mutant runners, workspace splitting, operators.

Everything here is importable library code. The campaign drivers live in
``experiments/``, the scorers in ``analysis/``. ``run_mutation.py`` is the one module
that is both -- it is the end-to-end pipeline CLI and the source of the
setting -> directory map every driver needs.
"""
