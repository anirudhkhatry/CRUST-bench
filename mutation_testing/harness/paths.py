"""The three directories every script agrees on.

Scripts sit two levels down now (``experiments/x.py``), so ``Path(__file__).parent`` is
no longer the study root and each script computing its own paths would drift. One
definition here, imported everywhere.
"""

from pathlib import Path

#: ``mutation_testing/`` -- the study root.
ROOT = Path(__file__).resolve().parents[1]

#: Run outputs: ``results/<model>/<run>/...``. Git-ignored, multi-gigabyte.
RESULTS = ROOT / "results"

#: Small committed artefacts: corpus manifests, readiness reports, cross-model CSVs.
DATA = ROOT / "data"

# Note for entry points: importing this module already requires the study root on
# ``sys.path``, so each runnable script inserts it itself before the first ``harness``
# import. That three-line preamble is what lets the scripts run both as plain files
# (``python experiments/run_transpile_full.py``) and as modules (``python -m ...``).
