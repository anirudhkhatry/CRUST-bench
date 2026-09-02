"""Test package.

``unittest discover`` puts ``tests/`` on sys.path, not the study root, so the root is
added here -- once, before any test module runs -- to make ``harness.*`` importable
regardless of the working directory the run started from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
