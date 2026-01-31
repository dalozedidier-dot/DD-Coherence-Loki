"""Pytest configuration: ensure local package imports resolve.

The repository layout nests the dd_coherence package under dd_coherence_tool/.
When tests are executed from repo root, this directory is not on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add dd_coherence_tool/ to sys.path so `import dd_coherence` works.
HERE = Path(__file__).resolve()
TOOL_ROOT = HERE.parents[1]  # .../dd_coherence_tool
sys.path.insert(0, str(TOOL_ROOT))
