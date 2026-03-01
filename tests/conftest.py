"""
Shared test configuration.

Adds both the project root and src/ to sys.path so that:
  - Modules using relative imports (api.py) can be imported as `src.api`
  - Flat modules (enhanced_agents, correlation_engine, etc.) still work
    with plain `import module_name`

This replaces the duplicated sys.path.insert() hack in every test file.
"""

import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_dir = os.path.join(_project_root, "src")

# Project root first — lets `from src.api import ...` resolve relative imports
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# src/ second — lets `import enhanced_agents` etc. still work for flat modules
if _src_dir not in sys.path:
    sys.path.insert(1, _src_dir)
