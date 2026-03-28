from __future__ import annotations

import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
project_root = NOTEBOOK_DIR.parent
SRC_DIR = project_root / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clusters_unmixing.utils import run_experiments_notebook

run_experiments_notebook(project_root=project_root)
