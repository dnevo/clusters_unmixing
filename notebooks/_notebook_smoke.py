from __future__ import annotations

import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clusters_unmixing.utils import configure_notebook, default_config_path, run_diagnostics_notebook

configure_notebook(project_root=PROJECT_ROOT, autoreload=False)
run_diagnostics_notebook(config_path=default_config_path(PROJECT_ROOT), project_root=PROJECT_ROOT)

