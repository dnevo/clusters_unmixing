from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clusters_unmixing.config import ExperimentConfig
from clusters_unmixing.pipelines import run_correlation_experiments

if __name__ == "__main__":
    config_path = PROJECT_ROOT / "experiments" / "configs" / "correlation_options.yaml"
    experiment_config = ExperimentConfig.from_file(config_path)
    result = run_correlation_experiments(experiment_config)
    print(result)
