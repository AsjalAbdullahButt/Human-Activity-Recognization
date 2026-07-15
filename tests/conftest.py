import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deephar.config import load_config  # noqa: E402


@pytest.fixture
def small_config(tmp_path):
    """A config pointing all outputs at a pytest tmp_path, with tiny sizes
    so tests run fast and never touch the project's real outputs/ dir."""
    config = load_config()
    config.paths.outputs_dir = tmp_path / "outputs"
    config.paths.models_dir = tmp_path / "outputs" / "models"
    config.paths.plots_dir = tmp_path / "outputs" / "plots"
    config.paths.metrics_dir = tmp_path / "outputs" / "metrics"
    config.data.train_csv = tmp_path / "data" / "train.csv"
    config.data.test_csv = tmp_path / "data" / "test.csv"
    config.data.synthetic_samples_train = 120
    config.data.synthetic_samples_test = 40
    config.data.synthetic_n_features = 20
    config.train.epochs = 2
    config.train.batch_size = 16
    return config
