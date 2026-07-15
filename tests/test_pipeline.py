import pytest

from deephar.data import RealDataNotFoundError
from deephar.train import run_pipeline


def test_run_pipeline_end_to_end_produces_model_and_preprocessing_artifacts(small_config):
    result = run_pipeline(small_config, allow_synthetic=True, run_tuning=False)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["val_accuracy_group_split"] <= 1.0
    assert 0.0 <= result["val_accuracy_leaky_split"] <= 1.0
    assert 0.0 <= result["baseline_dense_test_accuracy"] <= 1.0
    assert result["used_synthetic"] is True
    assert result["model_path"].exists()

    models_dir = small_config.paths.models_dir
    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "label_encoder.joblib").exists()
    assert (models_dir / "best_model.keras").exists()
    assert (models_dir / "leaky_split_model.keras").exists()

    metrics_dir = small_config.paths.metrics_dir
    assert (metrics_dir / "split_comparison_report.txt").exists()


def test_run_pipeline_is_reproducible_given_same_seed(small_config, tmp_path):
    """Regression test for missing seeding: two runs with the same config/seed
    should produce the same final test accuracy."""
    import copy

    config_a = copy.deepcopy(small_config)
    config_a.paths.outputs_dir = tmp_path / "run_a"
    config_a.paths.models_dir = tmp_path / "run_a" / "models"
    config_a.paths.plots_dir = tmp_path / "run_a" / "plots"
    config_a.paths.metrics_dir = tmp_path / "run_a" / "metrics"

    config_b = copy.deepcopy(small_config)
    config_b.paths.outputs_dir = tmp_path / "run_b"
    config_b.paths.models_dir = tmp_path / "run_b" / "models"
    config_b.paths.plots_dir = tmp_path / "run_b" / "plots"
    config_b.paths.metrics_dir = tmp_path / "run_b" / "metrics"

    result_a = run_pipeline(config_a, allow_synthetic=True, run_tuning=False)
    result_b = run_pipeline(config_b, allow_synthetic=True, run_tuning=False)

    assert result_a["accuracy"] == result_b["accuracy"]


def test_run_pipeline_fails_loudly_without_real_data_and_no_demo_flag(small_config):
    """Production training must not silently fall back to synthetic data --
    it must raise, so a real run can never be confused with a demo run."""
    with pytest.raises(RealDataNotFoundError):
        run_pipeline(small_config, allow_synthetic=False, run_tuning=False)


def test_run_pipeline_with_tuning_writes_tuning_artifacts(small_config):
    result = run_pipeline(small_config, allow_synthetic=True, run_tuning=True)

    assert result["best_hyperparameters"] is not None
    metrics_dir = small_config.paths.metrics_dir
    assert (metrics_dir / "tuning_trials.csv").exists()
    assert (metrics_dir / "best_hyperparameters.json").exists()
