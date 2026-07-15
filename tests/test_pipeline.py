from deephar.train import run_pipeline


def test_run_pipeline_end_to_end_produces_model_and_preprocessing_artifacts(small_config):
    result = run_pipeline(small_config)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["used_synthetic"] is True
    assert result["model_path"].exists()

    models_dir = small_config.paths.models_dir
    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "label_encoder.joblib").exists()
    assert (models_dir / "best_model.keras").exists()


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

    result_a = run_pipeline(config_a)
    result_b = run_pipeline(config_b)

    assert result_a["accuracy"] == result_b["accuracy"]
