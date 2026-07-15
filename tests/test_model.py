import numpy as np

from deephar.config import load_config
from deephar.model import build_dense_baseline_model, build_rnn_model


def test_build_rnn_model_forward_pass_shape():
    config = load_config()
    num_classes = 6
    timesteps, n_channels = 16, 9
    model = build_rnn_model((timesteps, n_channels), num_classes, config)

    X = np.random.rand(4, timesteps, n_channels).astype("float32")
    preds = model.predict(X, verbose=0)

    assert preds.shape == (4, num_classes)
    # softmax output: each row sums to ~1
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-4)


def test_build_rnn_model_is_compiled():
    config = load_config()
    model = build_rnn_model((16, 9), 6, config)
    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"


def test_build_rnn_model_respects_num_lstm_layers():
    config = load_config()
    config.model.num_lstm_layers = 1
    model_1_layer = build_rnn_model((16, 9), 6, config)
    lstm_layers_1 = [l for l in model_1_layer.layers if l.__class__.__name__ == "LSTM"]
    assert len(lstm_layers_1) == 1

    config.model.num_lstm_layers = 2
    model_2_layer = build_rnn_model((16, 9), 6, config)
    lstm_layers_2 = [l for l in model_2_layer.layers if l.__class__.__name__ == "LSTM"]
    assert len(lstm_layers_2) == 2


def test_build_dense_baseline_model_forward_pass_shape():
    config = load_config()
    num_classes = 6
    n_features = 561
    model = build_dense_baseline_model(n_features, num_classes, config)

    X = np.random.rand(4, n_features).astype("float32")
    preds = model.predict(X, verbose=0)

    assert preds.shape == (4, num_classes)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-4)
    assert not any(l.__class__.__name__ == "LSTM" for l in model.layers)
