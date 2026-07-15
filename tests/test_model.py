import numpy as np

from deephar.config import load_config
from deephar.model import build_rnn_model


def test_build_rnn_model_forward_pass_shape():
    config = load_config()
    num_classes = 6
    n_features = 15
    model = build_rnn_model((1, n_features), num_classes, config)

    X = np.random.rand(4, 1, n_features).astype("float32")
    preds = model.predict(X, verbose=0)

    assert preds.shape == (4, num_classes)
    # softmax output: each row sums to ~1
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-4)


def test_build_rnn_model_is_compiled():
    config = load_config()
    model = build_rnn_model((1, 10), 6, config)
    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"
