"""Model architectures: the primary LSTM (raw signal windows) and a
non-temporal Dense baseline (561 engineered features) for comparison."""
from __future__ import annotations

import keras
from keras import Input
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

from deephar.config import Config


def build_rnn_model(input_shape: tuple[int, int], num_classes: int, config: Config) -> Sequential:
    """Build the LSTM classifier. input_shape is (timesteps, channels), e.g.
    (128, 9) for the real UCI HAR inertial-signal windows -- a genuine time
    axis, unlike the (1, 561) shape a single pre-engineered feature vector
    would give an LSTM."""
    layers = [Input(shape=input_shape)]
    num_lstm_layers = max(1, config.model.num_lstm_layers)
    for i in range(num_lstm_layers):
        is_last_lstm = i == num_lstm_layers - 1
        units = config.model.lstm_units_1 if i == 0 else config.model.lstm_units_2
        layers.append(LSTM(units, return_sequences=not is_last_lstm))
        layers.append(Dropout(config.model.dropout))
    layers.append(Dense(config.model.dense_units, activation="relu"))
    layers.append(Dense(num_classes, activation="softmax"))

    model = Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.model.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dense_baseline_model(input_dim: int, num_classes: int, config: Config) -> Sequential:
    """Non-temporal baseline: a plain MLP over the 561 pre-engineered
    features (no time axis to exploit). Used only to sanity-check that the
    LSTM is learning something the engineered features don't already give
    away for free."""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(config.model.dropout),
            Dense(64, activation="relu"),
            Dropout(config.model.dropout),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.model.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
