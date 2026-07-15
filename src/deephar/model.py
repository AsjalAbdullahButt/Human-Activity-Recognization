"""LSTM model architecture."""
from __future__ import annotations

from keras import Input
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

from deephar.config import Config


def build_rnn_model(input_shape: tuple[int, int], num_classes: int, config: Config) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(config.model.lstm_units_1, return_sequences=True),
            Dropout(config.model.dropout),
            LSTM(config.model.lstm_units_2),
            Dropout(config.model.dropout),
            Dense(config.model.dense_units, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
