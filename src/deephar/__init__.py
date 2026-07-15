"""DeepHAR: LSTM-based Human Activity Recognition pipeline."""
import os

# Keras 3 picks its backend at import time. TensorFlow has no wheel for very
# new Python releases yet, so default to the torch backend (already required
# by this project) unless the environment/user has explicitly chosen one.
os.environ.setdefault("KERAS_BACKEND", "torch")
