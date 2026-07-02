"""MLP regressor and classifier round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import pipeline, preprocessing
from sklearn.neural_network import MLPClassifier, MLPRegressor

from skeights import SklearnModel

from .conftest import _round_trip


@pytest.mark.parametrize("use_pipeline", [True, False])
def test_mlp_round_trip(regression_data, use_pipeline: bool):
    X, y = regression_data
    if use_pipeline:
        inner = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(4,),
                        activation="tanh",
                        random_state=0,
                        max_iter=1000,
                    ),
                ),
            ]
        )
    else:
        inner = MLPRegressor(
            hidden_layer_sizes=(4,), activation="relu", random_state=0, max_iter=1000
        )
    model = SklearnModel(inner)
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_mlp_classifier_round_trip(binary_data):
    X, y = binary_data
    model = SklearnModel(
        MLPClassifier(hidden_layer_sizes=(4,), random_state=0, max_iter=500),
        use_predict_proba=True,
    )
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
