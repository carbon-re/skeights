"""MLP regressor and classifier tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import pipeline, preprocessing
from sklearn.neural_network import MLPClassifier, MLPRegressor

from skeights._core import _arrays_from_estimator, _collect_fitted_state

from .conftest import round_trip


@pytest.mark.parametrize("use_pipeline", [True, False])
def test_mlp_regressor_round_trip(regression_data, use_pipeline: bool):
    X, y = regression_data
    if use_pipeline:
        model = pipeline.Pipeline(
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
        model = MLPRegressor(
            hidden_layer_sizes=(4,),
            activation="relu",
            random_state=0,
            max_iter=1000,
        )
    model.fit(X, y["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_mlp_classifier_round_trip(binary_data):
    X, y = binary_data
    model = MLPClassifier(hidden_layer_sizes=(4,), random_state=0, max_iter=500)
    model.fit(X, y["label"])
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict_proba(X), restored.predict_proba(X), atol=1e-10
    )


def test_mlp_arrays_contain_layer_weights(regression_data):
    X, y = regression_data
    model = MLPRegressor(hidden_layer_sizes=(4, 3), random_state=0, max_iter=1)
    model.fit(X, y["y"])
    arrays = _arrays_from_estimator(model)
    assert "coefs_0" in arrays
    assert "coefs_1" in arrays
    assert "intercepts_0" in arrays
    assert "intercepts_1" in arrays
    assert arrays["coefs_0"].shape == (2, 4)
    assert arrays["coefs_1"].shape == (4, 3)


def test_mlp_fitted_state_contains_layer_info(regression_data):
    X, y = regression_data
    model = MLPRegressor(hidden_layer_sizes=(4,), random_state=0, max_iter=1)
    model.fit(X, y["y"])
    fitted = _collect_fitted_state(model)
    assert "n_layers_" in fitted
    assert "out_activation_" in fitted
