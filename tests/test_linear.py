"""Linear model, pipeline, and scaler tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn import linear_model, pipeline, preprocessing

from skeights._core import _arrays_from_estimator, _collect_fitted_state
from skeights._params import get_model_params
from skeights._utils import get_sklearn_public_path

from .conftest import round_trip


def test_linear_round_trip(regression_data):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
    np.testing.assert_allclose(restored.coef_, model.coef_, atol=1e-15)
    np.testing.assert_allclose(restored.intercept_, model.intercept_, atol=1e-15)


def test_pipeline_round_trip(regression_data):
    X, y = regression_data
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    pipe.fit(X, y["y"])
    restored = round_trip(pipe)
    np.testing.assert_allclose(pipe.predict(X), restored.predict(X), atol=1e-10)


@pytest.mark.parametrize(
    "scaler_cls",
    [
        preprocessing.StandardScaler,
        preprocessing.MinMaxScaler,
        preprocessing.RobustScaler,
    ],
    ids=["standard", "minmax", "robust"],
)
def test_scaler_round_trip(regression_data, scaler_cls):
    X, _ = regression_data
    scaler = scaler_cls()
    scaler.fit(X)
    restored = round_trip(scaler)
    np.testing.assert_allclose(scaler.transform(X), restored.transform(X), atol=1e-10)


def test_linear_out_of_sample(regression_data_split):
    X_train, y_train, X_test, _ = regression_data_split
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X_train, y_train["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict(X_test), restored.predict(X_test), atol=1e-10
    )


def test_pipeline_out_of_sample(regression_data_split):
    X_train, y_train, X_test, _ = regression_data_split
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    pipe.fit(X_train, y_train["y"])
    restored = round_trip(pipe)
    np.testing.assert_allclose(
        pipe.predict(X_test), restored.predict(X_test), atol=1e-10
    )


def test_linear_arrays_contain_coef_and_intercept(regression_data):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    arrays = _arrays_from_estimator(model)
    assert "coef_" in arrays
    assert "intercept_" in arrays
    assert 2 in arrays["coef_"].shape


def test_pipeline_arrays_are_namespaced(regression_data):
    X, y = regression_data
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    pipe.fit(X, y["y"])
    arrays = _arrays_from_estimator(pipe)
    assert "scaler/mean_" in arrays
    assert "scaler/scale_" in arrays
    assert "model/coef_" in arrays
    assert "model/intercept_" in arrays


def test_state_contains_hyperparams(regression_data):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    params = get_model_params(model)
    params["type"] = get_sklearn_public_path(type(model))
    assert params["alpha"] == 0.1
    assert "Ridge" in params["type"]


def test_state_is_json_serializable(regression_data):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    params = get_model_params(model)
    fitted_state = _collect_fitted_state(model)
    state = {"model_params": params, "fitted_state": fitted_state}
    serialized = json.dumps(state)
    roundtripped = json.loads(serialized)
    assert roundtripped["model_params"]["alpha"] == 0.1


def test_scaler_arrays_contain_mean_and_scale(regression_data):
    X, _ = regression_data
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    arrays = _arrays_from_estimator(scaler)
    assert "mean_" in arrays
    assert "scale_" in arrays
    assert "var_" in arrays
    assert arrays["mean_"].shape == (2,)
