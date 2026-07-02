"""Structural tests: verify serialized state/arrays have expected keys.

Round-trip tests prove predictions match, but a silent default can
mask a missing key. These tests pin the serialized structure.
"""

from __future__ import annotations

import json

from sklearn import linear_model, pipeline, preprocessing
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor

from skeights._core import _arrays_from_estimator, _collect_fitted_state
from skeights._params import get_model_params
from skeights._utils import get_sklearn_public_path


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


def test_rf_arrays_contain_tree_nodes(regression_data):
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y["y"])
    arrays = _arrays_from_estimator(model)
    assert "_trees/0/values" in arrays
    assert "_trees/1/values" in arrays
    assert "_trees/2/values" in arrays
    assert any("nodes_" in k for k in arrays)


def test_rf_state_contains_tree_count(regression_data):
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y["y"])
    assert _collect_fitted_state(model)["n_trees"] == 3


def test_gb_state_contains_init_type(regression_data):
    X, y = regression_data
    model = GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y["y"])
    fitted = _collect_fitted_state(model)
    assert "_init/type" in fitted
    assert "DummyRegressor" in fitted["_init/type"]


def test_hgb_arrays_contain_predictors_and_bin_mapper(regression_data):
    X, y = regression_data
    model = HistGradientBoostingRegressor(max_iter=3, random_state=0)
    model.fit(X, y["y"])
    arrays = _arrays_from_estimator(model)
    assert "_baseline_prediction" in arrays
    assert "_bin_mapper/bin_thresholds_0" in arrays
    assert "_bin_mapper/is_categorical_" in arrays
    assert "_predictors/0/0/nodes_value" in arrays


def test_scaler_arrays_contain_mean_and_scale(regression_data):
    X, _ = regression_data
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    arrays = _arrays_from_estimator(scaler)
    assert "mean_" in arrays
    assert "scale_" in arrays
    assert "var_" in arrays
    assert arrays["mean_"].shape == (2,)
