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

from skeights import SklearnModel, SklearnScaler


def _fit(estimator, X, y):
    model = SklearnModel(estimator)
    model.fit(X, y)
    return model


def test_linear_arrays_contain_coef_and_intercept(regression_data):
    X, y = regression_data
    model = _fit(linear_model.Ridge(alpha=0.1), X, y)
    arrays = model.get_arrays()
    assert "coef_" in arrays
    assert "intercept_" in arrays
    assert arrays["coef_"].ndim in (1, 2)
    assert 2 in arrays["coef_"].shape


def test_pipeline_arrays_are_namespaced(regression_data):
    X, y = regression_data
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    model = _fit(pipe, X, y)
    arrays = model.get_arrays()
    assert "scaler/mean_" in arrays
    assert "scaler/scale_" in arrays
    assert "model/coef_" in arrays
    assert "model/intercept_" in arrays


def test_state_contains_hyperparams(regression_data):
    X, y = regression_data
    model = _fit(linear_model.Ridge(alpha=0.1), X, y)
    state = model.get_state()
    assert state["model_params"]["alpha"] == 0.1
    assert state["type"] == "SklearnModel"
    assert state["features"] == ["f0", "f1"]
    assert state["targets"] == ["y"]


def test_state_is_json_serializable(regression_data):
    X, y = regression_data
    model = _fit(linear_model.Ridge(alpha=0.1), X, y)
    state = model.get_state()
    serialized = json.dumps(state)
    roundtripped = json.loads(serialized)
    assert roundtripped["model_params"]["alpha"] == 0.1


def test_mlp_arrays_contain_layer_weights(regression_data):
    X, y = regression_data
    model = _fit(
        MLPRegressor(hidden_layer_sizes=(4, 3), random_state=0, max_iter=1),
        X,
        y,
    )
    arrays = model.get_arrays()
    assert "coefs_0" in arrays
    assert "coefs_1" in arrays
    assert "intercepts_0" in arrays
    assert "intercepts_1" in arrays
    assert arrays["coefs_0"].shape == (2, 4)
    assert arrays["coefs_1"].shape == (4, 3)


def test_mlp_fitted_state_contains_layer_info(regression_data):
    X, y = regression_data
    model = _fit(
        MLPRegressor(hidden_layer_sizes=(4,), random_state=0, max_iter=1),
        X,
        y,
    )
    fitted = model.get_state()["fitted_state"]
    assert "n_layers_" in fitted
    assert "out_activation_" in fitted


def test_rf_arrays_contain_tree_nodes(regression_data):
    X, y = regression_data
    model = _fit(
        RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0),
        X,
        y,
    )
    arrays = model.get_arrays()
    assert "_trees/0/values" in arrays
    assert "_trees/1/values" in arrays
    assert "_trees/2/values" in arrays
    assert any("nodes_" in k for k in arrays)


def test_rf_state_contains_tree_count(regression_data):
    X, y = regression_data
    model = _fit(
        RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0),
        X,
        y,
    )
    assert model.get_state()["fitted_state"]["n_trees"] == 3


def test_gb_state_contains_init_type(regression_data):
    X, y = regression_data
    model = _fit(
        GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0),
        X,
        y,
    )
    fitted = model.get_state()["fitted_state"]
    assert "_init/type" in fitted
    assert "DummyRegressor" in fitted["_init/type"]


def test_hgb_arrays_contain_predictors_and_bin_mapper(regression_data):
    X, y = regression_data
    model = _fit(
        HistGradientBoostingRegressor(max_iter=3, random_state=0),
        X,
        y,
    )
    arrays = model.get_arrays()
    assert "_baseline_prediction" in arrays
    assert "_bin_mapper/bin_thresholds_0" in arrays
    assert "_bin_mapper/is_categorical_" in arrays
    assert "_predictors/0/0/nodes_value" in arrays


def test_scaler_state_contains_inner_type(regression_data):
    X, _ = regression_data
    scaler = SklearnScaler(preprocessing.StandardScaler())
    scaler.fit(X)
    state = scaler.get_state()
    assert state["type"] == "SklearnScaler"
    assert "StandardScaler" in state["inner_type"]
    assert "init_params" in state


def test_scaler_arrays_contain_mean_and_scale(regression_data):
    X, _ = regression_data
    scaler = SklearnScaler(preprocessing.StandardScaler())
    scaler.fit(X)
    arrays = scaler.get_arrays()
    assert "mean_" in arrays
    assert "scale_" in arrays
    assert "var_" in arrays
    assert arrays["mean_"].shape == (2,)
