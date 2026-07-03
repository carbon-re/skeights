"""File I/O save/load round-trip tests.

These go through JSON + safetensors on disk and predict on
held-out data, catching serialization issues that in-memory
round_trip() tests miss.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn import ensemble, linear_model, preprocessing
from sklearn.neural_network import MLPRegressor

import skeights


def test_save_load_ridge(regression_data_split, tmp_path):
    X_train, y_train, X_test, _ = regression_data_split
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X_train, y_train["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    assert isinstance(loaded, linear_model.Ridge)
    np.testing.assert_allclose(
        model.predict(X_test), loaded.predict(X_test), atol=1e-10
    )


def test_save_load_scaler(regression_data_split, tmp_path):
    X_train, _, X_test, _ = regression_data_split
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    skeights.save(scaler, tmp_path / "s.safetensors", tmp_path / "s.json")
    loaded = skeights.load(tmp_path / "s.safetensors", tmp_path / "s.json")
    assert isinstance(loaded, preprocessing.StandardScaler)
    np.testing.assert_allclose(
        scaler.transform(X_test), loaded.transform(X_test), atol=1e-10
    )


def test_save_load_random_forest(regression_data_split, tmp_path):
    X_train, y_train, X_test, _ = regression_data_split
    model = ensemble.RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X_train, y_train["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    np.testing.assert_allclose(
        model.predict(X_test), loaded.predict(X_test), atol=1e-10
    )


def test_save_load_gradient_boosting(regression_data_split, tmp_path):
    X_train, y_train, X_test, _ = regression_data_split
    model = ensemble.GradientBoostingRegressor(
        n_estimators=5, max_depth=3, random_state=0
    )
    model.fit(X_train, y_train["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    np.testing.assert_allclose(
        model.predict(X_test), loaded.predict(X_test), atol=1e-10
    )


def test_save_load_hgb(regression_data_split, tmp_path):
    X_train, y_train, X_test, _ = regression_data_split
    model = ensemble.HistGradientBoostingRegressor(max_iter=5, random_state=0)
    model.fit(X_train, y_train["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    np.testing.assert_allclose(
        model.predict(X_test), loaded.predict(X_test), atol=1e-10
    )


def test_save_load_mlp(regression_data_split, tmp_path):
    X_train, y_train, X_test, _ = regression_data_split
    model = MLPRegressor(hidden_layer_sizes=(4,), random_state=0, max_iter=500)
    model.fit(X_train, y_train["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    np.testing.assert_allclose(
        model.predict(X_test), loaded.predict(X_test), atol=1e-10
    )


def test_saved_state_contains_sklearn_version(regression_data, tmp_path):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    state = json.loads((tmp_path / "m.json").read_text())
    assert "skeights_version" in state
    assert "sklearn" in state["skeights_version"]
    import sklearn

    assert state["skeights_version"]["sklearn"] == sklearn.__version__


def test_saved_lgbm_contains_lightgbm_version(regression_data, tmp_path):
    lgb = __import__("pytest").importorskip("lightgbm")
    X, y = regression_data
    model = lgb.LGBMRegressor(n_estimators=3, verbose=-1)
    model.fit(X, y["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    state = json.loads((tmp_path / "m.json").read_text())
    assert state["skeights_version"]["lightgbm"] == lgb.__version__


def test_saved_xgb_contains_xgboost_version(regression_data, tmp_path):
    xgb = __import__("pytest").importorskip("xgboost")
    X, y = regression_data
    model = xgb.XGBRegressor(n_estimators=3)
    model.fit(X, y["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    state = json.loads((tmp_path / "m.json").read_text())
    assert state["skeights_version"]["xgboost"] == xgb.__version__


def test_load_warns_on_version_mismatch(regression_data, tmp_path):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")

    state = json.loads((tmp_path / "m.json").read_text())
    state["skeights_version"]["sklearn"] = "0.99.0"
    (tmp_path / "m.json").write_text(json.dumps(state))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    version_warnings = [x for x in w if "0.99.0" in str(x.message)]
    assert len(version_warnings) == 1


def test_load_no_warning_on_matching_version(regression_data, tmp_path):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    version_warnings = [x for x in w if "Predictions may differ" in str(x.message)]
    assert len(version_warnings) == 0
