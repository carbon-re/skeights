"""LightGBM round-trip and structural tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

import skeights  # noqa: E402
from skeights._core import _collect_fitted_state  # noqa: E402

from .conftest import round_trip  # noqa: E402


def test_lgbm_regressor_round_trip(regression_data):
    X, y = regression_data
    model = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
    model.fit(X, y["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_lgbm_classifier_round_trip(binary_data):
    X, y = binary_data
    model = lgb.LGBMClassifier(n_estimators=5, max_depth=3, verbose=-1)
    model.fit(X, y["label"])
    restored = round_trip(model)
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    np.testing.assert_allclose(
        model.predict_proba(X), restored.predict_proba(X), atol=1e-10
    )


def test_lgbm_state_contains_model_string(regression_data):
    X, y = regression_data
    model = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
    model.fit(X, y["y"])
    state = _collect_fitted_state(model)
    assert "model_str" in state
    assert state["model_str"].startswith("tree\n")
    assert "objective" in state


def test_lgbm_state_is_json_serializable(regression_data):
    X, y = regression_data
    model = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
    model.fit(X, y["y"])
    state = _collect_fitted_state(model)
    json.dumps(state)


def test_lgbm_save_load_round_trip(regression_data, tmp_path):
    X, y = regression_data
    model = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
    model.fit(X, y["y"])
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    np.testing.assert_allclose(model.predict(X), loaded.predict(X), atol=1e-10)
