"""Test that models can be retrained after loading from serialized artifacts.

For each model type: train on data A, save/load, retrain on data B.
Also train a fresh model (same params) on data B from scratch.
Assert both give identical predictions -- proving the loaded model
retrains equivalently to a fresh one.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import skeights


@pytest.fixture
def train_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + rng.standard_normal(100) * 0.1
    return X, y


@pytest.fixture
def new_data():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((80, 5))
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + rng.standard_normal(80) * 0.1
    return X, y


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def new_binary_data():
    rng = np.random.default_rng(77)
    X = rng.standard_normal((60, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _save_load(model, tmp_path):
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    return skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")


# -- sklearn ------------------------------------------------------------------


def test_retrain_ridge(train_data, new_data, tmp_path):
    X, y = train_data
    X2, y2 = new_data

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = Ridge(alpha=1.0)
    fresh.fit(X2, y2)

    np.testing.assert_allclose(loaded.predict(X2), fresh.predict(X2))


def test_retrain_pipeline(train_data, new_data, tmp_path):
    X, y = train_data
    X2, y2 = new_data

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe.fit(X, y)
    loaded = _save_load(pipe, tmp_path)
    loaded.fit(X2, y2)

    fresh = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    fresh.fit(X2, y2)

    np.testing.assert_allclose(loaded.predict(X2), fresh.predict(X2))


def test_retrain_random_forest(train_data, new_data, tmp_path):
    X, y = train_data
    X2, y2 = new_data

    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    fresh.fit(X2, y2)

    np.testing.assert_allclose(loaded.predict(X2), fresh.predict(X2))


def test_retrain_gradient_boosting(train_data, new_data, tmp_path):
    X, y = train_data
    X2, y2 = new_data

    model = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=0)
    fresh.fit(X2, y2)

    np.testing.assert_allclose(loaded.predict(X2), fresh.predict(X2))


def test_retrain_gradient_boosting_classifier(binary_data, new_binary_data, tmp_path):
    X, y = binary_data
    X2, y2 = new_binary_data

    model = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=0)
    fresh.fit(X2, y2)

    np.testing.assert_array_equal(loaded.predict(X2), fresh.predict(X2))
    np.testing.assert_allclose(loaded.predict_proba(X2), fresh.predict_proba(X2))


# -- LightGBM -----------------------------------------------------------------

lgb = pytest.importorskip("lightgbm")


def test_retrain_lgbm_regressor(train_data, new_data, tmp_path):
    X, y = train_data
    X2, y2 = new_data

    model = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
    fresh.fit(X2, y2)

    np.testing.assert_allclose(loaded.predict(X2), fresh.predict(X2))


def test_retrain_lgbm_classifier(binary_data, new_binary_data, tmp_path):
    X, y = binary_data
    X2, y2 = new_binary_data

    model = lgb.LGBMClassifier(n_estimators=5, max_depth=3, verbose=-1)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = lgb.LGBMClassifier(n_estimators=5, max_depth=3, verbose=-1)
    fresh.fit(X2, y2)

    np.testing.assert_array_equal(loaded.predict(X2), fresh.predict(X2))
    np.testing.assert_allclose(loaded.predict_proba(X2), fresh.predict_proba(X2))


# -- XGBoost -------------------------------------------------------------------

xgb = pytest.importorskip("xgboost")


def test_retrain_xgb_regressor(train_data, new_data, tmp_path):
    X, y = train_data
    X2, y2 = new_data

    model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
    fresh.fit(X2, y2)

    np.testing.assert_allclose(loaded.predict(X2), fresh.predict(X2))


def test_retrain_xgb_classifier(binary_data, new_binary_data, tmp_path):
    X, y = binary_data
    X2, y2 = new_binary_data

    model = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    loaded = _save_load(model, tmp_path)
    loaded.fit(X2, y2)

    fresh = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=0)
    fresh.fit(X2, y2)

    np.testing.assert_array_equal(loaded.predict(X2), fresh.predict(X2))
    np.testing.assert_allclose(loaded.predict_proba(X2), fresh.predict_proba(X2))
