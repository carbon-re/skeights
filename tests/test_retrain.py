"""Test that models can be retrained after loading from serialized artifacts.

Verifies that save/load produces models that are not just usable for
inference but can also be .fit() again on new data and produce valid
predictions afterwards.
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


def _save_load(model, tmp_path):
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json")
    return skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")


class TestRetrainSklearn:
    def test_ridge(self, train_data, new_data, tmp_path):
        X, y = train_data
        X2, y2 = new_data
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        assert preds.shape == (80,)
        assert np.isfinite(preds).all()

    def test_pipeline(self, train_data, new_data, tmp_path):
        X, y = train_data
        X2, y2 = new_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe.fit(X, y)
        loaded = _save_load(pipe, tmp_path)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        assert preds.shape == (80,)
        assert np.isfinite(preds).all()

    def test_random_forest(self, train_data, new_data, tmp_path):
        X, y = train_data
        X2, y2 = new_data
        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        assert preds.shape == (80,)
        assert np.isfinite(preds).all()

    def test_gradient_boosting(self, train_data, new_data, tmp_path):
        X, y = train_data
        X2, y2 = new_data
        model = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=0)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        assert preds.shape == (80,)
        assert np.isfinite(preds).all()

    def test_gradient_boosting_classifier(self, binary_data, tmp_path):
        X, y = binary_data
        model = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=0)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        rng = np.random.default_rng(77)
        X2 = rng.standard_normal((60, 5))
        y2 = (X2[:, 0] + X2[:, 1] > 0).astype(int)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        assert preds.shape == (60,)
        assert set(preds).issubset({0, 1})


class TestRetrainLightGBM:
    lgb = pytest.importorskip("lightgbm")

    def test_regressor(self, train_data, new_data, tmp_path):
        X, y = train_data
        X2, y2 = new_data
        model = self.lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        assert preds.shape == (80,)
        assert np.isfinite(preds).all()

    def test_classifier(self, binary_data, tmp_path):
        X, y = binary_data
        model = self.lgb.LGBMClassifier(n_estimators=5, max_depth=3, verbose=-1)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        rng = np.random.default_rng(77)
        X2 = rng.standard_normal((60, 5))
        y2 = (X2[:, 0] + X2[:, 1] > 0).astype(int)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        proba = loaded.predict_proba(X2)
        assert preds.shape == (60,)
        assert proba.shape == (60, 2)
        assert set(preds).issubset({0, 1})


class TestRetrainXGBoost:
    xgb = pytest.importorskip("xgboost")

    def test_regressor(self, train_data, new_data, tmp_path):
        X, y = train_data
        X2, y2 = new_data
        model = self.xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        assert preds.shape == (80,)
        assert np.isfinite(preds).all()

    def test_classifier(self, binary_data, tmp_path):
        X, y = binary_data
        model = self.xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=0)
        model.fit(X, y)
        loaded = _save_load(model, tmp_path)
        rng = np.random.default_rng(77)
        X2 = rng.standard_normal((60, 5))
        y2 = (X2[:, 0] + X2[:, 1] > 0).astype(int)
        loaded.fit(X2, y2)
        preds = loaded.predict(X2)
        proba = loaded.predict_proba(X2)
        assert preds.shape == (60,)
        assert proba.shape == (60, 2)
        assert set(preds).issubset({0, 1})
