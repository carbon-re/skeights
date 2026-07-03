"""HistGradientBoosting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import ensemble

from skeights._core import _arrays_from_estimator

from .conftest import assert_serializable, round_trip


def test_hgb_regressor_round_trip():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((80, 3)), columns=["f0", "f1", "f2"])
    y = np.sin(X["f0"]) + 0.1 * rng.standard_normal(80)
    model = ensemble.HistGradientBoostingRegressor(
        max_iter=5, max_leaf_nodes=8, random_state=0
    )
    model.fit(X, y)
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
    np.testing.assert_allclose(
        restored._baseline_prediction,  # type: ignore[attr-defined]
        model._baseline_prediction,  # type: ignore[attr-defined]
        atol=1e-15,
    )


def test_hgb_classifier_round_trip(binary_data):
    X, y = binary_data
    model = ensemble.HistGradientBoostingClassifier(max_iter=5, random_state=0)
    model.fit(X, y["label"])
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict_proba(X), restored.predict_proba(X), atol=1e-10
    )


def test_hgb_out_of_sample(regression_data_split):
    X_train, y_train, X_test, _ = regression_data_split
    model = ensemble.HistGradientBoostingRegressor(
        max_iter=10, max_leaf_nodes=8, random_state=0
    )
    model.fit(X_train, y_train["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict(X_test), restored.predict(X_test), atol=1e-10
    )


def test_hgb_arrays_contain_predictors_and_bin_mapper(regression_data):
    X, y = regression_data
    model = ensemble.HistGradientBoostingRegressor(max_iter=3, random_state=0)
    model.fit(X, y["y"])
    arrays = _arrays_from_estimator(model)
    assert "_baseline_prediction" in arrays
    assert "_bin_mapper/bin_thresholds_0" in arrays
    assert "_bin_mapper/is_categorical_" in arrays
    assert "_predictors/0/0/nodes_value" in arrays


def test_hgb_serialization_formats(regression_data):
    X, y = regression_data
    model = ensemble.HistGradientBoostingRegressor(max_iter=3, random_state=0)
    model.fit(X, y["y"])
    assert_serializable(model)
