"""HistGradientBoosting round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import ensemble

from .conftest import round_trip


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


def test_hgb_classifier_round_trip(binary_data):
    X, y = binary_data
    model = ensemble.HistGradientBoostingClassifier(max_iter=5, random_state=0)
    model.fit(X, y["label"])
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict_proba(X), restored.predict_proba(X), atol=1e-10
    )
