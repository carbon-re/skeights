"""HistGradientBoosting round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import ensemble

from skeights import SklearnModel

from .conftest import _round_trip


def test_hgb_regressor_round_trip():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((80, 3)), columns=["f0", "f1", "f2"])
    y = pd.DataFrame({"target": np.sin(X["f0"]) + 0.1 * rng.standard_normal(80)})
    model = SklearnModel(
        ensemble.HistGradientBoostingRegressor(
            max_iter=5, max_leaf_nodes=8, random_state=0
        )
    )
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_hgb_classifier_round_trip(binary_data):
    X, y = binary_data
    model = SklearnModel(
        ensemble.HistGradientBoostingClassifier(max_iter=5, random_state=0),
        use_predict_proba=True,
    )
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
