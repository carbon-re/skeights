"""Error handling tests."""

from __future__ import annotations

import pytest
from sklearn.neighbors import KNeighborsRegressor

from skeights import SklearnModel


def test_unsupported_estimator_raises(regression_data):
    X, y = regression_data
    model = SklearnModel(KNeighborsRegressor())
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.get_arrays()


def test_get_targets_before_fit_raises():
    model = SklearnModel(KNeighborsRegressor())
    with pytest.raises(ValueError, match="Targets are not set"):
        model.get_targets()


def test_predict_before_fit_raises():
    import pandas as pd

    model = SklearnModel(KNeighborsRegressor())
    X = pd.DataFrame({"a": [1.0]})
    with pytest.raises(AssertionError, match="fitted"):
        model.predict_numpy(X.values)
