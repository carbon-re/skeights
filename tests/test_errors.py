"""Error handling tests."""

from __future__ import annotations

import pytest
from sklearn.neighbors import KNeighborsRegressor

from skeights._core import _arrays_from_estimator


def test_unsupported_estimator_raises(regression_data):
    X, y = regression_data
    model = KNeighborsRegressor()
    model.fit(X, y["y"])
    with pytest.raises(NotImplementedError):
        _arrays_from_estimator(model)
