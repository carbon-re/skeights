"""Shared fixtures for skeights tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from skeights._core import (
    _arrays_from_estimator,
    _collect_fitted_state,
    _restore_estimator_arrays,
    _restore_fitted_state,
)
from skeights._params import _rebuild_estimator_from_params, get_model_params
from skeights._utils import get_sklearn_public_path


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 2)), columns=["f0", "f1"])
    y = pd.DataFrame({"y": X["f0"] * 0.3 + X["f1"] * 0.7})
    return X, y


@pytest.fixture
def binary_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((40, 3)), columns=["f0", "f1", "f2"])
    y = pd.DataFrame({"label": (X["f0"] + X["f1"] > 0).astype(int)})
    return X, y


def round_trip(estimator: BaseEstimator) -> BaseEstimator:
    """Serialize and deserialize an estimator via state + arrays."""
    params = get_model_params(estimator)
    if "type" not in params:
        params["type"] = get_sklearn_public_path(type(estimator))
    arrays = _arrays_from_estimator(estimator)
    fitted_state = _collect_fitted_state(estimator)

    restored = _rebuild_estimator_from_params(params)
    _restore_estimator_arrays(restored, arrays, fitted_state=fitted_state or None)
    if fitted_state:
        _restore_fitted_state(restored, fitted_state)
    return restored
