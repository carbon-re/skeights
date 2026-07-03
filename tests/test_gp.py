"""Gaussian Process round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from .conftest import round_trip


def test_gpr_round_trip():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((20, 2)), columns=["f0", "f1"])
    y = np.sin(X["f0"]) + 0.1 * rng.standard_normal(20)
    model = GaussianProcessRegressor(
        kernel=WhiteKernel(noise_level=0.1) + RBF(length_scale=1.0),
        random_state=0,
    )
    model.fit(X, y)
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
    np.testing.assert_allclose(restored.alpha_, model.alpha_, atol=1e-15)
    np.testing.assert_allclose(restored.L_, model.L_, atol=1e-15)


def test_gpc_round_trip(binary_data):
    X, y = binary_data
    model = GaussianProcessClassifier(kernel=WhiteKernel() + RBF(), random_state=0)
    model.fit(X, y["label"])
    restored = round_trip(model)
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    np.testing.assert_allclose(
        model.predict_proba(X), restored.predict_proba(X), atol=1e-10
    )
