"""Gaussian Process round-trip tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from skeights import SklearnModel

from .conftest import _round_trip


def test_gpr_round_trip():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((20, 2)), columns=["f0", "f1"])
    y = pd.DataFrame({"y": np.sin(X["f0"]) + 0.1 * rng.standard_normal(20)})
    model = SklearnModel(
        GaussianProcessRegressor(
            kernel=WhiteKernel(noise_level=0.1) + RBF(length_scale=1.0),
            random_state=0,
        )
    )
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_gpc_round_trip(binary_data):
    X, y = binary_data
    model = SklearnModel(
        GaussianProcessClassifier(kernel=WhiteKernel() + RBF(), random_state=0)
    )
    model.fit(X, y)
    orig = cast(GaussianProcessClassifier, model.model)
    rest = cast(GaussianProcessClassifier, _round_trip(model).model)
    np.testing.assert_array_equal(orig.predict(X.values), rest.predict(X.values))
    np.testing.assert_allclose(
        orig.predict_proba(X.values), rest.predict_proba(X.values), atol=1e-10
    )
