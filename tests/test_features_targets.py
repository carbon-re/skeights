"""Verify predictions survive round-trip for edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import linear_model, pipeline, preprocessing

from .conftest import round_trip


def test_multi_target_round_trip():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 2)), columns=["a", "b"])
    y = np.column_stack([X["a"] * 2, X["b"] * 3])
    model = linear_model.Ridge()
    model.fit(X, y)
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_single_sample():
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([4.0])
    model = linear_model.Ridge()
    model.fit(X, y)
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_pipeline_predict_matches(regression_data):
    X, y = regression_data
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge()),
        ]
    )
    pipe.fit(X, y["y"])
    restored = round_trip(pipe)
    np.testing.assert_allclose(pipe.predict(X), restored.predict(X), atol=1e-10)
