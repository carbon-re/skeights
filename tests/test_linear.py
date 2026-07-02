"""Linear model, pipeline, and scaler round-trip tests."""

from __future__ import annotations

import numpy as np
from sklearn import linear_model, pipeline, preprocessing

from .conftest import round_trip


def test_linear_round_trip(regression_data):
    X, y = regression_data
    model = linear_model.Ridge(alpha=0.1)
    model.fit(X, y["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_pipeline_round_trip(regression_data):
    X, y = regression_data
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    pipe.fit(X, y["y"])
    restored = round_trip(pipe)
    np.testing.assert_allclose(pipe.predict(X), restored.predict(X), atol=1e-10)


def test_scaler_round_trip(regression_data):
    X, _ = regression_data
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    restored = round_trip(scaler)
    np.testing.assert_allclose(scaler.transform(X), restored.transform(X), atol=1e-10)
