"""Linear model, pipeline, and scaler round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model, pipeline, preprocessing

from skeights import SklearnModel, SklearnScaler

from .conftest import _round_trip


def test_linear_round_trip(regression_data):
    X, y = regression_data
    model = SklearnModel(linear_model.Ridge(alpha=0.1))
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_pipeline_round_trip(regression_data):
    X, y = regression_data
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    model = SklearnModel(pipe)
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


@pytest.mark.parametrize(
    "scaler_cls",
    [
        preprocessing.StandardScaler,
        preprocessing.MinMaxScaler,
        preprocessing.RobustScaler,
    ],
)
def test_scaler_round_trip(regression_data, scaler_cls):
    X, _ = regression_data
    scaler = SklearnScaler(scaler_cls())
    scaler.fit(X)
    state = scaler.get_state()
    arrays = scaler.get_arrays()
    loaded = SklearnScaler.from_state(state, arrays)
    pd.testing.assert_frame_equal(scaler.transform(X), loaded.transform(X))
