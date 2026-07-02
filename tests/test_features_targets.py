"""Verify features and targets survive round-trip."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import linear_model, pipeline, preprocessing

from skeights import SklearnModel

from .conftest import _round_trip


def test_features_preserved(regression_data):
    X, y = regression_data
    model = SklearnModel(linear_model.Ridge())
    model.fit(X, y)
    restored = _round_trip(model)
    assert restored.get_features() == ["f0", "f1"]


def test_targets_preserved(regression_data):
    X, y = regression_data
    model = SklearnModel(linear_model.Ridge())
    model.fit(X, y)
    restored = _round_trip(model)
    assert restored.get_targets() == ["y"]


def test_multi_target_preserved():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 2)), columns=["a", "b"])
    y = pd.DataFrame({"t1": X["a"] * 2, "t2": X["b"] * 3})
    model = SklearnModel(linear_model.Ridge())
    model.fit(X, y)
    restored = _round_trip(model)
    assert restored.get_features() == ["a", "b"]
    assert restored.get_targets() == ["t1", "t2"]
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_pipeline_features_from_inner_step(regression_data):
    X, y = regression_data
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge()),
        ]
    )
    model = SklearnModel(pipe)
    model.fit(X, y)
    restored = _round_trip(model)
    assert restored.get_features() == ["f0", "f1"]


def test_use_predict_proba_preserved(binary_data):
    X, y = binary_data
    model = SklearnModel(linear_model.LogisticRegression(), use_predict_proba=True)
    model.fit(X, y)
    restored = _round_trip(model)
    assert restored.use_predict_proba is True
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
