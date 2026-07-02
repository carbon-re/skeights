"""File I/O save/load round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing

import skeights
from skeights import SklearnModel, SklearnScaler


def test_save_load_round_trip(regression_data, tmp_path):
    X, y = regression_data
    model = SklearnModel(linear_model.Ridge(alpha=0.1))
    model.fit(X, y)

    arrays_path = tmp_path / "model.safetensors"
    state_path = tmp_path / "model.json"
    skeights.save(model, arrays_path, state_path)
    loaded = skeights.load(arrays_path, state_path)

    assert isinstance(loaded, SklearnModel)
    np.testing.assert_allclose(model.predict(X), loaded.predict(X), atol=1e-10)


def test_save_load_scaler(regression_data, tmp_path):
    X, _ = regression_data
    scaler = SklearnScaler(preprocessing.StandardScaler())
    scaler.fit(X)

    arrays_path = tmp_path / "scaler.safetensors"
    state_path = tmp_path / "scaler.json"
    skeights.save(scaler, arrays_path, state_path)
    loaded = skeights.load(arrays_path, state_path)

    assert isinstance(loaded, SklearnScaler)
    pd.testing.assert_frame_equal(scaler.transform(X), loaded.transform(X))
