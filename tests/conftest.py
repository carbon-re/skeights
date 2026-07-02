"""Shared fixtures for skeights tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skeights import SklearnModel


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


def _round_trip(model: SklearnModel) -> SklearnModel:
    state = model.get_state()
    arrays = model.get_arrays()
    return SklearnModel.from_state(state, arrays)
