"""TransformedTargetRegressor round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import compose, linear_model, pipeline, preprocessing

from skeights._core import _arrays_from_estimator, _collect_fitted_state

from .conftest import round_trip


def _ttr(alpha: float = 0.001) -> compose.TransformedTargetRegressor:
    """Pipeline-wrapped ElasticNet behind a StandardScaler target transformer."""
    return compose.TransformedTargetRegressor(
        regressor=pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("model", linear_model.ElasticNet(alpha=alpha, random_state=0)),
            ]
        ),
        transformer=preprocessing.StandardScaler(),
    )


def test_ttr_multi_output_round_trip():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(300, 4), columns=list("abcd"))
    y = np.column_stack(
        [
            X["a"] * 2 + 0.1 + 0.01 * rng.randn(300),
            X["b"] * 50 + 100 + rng.randn(300),
        ]
    )
    model = _ttr()
    model.fit(X, y)

    arrays = _arrays_from_estimator(model)
    assert "regressor_/model/coef_" in arrays
    assert "regressor_/scaler/mean_" in arrays
    assert "transformer_/mean_" in arrays
    assert "transformer_/scale_" in arrays

    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_ttr_single_target_round_trip():
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(300, 4), columns=list("abcd"))
    y = X["b"] * 50 + 100 + rng.randn(300)
    model = _ttr()
    model.fit(X, y)

    fitted_state = _collect_fitted_state(model)
    assert fitted_state["_training_dim"] == 1

    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
