"""TransformedTargetRegressor round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import compose, linear_model, pipeline, preprocessing

from skeights import SklearnModel

from .conftest import _round_trip


def _ttr(alpha: float = 0.001) -> compose.TransformedTargetRegressor:
    """A pipeline-wrapped ElasticNet behind a StandardScaler target transformer."""
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
    y = pd.DataFrame(
        {
            "t_small": X["a"] * 2 + 0.1 + 0.01 * rng.randn(300),
            "t_big": X["b"] * 50 + 100 + rng.randn(300),
        }
    )
    model = SklearnModel(_ttr())
    model.fit(X, y)

    arrays = model.get_arrays()
    assert "regressor_/model/coef_" in arrays
    assert "regressor_/scaler/mean_" in arrays
    assert "transformer_/mean_" in arrays
    assert "transformer_/scale_" in arrays

    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_ttr_single_target_round_trip():
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(300, 4), columns=list("abcd"))
    y = pd.DataFrame({"only": X["b"] * 50 + 100 + rng.randn(300)})
    model = SklearnModel(_ttr())
    model.fit(X, y)

    assert model.get_state()["fitted_state"]["_training_dim"] == 1

    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)
