"""Round-trip tests: fit → serialize → deserialize → predict matches."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn import compose, ensemble, linear_model, pipeline, preprocessing
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import RBF, Matern, Product, Sum, WhiteKernel
from sklearn.neural_network import MLPRegressor

import skeights
from skeights import SklearnModel, SklearnScaler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Linear model round-trips
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MLP round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pipeline", [True, False])
def test_mlp_round_trip(regression_data, use_pipeline: bool):
    X, y = regression_data
    if use_pipeline:
        inner = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(4,),
                        activation="tanh",
                        random_state=0,
                        max_iter=1000,
                    ),
                ),
            ]
        )
    else:
        inner = MLPRegressor(
            hidden_layer_sizes=(4,), activation="relu", random_state=0, max_iter=1000
        )
    model = SklearnModel(inner)
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# GP round-trips
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HGB round-trips
# ---------------------------------------------------------------------------


def test_hgb_regressor_round_trip():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((80, 3)), columns=["f0", "f1", "f2"])
    y = pd.DataFrame({"target": np.sin(X["f0"]) + 0.1 * rng.standard_normal(80)})
    model = SklearnModel(
        ensemble.HistGradientBoostingRegressor(
            max_iter=5, max_leaf_nodes=8, random_state=0
        )
    )
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


def test_hgb_classifier_round_trip(binary_data):
    X, y = binary_data
    model = SklearnModel(
        ensemble.HistGradientBoostingClassifier(max_iter=5, random_state=0),
        use_predict_proba=True,
    )
    model.fit(X, y)
    restored = _round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# TransformedTargetRegressor round-trips
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scaler round-trip
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Kernel serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(length_scale=2.5),
        WhiteKernel(noise_level=0.1) + RBF(length_scale=1.5),
        RBF(length_scale=1.0) * Matern(length_scale=2.0, nu=1.5),
        Sum(Product(RBF(), Matern()), WhiteKernel()),
    ],
)
def test_kernel_round_trip(kernel):
    from skeights._core import _deserialize_kernel, _serialize_kernel

    restored = _deserialize_kernel(_serialize_kernel(kernel))
    assert type(restored) is type(kernel)


# ---------------------------------------------------------------------------
# File I/O round-trip
# ---------------------------------------------------------------------------


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
