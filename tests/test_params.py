"""Param serialization tests: get/set/rebuild."""

from __future__ import annotations

from sklearn import ensemble, linear_model, pipeline, preprocessing

from skeights._params import (
    _rebuild_estimator_from_params,
    get_model_params,
    set_model_params,
)
from skeights._utils import get_sklearn_public_path


def test_get_params_bare_estimator():
    model = linear_model.Ridge(alpha=0.5)
    params = get_model_params(model)
    assert params["alpha"] == 0.5


def test_get_params_pipeline():
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    params = get_model_params(pipe)
    assert "steps" in params
    assert "scaler" in params["steps"]
    assert "model" in params["steps"]
    assert params["steps"]["model"]["alpha"] == 0.1


def test_set_params_pipeline():
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    set_model_params(
        pipe,
        {
            "steps": {
                "model": {"alpha": 0.5},
            },
        },
    )
    assert pipe.named_steps["model"].alpha == 0.5


def test_rebuild_estimator_from_params():
    params = {
        "type": "sklearn.linear_model.Ridge",
        "alpha": 0.42,
    }
    estimator = _rebuild_estimator_from_params(params)
    assert isinstance(estimator, linear_model.Ridge)
    assert estimator.alpha == 0.42


def test_rebuild_pipeline_from_params():
    pipe = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            ("model", linear_model.Ridge(alpha=0.1)),
        ]
    )
    params = get_model_params(pipe)
    # get_model_params doesn't add "type" at top level; SklearnModel does.
    params["type"] = get_sklearn_public_path(type(pipe))
    rebuilt = _rebuild_estimator_from_params(params)
    assert isinstance(rebuilt, pipeline.Pipeline)
    assert isinstance(rebuilt.named_steps["scaler"], preprocessing.StandardScaler)
    assert rebuilt.named_steps["model"].alpha == 0.1


def test_get_sklearn_public_path():
    path = get_sklearn_public_path(preprocessing.StandardScaler)
    assert path == "sklearn.preprocessing.StandardScaler"


def test_get_sklearn_public_path_linear():
    path = get_sklearn_public_path(linear_model.Ridge)
    assert path == "sklearn.linear_model.Ridge"


def test_meta_estimator_params():
    estimators = [
        ("ridge", linear_model.Ridge(alpha=0.1)),
        ("lasso", linear_model.Lasso(alpha=0.2)),
    ]
    stack = ensemble.StackingRegressor(
        estimators=estimators,
        final_estimator=linear_model.Ridge(alpha=0.5),
    )
    params = get_model_params(stack)
    assert "estimators" in params
    assert "final_estimator" in params
    assert params["final_estimator"]["alpha"] == 0.5
