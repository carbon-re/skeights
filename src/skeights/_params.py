"""Hyperparameter serialization: get/set model params and rebuild from params."""

from __future__ import annotations

import importlib
import warnings
from typing import Any, cast

from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel
from sklearn.pipeline import Pipeline

from skeights._kernels import _deserialize_kernel, _serialize_kernel
from skeights._utils import _fix_json_param_types, get_sklearn_public_path


def get_model_params(model: BaseEstimator) -> dict[str, Any]:
    """Recursively extract hyperparameters from a fitted sklearn estimator."""

    def _get_params(model):
        params = model.get_params(deep=False)
        for k in list(params):
            if hasattr(model, f"named_{k}"):
                params[k] = getattr(model, f"named_{k}")
        return params

    def _serialize(obj):
        if isinstance(obj, list):
            return [_serialize(o) for o in obj]
        elif isinstance(obj, tuple):
            return tuple(_serialize(o) for o in obj)
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        elif isinstance(obj, Kernel):
            return _serialize_kernel(obj)
        elif isinstance(obj, BaseEstimator):
            if isinstance(obj, GaussianProcessClassifier):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    return _serialize(
                        _get_params(obj) | {"type": get_sklearn_public_path(type(obj))}
                    )
            return _serialize(
                _get_params(obj) | {"type": get_sklearn_public_path(type(obj))}
            )
        else:
            return obj

    return _serialize(_get_params(model))  # type: ignore


def set_model_params(model: BaseEstimator, params: dict[str, Any]) -> BaseEstimator:
    """Recursively set hyperparameters on an sklearn estimator."""
    params.pop("type", None)
    # Make TransformedTargetRegressor transparent to dotted-path param setting:
    # route params to the wrapped regressor so callers don't need to know about
    # the target scaler. When the params *do* address the wrapper explicitly
    # (e.g. a serialised state carrying "regressor"/"transformer"), fall through.
    if isinstance(model, TransformedTargetRegressor) and not (
        {"regressor", "transformer"} & set(params)
    ):
        set_model_params(cast(BaseEstimator, model.regressor), params)
        return model
    keys = list(params.keys())
    for key in keys:
        if hasattr(model, f"named_{key}"):
            attr = getattr(model, f"named_{key}")
        else:
            attr = getattr(model, key)
        if isinstance(attr, BaseEstimator):
            set_model_params(attr, params.pop(key))
        elif isinstance(attr, dict):
            values = params[key]
            for k, v in attr.items():
                if isinstance(v, BaseEstimator):
                    if k in values:
                        set_model_params(v, values.pop(k))
            if not values:
                params.pop(key)
    return model.set_params(**params)  # type: ignore


def _rebuild_estimator_from_params(params: dict[str, Any]) -> BaseEstimator:
    """Instantiate a sklearn estimator tree from a params dict."""
    params = dict(params)
    type_path: str = params.pop("type")
    module_path, _, class_name = type_path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)

    is_pipeline = issubclass(cls, Pipeline)

    init_params: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, dict) and "type" in v and "kernels." in v["type"]:
            init_params[k] = _deserialize_kernel(v)
        elif isinstance(v, dict) and "type" in v:
            init_params[k] = _rebuild_estimator_from_params(v)
        elif is_pipeline and k == "steps" and isinstance(v, dict):
            init_params[k] = [
                (name, _rebuild_estimator_from_params(step_params))
                for name, step_params in v.items()
            ]
        elif isinstance(v, list):
            rebuilt = []
            for item in v:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[1], dict)
                ):
                    rebuilt.append((item[0], _rebuild_estimator_from_params(item[1])))
                else:
                    rebuilt.append(item)
            init_params[k] = rebuilt
        else:
            init_params[k] = v

    init_params = _fix_json_param_types(cls, init_params)
    return cls(**init_params)
