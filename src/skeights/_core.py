"""Core serialization for fitted scikit-learn estimators.

Decomposes a fitted sklearn estimator into:
- A JSON-safe dict of hyperparameters and fitted scalar state
- A flat dict of numpy arrays (weights, fitted arrays)

This pair can be saved to safetensors + JSON via the top-level
:func:`skeights.save` / :func:`skeights.load` functions.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Self

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from skeights._utils import (
    SKLEARN_ARRAY_ATTRS as _SKLEARN_ARRAY_ATTRS,
)
from skeights._utils import (
    _fix_json_param_types,
    get_sklearn_public_path,
)

# Supported inner sklearn scalers. Extend when new scaler types are wrapped.
BaseScaler = StandardScaler | MinMaxScaler | RobustScaler


def _get_handlers():
    """Lazy-load estimator handler modules to avoid circular imports."""
    from skeights import _gp, _hgb, _mlp, _trees, _ttr

    return [_trees, _ttr, _mlp, _gp, _hgb]


# ---------------------------------------------------------------------------
# Feature / weight inspection
# ---------------------------------------------------------------------------


def _get_features_from_estimator(model: BaseEstimator) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)  # type: ignore[attr-defined]
    if isinstance(model, Pipeline):
        for step in model.named_steps.values():
            try:
                return _get_features_from_estimator(step)
            except ValueError:
                continue
        raise ValueError("No step in the pipeline has features.")
    raise ValueError(f"Unsupported model type: {type(model)}")


def _get_weights_from_estimator(model: BaseEstimator) -> dict[str, Any]:
    if isinstance(model, Pipeline):
        return {
            "steps": {
                k: _get_weights_from_estimator(v) for k, v in model.named_steps.items()
            }
        }

    fields = {
        "coefficients": "coef_",
        "intercept": "intercept_",
        "feature_importances": "feature_importances_",
        "means": "mean_",
        "scales": "scale_",
        "variances": "var_",
    }

    def _serialize(x: Any) -> Any:
        if hasattr(x, "tolist"):
            return x.tolist()
        return x

    weights: dict[str, Any] = {}
    for key, attr in fields.items():
        if hasattr(model, attr):
            weights[key] = _serialize(getattr(model, attr))
    return weights


# ---------------------------------------------------------------------------
# Dispatch: collect / restore fitted state and arrays
# ---------------------------------------------------------------------------


def _collect_fitted_state(estimator: BaseEstimator, prefix: str = "") -> dict[str, Any]:
    """Collect non-array fitted attributes from estimator instances.

    Walks Pipeline steps recursively. Dispatches to estimator-family modules.
    """
    state: dict[str, Any] = {}

    if isinstance(estimator, Pipeline):
        for step_name, step in estimator.named_steps.items():
            step_prefix = f"{prefix}{step_name}/" if prefix else f"{step_name}/"
            state.update(_collect_fitted_state(step, prefix=step_prefix))
        return state

    for handler in _get_handlers():
        if handler.handles(estimator):
            return handler.collect_state(estimator, prefix)

    return state


def _restore_fitted_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str = "",
) -> None:
    """Restore non-array fitted attributes onto an estimator in-place."""
    if isinstance(estimator, Pipeline):
        for step_name, step in estimator.named_steps.items():
            step_prefix = f"{prefix}{step_name}/" if prefix else f"{step_name}/"
            _restore_fitted_state(step, fitted_state, prefix=step_prefix)
        return

    for handler in _get_handlers():
        if handler.handles(estimator):
            handler.restore_state(estimator, fitted_state, prefix)
            return


def _arrays_from_estimator(
    estimator: BaseEstimator, prefix: str = ""
) -> dict[str, np.ndarray]:
    """Recursively extract numpy arrays from a fitted sklearn estimator."""
    if isinstance(estimator, Pipeline):
        arrays: dict[str, np.ndarray] = {}
        for step_name, step in estimator.named_steps.items():
            step_prefix = f"{prefix}{step_name}/" if prefix else f"{step_name}/"
            arrays.update(_arrays_from_estimator(step, prefix=step_prefix))
        return arrays

    for handler in _get_handlers():
        if handler.handles(estimator):
            return handler.extract_arrays(estimator, prefix)

    # Generic fallback for linear models, scalers, etc.
    supported = any(hasattr(estimator, attr) for attr in _SKLEARN_ARRAY_ATTRS)
    if not supported:
        raise NotImplementedError(
            f"_arrays_from_estimator does not support estimator type "
            f"'{type(estimator).__name__}'. Only linear models, scalers, "
            f"MLPRegressor, GaussianProcess*, HistGradientBoosting*, "
            f"RandomForest*, GradientBoosting*, "
            f"and Pipelines of those are supported."
        )

    result: dict[str, np.ndarray] = {}
    for attr in _SKLEARN_ARRAY_ATTRS:
        if hasattr(estimator, attr):
            val = getattr(estimator, attr)
            result[f"{prefix}{attr}"] = np.asarray(val)
    return result


def _restore_estimator_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str = "",
    fitted_state: dict[str, Any] | None = None,
) -> None:
    """Restore numpy arrays onto a fitted skeleton estimator in-place."""
    if isinstance(estimator, Pipeline):
        for step_name, step in estimator.named_steps.items():
            step_prefix = f"{prefix}{step_name}/" if prefix else f"{step_name}/"
            _restore_estimator_arrays(
                step, arrays, prefix=step_prefix, fitted_state=fitted_state
            )
        return

    for handler in _get_handlers():
        if handler.handles(estimator):
            handler.restore_arrays(estimator, arrays, prefix, fitted_state)
            return

    # Generic fallback for linear models, scalers, etc.
    for attr in _SKLEARN_ARRAY_ATTRS:
        key = f"{prefix}{attr}"
        if key in arrays:
            setattr(estimator, attr, arrays[key])


# ---------------------------------------------------------------------------
# SklearnScaler
# ---------------------------------------------------------------------------


class SklearnScaler:
    """Wraps an sklearn scaler with serialization support."""

    _registry_key: str = "SklearnScaler"
    _array_attrs: tuple[str, ...] = (
        "center_",
        "mean_",
        "n_samples_seen_",
        "scale_",
        "var_",
        "min_",
        "data_min_",
        "data_max_",
        "data_range_",
    )

    def __init__(self, scaler: BaseScaler) -> None:
        self.scaler = scaler

    def fit(self, X: pd.DataFrame) -> SklearnScaler:
        self.scaler.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaler.transform(X.values),
            index=X.index,
            columns=X.columns,
        )

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaler.inverse_transform(X.values),
            index=X.index,
            columns=X.columns,
        )

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.scaler.transform(X))

    def inverse_transform_numpy(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.scaler.inverse_transform(X))

    def get_arrays(self) -> dict[str, np.ndarray]:
        """Return fitted scaler arrays as a flat dict of numpy arrays."""
        return {
            attr: np.asarray(getattr(self.scaler, attr))
            for attr in self._array_attrs
            if hasattr(self.scaler, attr)
        }

    def get_state(self) -> dict[str, Any]:
        """Return JSON-safe config and type info for this scaler."""
        params = self.scaler.get_params()
        return {
            "type": self._registry_key,
            "inner_type": get_sklearn_public_path(type(self.scaler)),
            "init_params": {k: v for k, v in params.items() if not k.endswith("_")},
        }

    @classmethod
    def from_state(cls, state: dict, arrays: dict[str, np.ndarray]) -> SklearnScaler:
        """Reconstruct a fitted SklearnScaler from state and arrays."""
        inner_module, _, inner_class = state["inner_type"].rpartition(".")
        inner_cls = getattr(importlib.import_module(inner_module), inner_class)
        init_params = state.get("init_params", {})
        init_params = _fix_json_param_types(inner_cls, init_params)
        inner = inner_cls(**init_params)
        for attr, arr in arrays.items():
            setattr(inner, attr, arr)
        return cls(inner)


# ---------------------------------------------------------------------------
# SklearnModel
# ---------------------------------------------------------------------------


class SklearnModel:
    """Wraps a fitted sklearn estimator with serialization support.

    Decomposes the estimator into a JSON-safe state dict and a flat dict
    of numpy arrays, suitable for saving to safetensors + JSON.
    """

    _registry_key: str = "SklearnModel"
    use_predict_proba: bool = False

    def __init__(self, model: BaseEstimator, *, use_predict_proba: bool = False):
        self.model = model
        self.use_predict_proba = use_predict_proba
        self.targets: list[str] | None = None
        self.features: list[str] | None = None
        self.is_fitted_ = False

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None, **kwargs
    ) -> SklearnModel:
        if y is not None and y.shape[1] == 1:
            self.model.fit(X, y.iloc[:, 0], **kwargs)  # type: ignore
        else:
            self.model.fit(X, y, **kwargs)  # type: ignore
        self.features = X.columns.to_list()
        if y is not None:
            self.targets = y.columns.to_list()
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.features is not None:
            X = X.reindex(columns=self.features)
        return pd.DataFrame(
            index=X.index,
            data=self.predict_numpy(X.values),
            columns=self.targets,  # type: ignore[arg-type]
        )

    def predict_numpy(self, X: np.ndarray) -> np.ndarray:
        assert self.targets is not None, "Model must be fitted before prediction."
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            with sklearn.config_context(
                assume_finite=True, skip_parameter_validation=True
            ):  # type: ignore
                if self.use_predict_proba:
                    proba = self.model.predict_proba(X)  # type: ignore
                    y = proba[:, 1:] if proba.shape[1] == 2 else proba
                else:
                    y = self.model.predict(X)  # type: ignore
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def get_features(self) -> list[str]:
        if self.features is not None:
            return self.features
        return _get_features_from_estimator(self.model)

    def get_weights(self) -> dict[str, Any]:
        return _get_weights_from_estimator(self.model)

    def set_params(self, **params) -> SklearnModel:
        from skeights._params import set_model_params

        set_model_params(self.model, params)
        return self

    def get_params(self) -> dict[str, Any]:
        from skeights._params import get_model_params

        return get_model_params(self.model)

    def get_targets(self) -> list[str]:
        if self.targets is not None:
            return self.targets
        raise ValueError("Targets are not set. Fit the model first.")

    def get_arrays(self) -> dict[str, np.ndarray]:
        """Return numeric weights as a flat dict of numpy arrays."""
        return _arrays_from_estimator(self.model)

    def get_state(self) -> dict:
        """Return JSON-safe config and fitted state for this model."""
        from skeights._params import get_model_params

        params = get_model_params(self.model)
        if "type" not in params:
            params["type"] = get_sklearn_public_path(type(self.model))
        return {
            "type": self._registry_key,
            "features": self.features,
            "targets": self.targets,
            "use_predict_proba": self.use_predict_proba,
            "model_params": params,
            "fitted_state": _collect_fitted_state(self.model),
        }

    @classmethod
    def from_state(cls, state: dict, arrays: dict[str, np.ndarray]) -> Self:
        """Reconstruct a fitted SklearnModel from state and arrays."""
        from skeights._params import _rebuild_estimator_from_params

        estimator = _rebuild_estimator_from_params(state["model_params"])
        fitted_state = state.get("fitted_state") or state.get("mlp_fitted_state", {})
        _restore_estimator_arrays(estimator, arrays, fitted_state=fitted_state or None)
        if fitted_state:
            _restore_fitted_state(estimator, fitted_state)
        obj = cls.__new__(cls)
        obj.model = estimator
        obj.features = state["features"]
        obj.targets = state["targets"]
        obj.use_predict_proba = bool(state.get("use_predict_proba", False))
        obj.is_fitted_ = True
        return obj


# Re-export param functions at module level for backward compat with __init__.py
def get_model_params(model: BaseEstimator) -> dict[str, Any]:
    """Recursively extract hyperparameters from a fitted sklearn estimator."""
    from skeights._params import get_model_params as _get_model_params

    return _get_model_params(model)


def set_model_params(model: BaseEstimator, params: dict[str, Any]) -> BaseEstimator:
    """Recursively set hyperparameters on an sklearn estimator."""
    from skeights._params import set_model_params as _set_model_params

    return _set_model_params(model, params)


# Re-export kernel functions for backward compatibility (tests import from _core).
def _serialize_kernel(kernel: Any) -> dict[str, Any]:
    from skeights._kernels import _serialize_kernel as _sk

    return _sk(kernel)


def _deserialize_kernel(data: dict[str, Any]) -> Any:
    from skeights._kernels import _deserialize_kernel as _dk

    return _dk(data)
