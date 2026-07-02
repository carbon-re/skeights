"""Gaussian Process estimator serialization (GPR, GPC)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)

from skeights._kernels import _deserialize_kernel, _serialize_kernel


def handles(estimator: BaseEstimator) -> bool:
    """Return True if this module handles the given estimator type."""
    return isinstance(estimator, (GaussianProcessRegressor, GaussianProcessClassifier))


def collect_state(estimator: BaseEstimator, prefix: str) -> dict[str, Any]:
    """Collect non-array fitted state from a GP estimator."""
    state: dict[str, Any] = {}
    if hasattr(estimator, "log_marginal_likelihood_value_"):
        state[f"{prefix}log_marginal_likelihood_value_"] = (
            estimator.log_marginal_likelihood_value_  # type: ignore[union-attr]
        )
    if isinstance(estimator, GaussianProcessRegressor):
        if hasattr(estimator, "kernel_"):
            state[f"{prefix}kernel_"] = _serialize_kernel(
                estimator.kernel_  # type: ignore[arg-type]
            )
    else:
        assert isinstance(estimator, GaussianProcessClassifier)
        if hasattr(estimator, "n_classes_"):
            state[f"{prefix}n_classes_"] = estimator.n_classes_
        if hasattr(estimator, "base_estimator_"):
            be_prefix = f"{prefix}base_estimator_/"
            be = estimator.base_estimator_
            if hasattr(be, "kernel_"):
                state[f"{be_prefix}kernel_"] = _serialize_kernel(
                    be.kernel_  # type: ignore[union-attr]
                )
            if hasattr(be, "log_marginal_likelihood_value_"):
                state[f"{be_prefix}log_marginal_likelihood_value_"] = (
                    be.log_marginal_likelihood_value_  # type: ignore[union-attr]
                )
    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    """Restore non-array fitted state onto a GP estimator."""
    lml_key = f"{prefix}log_marginal_likelihood_value_"
    if lml_key in fitted_state:
        estimator.log_marginal_likelihood_value_ = fitted_state[lml_key]  # type: ignore[union-attr]
    if isinstance(estimator, GaussianProcessRegressor):
        kernel_key = f"{prefix}kernel_"
        if kernel_key in fitted_state:
            estimator.kernel_ = _deserialize_kernel(fitted_state[kernel_key])
    else:
        assert isinstance(estimator, GaussianProcessClassifier)
        n_classes_key = f"{prefix}n_classes_"
        if n_classes_key in fitted_state:
            estimator.n_classes_ = fitted_state[n_classes_key]
        be_prefix = f"{prefix}base_estimator_/"
        if hasattr(estimator, "base_estimator_"):
            be = estimator.base_estimator_
            be_kernel_key = f"{be_prefix}kernel_"
            if be_kernel_key in fitted_state:
                be.kernel_ = _deserialize_kernel(  # type: ignore[union-attr]
                    fitted_state[be_kernel_key]
                )
            be_lml_key = f"{be_prefix}log_marginal_likelihood_value_"
            if be_lml_key in fitted_state:
                be.log_marginal_likelihood_value_ = fitted_state[be_lml_key]  # type: ignore[union-attr]


def extract_arrays(estimator: BaseEstimator, prefix: str) -> dict[str, np.ndarray]:
    """Extract arrays from a GP estimator."""

    def _key(name: str) -> str:
        return f"{prefix}{name}"

    arrays: dict[str, np.ndarray] = {}
    if isinstance(estimator, GaussianProcessRegressor):
        for attr in (
            "X_train_",
            "alpha_",
            "L_",
            "_y_train_mean",
            "_y_train_std",
        ):
            if hasattr(estimator, attr):
                arrays[_key(attr)] = np.asarray(getattr(estimator, attr))
        return arrays
    assert isinstance(estimator, GaussianProcessClassifier)
    if hasattr(estimator, "classes_"):
        arrays[_key("classes_")] = np.asarray(estimator.classes_)
    if hasattr(estimator, "base_estimator_"):
        be_prefix = f"{prefix}base_estimator_/"
        arrays.update(
            _extract_base_estimator_arrays(estimator.base_estimator_, be_prefix)
        )
    return arrays


def _extract_base_estimator_arrays(
    estimator: BaseEstimator, prefix: str
) -> dict[str, np.ndarray]:
    """Extract arrays from a GPC base_estimator_ (binary GPR-like)."""
    from skeights._utils import SKLEARN_ARRAY_ATTRS as _SKLEARN_ARRAY_ATTRS

    arrays: dict[str, np.ndarray] = {}
    for attr in _SKLEARN_ARRAY_ATTRS:
        if hasattr(estimator, attr):
            arrays[f"{prefix}{attr}"] = np.asarray(getattr(estimator, attr))
    return arrays


def restore_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
    fitted_state: dict[str, Any] | None = None,
) -> None:
    """Restore arrays onto a GP estimator."""
    if isinstance(estimator, GaussianProcessRegressor):
        for attr in (
            "X_train_",
            "alpha_",
            "L_",
            "_y_train_mean",
            "_y_train_std",
        ):
            key = f"{prefix}{attr}"
            if key in arrays:
                val = arrays[key]
                if attr in ("_y_train_mean", "_y_train_std") and val.ndim == 0:
                    setattr(estimator, attr, val.item())
                else:
                    setattr(estimator, attr, val)
        return

    assert isinstance(estimator, GaussianProcessClassifier)
    classes_key = f"{prefix}classes_"
    if classes_key in arrays:
        estimator.classes_ = arrays[classes_key]
    be_prefix = f"{prefix}base_estimator_/"
    any_be_array = any(k.startswith(be_prefix) for k in arrays)
    if any_be_array:
        if not hasattr(estimator, "base_estimator_"):
            from sklearn.gaussian_process._gpc import (
                _BinaryGaussianProcessClassifierLaplace,
            )

            estimator.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(
                kernel=estimator.kernel,
                optimizer=estimator.optimizer,
                n_restarts_optimizer=estimator.n_restarts_optimizer,
                max_iter_predict=estimator.max_iter_predict,
                warm_start=estimator.warm_start,
                copy_X_train=estimator.copy_X_train,
                random_state=estimator.random_state,
            )
        _restore_base_estimator_arrays(estimator.base_estimator_, arrays, be_prefix)


def _restore_base_estimator_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> None:
    """Restore arrays onto a GPC base_estimator_."""
    from skeights._utils import SKLEARN_ARRAY_ATTRS as _SKLEARN_ARRAY_ATTRS

    for attr in _SKLEARN_ARRAY_ATTRS:
        key = f"{prefix}{attr}"
        if key in arrays:
            setattr(estimator, attr, arrays[key])
