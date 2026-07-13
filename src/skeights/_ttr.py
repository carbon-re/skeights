"""TransformedTargetRegressor serialization."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor


def handles(estimator: BaseEstimator) -> bool:
    """Return True if this module handles the given estimator type."""
    return isinstance(estimator, TransformedTargetRegressor)


def collect_state(estimator: BaseEstimator, prefix: str, format: str | None = None) -> dict[str, Any]:
    """Collect non-array fitted state from a TTR estimator."""
    from skeights._core import _collect_fitted_state

    assert isinstance(estimator, TransformedTargetRegressor)
    state: dict[str, Any] = {}
    if hasattr(estimator, "_training_dim"):
        state[f"{prefix}_training_dim"] = estimator._training_dim
    if hasattr(estimator, "regressor_"):
        state.update(
            _collect_fitted_state(
                cast(BaseEstimator, estimator.regressor_),
                prefix=f"{prefix}regressor_/",
                format=format,
            )
        )
    if hasattr(estimator, "transformer_"):
        state.update(
            _collect_fitted_state(
                cast(BaseEstimator, estimator.transformer_),
                prefix=f"{prefix}transformer_/",
                format=format,
            )
        )
    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    """Restore non-array fitted state onto a TTR estimator."""
    from skeights._core import _restore_fitted_state

    assert isinstance(estimator, TransformedTargetRegressor)
    key = f"{prefix}_training_dim"
    if key in fitted_state:
        estimator._training_dim = fitted_state[key]
    if hasattr(estimator, "regressor_"):
        _restore_fitted_state(
            cast(BaseEstimator, estimator.regressor_),
            fitted_state,
            prefix=f"{prefix}regressor_/",
        )
    if hasattr(estimator, "transformer_"):
        _restore_fitted_state(
            cast(BaseEstimator, estimator.transformer_),
            fitted_state,
            prefix=f"{prefix}transformer_/",
        )


def extract_arrays(estimator: BaseEstimator, prefix: str, format: str | None = None) -> dict[str, np.ndarray]:
    """Extract arrays from a TTR estimator."""
    from skeights._core import _arrays_from_estimator

    assert isinstance(estimator, TransformedTargetRegressor)
    arrays: dict[str, np.ndarray] = {}
    if hasattr(estimator, "regressor_"):
        arrays.update(
            _arrays_from_estimator(
                cast(BaseEstimator, estimator.regressor_),
                prefix=f"{prefix}regressor_/",
                format=format,
            )
        )
    if hasattr(estimator, "transformer_"):
        arrays.update(
            _arrays_from_estimator(
                cast(BaseEstimator, estimator.transformer_),
                prefix=f"{prefix}transformer_/",
                format=format,
            )
        )
    return arrays


def restore_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
    fitted_state: dict[str, Any] | None = None,
) -> None:
    """Restore arrays onto a TTR estimator."""
    from sklearn.base import clone
    from sklearn.discriminant_analysis import StandardScaler

    from skeights._core import _restore_estimator_arrays

    assert isinstance(estimator, TransformedTargetRegressor)
    reg_prefix = f"{prefix}regressor_/"
    if any(k.startswith(reg_prefix) for k in arrays):
        if not hasattr(estimator, "regressor_"):
            estimator.regressor_ = clone(estimator.regressor)
        _restore_estimator_arrays(
            cast(BaseEstimator, estimator.regressor_),
            arrays,
            prefix=reg_prefix,
        )
    tr_prefix = f"{prefix}transformer_/"
    if any(k.startswith(tr_prefix) for k in arrays):
        if not hasattr(estimator, "transformer_"):
            template = (
                estimator.transformer
                if estimator.transformer is not None
                else StandardScaler()
            )
            estimator.transformer_ = clone(template)
        _restore_estimator_arrays(
            cast(BaseEstimator, estimator.transformer_),
            arrays,
            prefix=tr_prefix,
        )
