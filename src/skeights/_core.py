"""Core dispatch for serializing fitted scikit-learn estimators.

Decomposes a fitted sklearn estimator into:
- A JSON-safe dict of hyperparameters and fitted scalar state
- A flat dict of numpy arrays (weights, fitted arrays)

This pair can be saved to safetensors + JSON via the top-level
:func:`skeights.save` / :func:`skeights.load` functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from skeights._utils import SKLEARN_ARRAY_ATTRS as _SKLEARN_ARRAY_ATTRS


def _get_handlers():
    """Lazy-load estimator handler modules to avoid circular imports."""
    from skeights import _gp, _hgb, _lgbm, _mlp, _trees, _ttr

    return [_trees, _ttr, _mlp, _gp, _hgb, _lgbm]


# ---------------------------------------------------------------------------
# Dispatch: collect / restore fitted state and arrays
# ---------------------------------------------------------------------------


def _collect_fitted_state(estimator: BaseEstimator, prefix: str = "") -> dict[str, Any]:
    """Collect non-array fitted attributes from estimator instances.

    Walks Pipeline steps recursively. Dispatches to estimator-family
    modules.
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
    """Restore non-array fitted attributes onto an estimator."""
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
    """Recursively extract numpy arrays from a fitted estimator."""
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
            f"Unsupported estimator type '{type(estimator).__name__}'. "
            f"See README for supported types."
        )

    result: dict[str, np.ndarray] = {}
    for attr in _SKLEARN_ARRAY_ATTRS:
        if hasattr(estimator, attr):
            result[f"{prefix}{attr}"] = np.asarray(getattr(estimator, attr))
    return result


def _restore_estimator_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str = "",
    fitted_state: dict[str, Any] | None = None,
) -> None:
    """Restore numpy arrays onto a fitted skeleton estimator."""
    if isinstance(estimator, Pipeline):
        for step_name, step in estimator.named_steps.items():
            step_prefix = f"{prefix}{step_name}/" if prefix else f"{step_name}/"
            _restore_estimator_arrays(
                step,
                arrays,
                prefix=step_prefix,
                fitted_state=fitted_state,
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
