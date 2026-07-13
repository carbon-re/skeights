"""MLP estimator serialization (MLPRegressor, MLPClassifier)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Non-array fitted attributes on MLP that must be persisted in JSON
# (beyond coefs_/intercepts_ which go to safetensors).
MLP_FITTED_STATE_ATTRS = (
    "n_layers_",
    "n_outputs_",
    "out_activation_",
    "n_iter_",
    "t_",
    "loss_",
    "best_loss_",
    "n_iter_no_change",
)


def handles(estimator: BaseEstimator) -> bool:
    """Return True if this module handles the given estimator type."""
    return isinstance(estimator, (MLPRegressor, MLPClassifier))


def collect_state(
    estimator: BaseEstimator, prefix: str, format: str | None = None
) -> dict[str, Any]:
    """Collect non-array fitted state from an MLP estimator."""
    state: dict[str, Any] = {}
    for attr in MLP_FITTED_STATE_ATTRS:
        if hasattr(estimator, attr):
            state[f"{prefix}{attr}"] = getattr(estimator, attr)
    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    """Restore non-array fitted state onto an MLP estimator."""
    for attr in MLP_FITTED_STATE_ATTRS:
        key = f"{prefix}{attr}"
        if key in fitted_state:
            setattr(estimator, attr, fitted_state[key])


def extract_arrays(
    estimator: BaseEstimator, prefix: str, format: str | None = None
) -> dict[str, np.ndarray]:
    """Extract weight arrays from an MLP estimator."""
    arrays: dict[str, np.ndarray] = {}
    if hasattr(estimator, "coefs_"):
        for i, w in enumerate(estimator.coefs_):  # type: ignore[union-attr]
            arrays[f"{prefix}coefs_{i}"] = np.asarray(w)
    if hasattr(estimator, "intercepts_"):
        for i, b in enumerate(estimator.intercepts_):  # type: ignore[union-attr]
            arrays[f"{prefix}intercepts_{i}"] = np.asarray(b)
    return arrays


def restore_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
    fitted_state: dict[str, Any] | None = None,
) -> None:
    """Restore weight arrays onto an MLP estimator."""
    coefs = []
    intercepts = []
    i = 0
    while f"{prefix}coefs_{i}" in arrays:
        coefs.append(arrays[f"{prefix}coefs_{i}"])
        i += 1
    i = 0
    while f"{prefix}intercepts_{i}" in arrays:
        intercepts.append(arrays[f"{prefix}intercepts_{i}"])
        i += 1
    if coefs:
        estimator.coefs_ = coefs  # type: ignore[union-attr]
    if intercepts:
        estimator.intercepts_ = intercepts  # type: ignore[union-attr]
