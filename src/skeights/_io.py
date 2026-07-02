"""Save and load fitted sklearn estimators to safetensors + JSON."""

from __future__ import annotations

import json
from pathlib import Path

import safetensors.numpy
from sklearn.base import BaseEstimator

from skeights._core import (
    _arrays_from_estimator,
    _collect_fitted_state,
    _restore_estimator_arrays,
    _restore_fitted_state,
)
from skeights._params import (
    _rebuild_estimator_from_params,
    get_model_params,
)
from skeights._utils import get_sklearn_public_path


def save(
    estimator: BaseEstimator,
    arrays_path: str | Path,
    state_path: str | Path,
) -> None:
    """Serialize a fitted sklearn estimator to safetensors + JSON.

    Args:
        estimator: A fitted sklearn estimator (Pipeline, Ridge, etc.).
        arrays_path: Destination path for the safetensors weight file.
        state_path: Destination path for the JSON state file.
    """
    arrays = _arrays_from_estimator(estimator)
    params = get_model_params(estimator)
    if "type" not in params:
        params["type"] = get_sklearn_public_path(type(estimator))
    state = {
        "model_params": params,
        "fitted_state": _collect_fitted_state(estimator),
    }
    safetensors.numpy.save_file(arrays, str(arrays_path))
    Path(state_path).write_text(json.dumps(state, indent=2))


def load(
    arrays_path: str | Path,
    state_path: str | Path,
) -> BaseEstimator:
    """Reconstruct a fitted sklearn estimator from safetensors + JSON.

    Args:
        arrays_path: Path to the safetensors weight file.
        state_path: Path to the JSON state file.

    Returns:
        A fully reconstructed, ready-to-predict sklearn estimator.
    """
    state = json.loads(Path(state_path).read_text())
    arrays = dict(safetensors.numpy.load_file(str(arrays_path)))
    estimator = _rebuild_estimator_from_params(state["model_params"])
    fitted_state = state.get("fitted_state", {})
    _restore_estimator_arrays(estimator, arrays, fitted_state=fitted_state or None)
    if fitted_state:
        _restore_fitted_state(estimator, fitted_state)
    return estimator
