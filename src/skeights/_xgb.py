"""XGBoost serialization.

Stores the booster model as a JSON dict (XGBoost's own JSON format)
in the state. The model JSON is a nested dict, not a string blob,
so it's directly inspectable and diffable.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


def _is_xgb(estimator: BaseEstimator) -> bool:
    try:
        from xgboost import XGBClassifier, XGBRegressor

        return isinstance(estimator, (XGBRegressor, XGBClassifier))
    except ImportError:
        return False


def _is_classifier(estimator: BaseEstimator) -> bool:
    try:
        from xgboost import XGBClassifier

        return isinstance(estimator, XGBClassifier)
    except ImportError:
        return False


def handles(estimator: BaseEstimator) -> bool:
    return _is_xgb(estimator)


def collect_state(estimator: BaseEstimator, prefix: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    booster = estimator.get_booster()  # type: ignore[attr-defined]
    model_json = json.loads(booster.save_raw(raw_format="json").decode())
    state[f"{prefix}model_json"] = model_json

    if _is_classifier(estimator):
        state[f"{prefix}n_classes"] = estimator.n_classes_  # type: ignore[attr-defined]

    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    import xgboost as xgb

    model_json = fitted_state[f"{prefix}model_json"]
    model_bytes = bytearray(json.dumps(model_json).encode())
    booster = xgb.Booster()
    booster.load_model(model_bytes)
    estimator._Booster = booster  # type: ignore[attr-defined]

    if _is_classifier(estimator):
        estimator.n_classes_ = fitted_state[f"{prefix}n_classes"]  # type: ignore[attr-defined]


def extract_arrays(estimator: BaseEstimator, prefix: str) -> dict[str, np.ndarray]:
    # Model lives in JSON state; arrays only has feature importances
    arrays: dict[str, np.ndarray] = {}
    if hasattr(estimator, "feature_importances_"):
        arrays[f"{prefix}feature_importances_"] = np.asarray(
            estimator.feature_importances_  # type: ignore[attr-defined]
        )
    return arrays


def restore_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
    fitted_state: dict[str, Any] | None = None,
) -> None:
    # Feature importances are recomputed from the booster on access.
    pass
