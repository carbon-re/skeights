"""LightGBM serialization.

Stores the booster model string (LightGBM's own text-based portability
format) in the JSON state, and fitted sklearn wrapper attributes needed
for predict/predict_proba to work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


def _is_lgbm(estimator: BaseEstimator) -> bool:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        return isinstance(estimator, (LGBMRegressor, LGBMClassifier))
    except ImportError:
        return False


def _is_classifier(estimator: BaseEstimator) -> bool:
    try:
        from lightgbm import LGBMClassifier

        return isinstance(estimator, LGBMClassifier)
    except ImportError:
        return False


def handles(estimator: BaseEstimator) -> bool:
    return _is_lgbm(estimator)


def collect_state(estimator: BaseEstimator, prefix: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    state[f"{prefix}model_str"] = estimator._Booster.model_to_string()  # type: ignore[attr-defined]
    state[f"{prefix}n_features"] = estimator._n_features  # type: ignore[attr-defined]
    state[f"{prefix}n_features_in"] = estimator._n_features_in  # type: ignore[attr-defined]
    state[f"{prefix}objective"] = estimator._objective  # type: ignore[attr-defined]
    state[f"{prefix}best_iteration"] = estimator._best_iteration  # type: ignore[attr-defined]

    if _is_classifier(estimator):
        state[f"{prefix}n_classes"] = estimator._n_classes  # type: ignore[attr-defined]
        state[f"{prefix}class_map"] = {
            int(k): int(v)
            for k, v in estimator._class_map.items()  # type: ignore[attr-defined]
        }

    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    import lightgbm as lgb

    model_str = fitted_state[f"{prefix}model_str"]
    estimator._Booster = lgb.Booster(model_str=model_str)  # type: ignore[attr-defined]
    estimator._n_features = fitted_state[f"{prefix}n_features"]  # type: ignore[attr-defined]
    estimator._n_features_in = fitted_state[f"{prefix}n_features_in"]  # type: ignore[attr-defined]
    estimator._objective = fitted_state[f"{prefix}objective"]  # type: ignore[attr-defined]
    estimator._best_iteration = fitted_state[f"{prefix}best_iteration"]  # type: ignore[attr-defined]
    estimator.fitted_ = True  # type: ignore[attr-defined]
    estimator._best_score = {}  # type: ignore[attr-defined]
    estimator._evals_result = {}  # type: ignore[attr-defined]
    estimator._other_params = {}  # type: ignore[attr-defined]

    if _is_classifier(estimator):
        from sklearn.preprocessing import LabelEncoder

        estimator._n_classes = fitted_state[f"{prefix}n_classes"]  # type: ignore[attr-defined]
        estimator._class_map = fitted_state[f"{prefix}class_map"]  # type: ignore[attr-defined]
        n_classes = estimator._n_classes  # type: ignore[attr-defined]
        estimator._classes = np.arange(n_classes)  # type: ignore[attr-defined]
        le = LabelEncoder()
        le.classes_ = np.arange(n_classes)
        estimator._le = le  # type: ignore[attr-defined]


def extract_arrays(estimator: BaseEstimator, prefix: str) -> dict[str, np.ndarray]:
    # Model string goes in state; arrays only has feature importances
    arrays: dict[str, np.ndarray] = {}
    if hasattr(estimator, "feature_importances_"):
        arrays[f"{prefix}feature_importances_"] = np.asarray(
            estimator.feature_importances_  # type: ignore[attr-defined]
        )
    if _is_classifier(estimator) and hasattr(estimator, "_classes"):
        arrays[f"{prefix}classes_"] = np.asarray(
            estimator._classes  # type: ignore[attr-defined]
        )
    return arrays


def restore_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
    fitted_state: dict[str, Any] | None = None,
) -> None:
    # Feature importances are recomputed from the booster on access,
    # so nothing to restore from arrays. Classes are restored in
    # restore_state from the fitted_state.
    pass
