"""Shared utilities used by both _core.py and estimator modules.

This module must not import from _core.py or any estimator module
to avoid circular imports.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from sklearn.base import BaseEstimator


def get_sklearn_public_path(cls: type) -> str:
    """Return the stable public import path for an sklearn estimator class.

    sklearn places concrete classes in private submodules (e.g.
    ``sklearn.preprocessing._data.StandardScaler``) but re-exports them
    from the public package (``sklearn.preprocessing.StandardScaler``).
    This walks up the hierarchy, verifying each candidate via ``getattr``,
    and falls back to the full private path if no public re-export is found.
    """
    parts = cls.__module__.split(".")
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("_"):
            continue
        candidate = ".".join(parts[:i])
        try:
            mod = importlib.import_module(candidate)
            if getattr(mod, cls.__name__, None) is cls:
                return f"{candidate}.{cls.__name__}"
        except ImportError:
            continue
    full_path = f"{cls.__module__}.{cls.__name__}"
    warnings.warn(
        f"No public re-export found for {cls.__name__}; using private path "
        f"'{full_path}'. This may break if sklearn reorganises internal modules.",
        stacklevel=2,
    )
    return full_path


def _fix_json_param_types(
    cls: type[BaseEstimator], params: dict[str, Any]
) -> dict[str, Any]:
    """Convert list params back to tuples where sklearn expects tuples.

    JSON serialization turns tuples into lists. sklearn's parameter
    validation rejects lists for params declared as tuple (e.g.
    RobustScaler.quantile_range, MinMaxScaler.feature_range).
    """
    try:
        defaults = cls().get_params()
    except TypeError:
        return params
    return {
        k: tuple(v) if isinstance(v, list) and isinstance(defaults.get(k), tuple) else v
        for k, v in params.items()
    }


# Array attrs used by the generic linear/scaler fallback and by GP base_estimator_.
SKLEARN_ARRAY_ATTRS = (
    "coef_",
    "intercept_",
    "feature_importances_",
    "mean_",
    "n_samples_seen_",
    "scale_",
    "var_",
    "min_",
    "data_min_",
    "data_max_",
    "data_range_",
    # GP inner estimator arrays (GPC base_estimator_ / GPR direct)
    "X_train_",
    "L_",
    "pi_",
    "W_sr_",
    "y_train_",
    "classes_",
)
