"""HistGradientBoosting estimator serialization."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


def handles(estimator: BaseEstimator) -> bool:
    """Return True if this module handles the given estimator type."""
    return isinstance(
        estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
    )


def collect_state(estimator: BaseEstimator, prefix: str) -> dict[str, Any]:
    """Collect non-array fitted state from an HGB estimator."""
    assert isinstance(
        estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
    )
    state: dict[str, Any] = {}
    for attr in (
        "n_trees_per_iteration_",
        "do_early_stopping_",
        "n_features_in_",
    ):
        if hasattr(estimator, attr):
            state[f"{prefix}{attr}"] = getattr(estimator, attr)
    if hasattr(estimator, "_random_seed"):
        state[f"{prefix}_random_seed"] = int(
            estimator._random_seed  # type: ignore[attr-defined]
        )
    bm = estimator._bin_mapper  # type: ignore[attr-defined]
    state[f"{prefix}_bin_mapper/n_bins"] = bm.n_bins
    state[f"{prefix}_bin_mapper/subsample"] = bm.subsample
    state[f"{prefix}_bin_mapper/random_state"] = int(bm.random_state)  # type: ignore[arg-type]
    state[f"{prefix}_bin_mapper/n_threads"] = bm.n_threads
    state[f"{prefix}_bin_mapper/missing_values_bin_idx_"] = int(
        bm.missing_values_bin_idx_  # type: ignore[arg-type]
    )
    state[f"{prefix}_bin_mapper/n_features_"] = len(bm.bin_thresholds_)
    state[f"{prefix}_predictors_shape"] = [
        len(pl)
        for pl in estimator._predictors  # type: ignore[attr-defined]
    ]
    if (
        estimator._predictors and estimator._predictors[0]  # type: ignore[attr-defined]
    ):
        sample_nodes = estimator._predictors[0][0].__getstate__()["nodes"]  # type: ignore[attr-defined]
        state[f"{prefix}_predictors_node_dtype"] = [
            [name, sample_nodes.dtype[name].str] for name in sample_nodes.dtype.names
        ]
    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    """No-op: HGB state restoration is handled in restore_arrays."""
    # HGB state restoration is handled entirely in restore_arrays.
    return


def extract_arrays(estimator: BaseEstimator, prefix: str) -> dict[str, np.ndarray]:
    """Extract arrays from an HGB estimator."""
    assert isinstance(
        estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
    )
    result: dict[str, np.ndarray] = {}
    for attr in (
        "classes_",
        "train_score_",
        "validation_score_",
        "_baseline_prediction",
    ):
        val = getattr(estimator, attr, None)
        if val is not None:
            result[f"{prefix}{attr}"] = np.asarray(val)
    bm = estimator._bin_mapper  # type: ignore[attr-defined]
    for k, bt in enumerate(bm.bin_thresholds_):
        result[f"{prefix}_bin_mapper/bin_thresholds_{k}"] = np.asarray(bt)
    result[f"{prefix}_bin_mapper/is_categorical_"] = np.asarray(bm.is_categorical_)
    result[f"{prefix}_bin_mapper/n_bins_non_missing_"] = np.asarray(
        bm.n_bins_non_missing_
    )
    le = getattr(estimator, "_label_encoder", None)
    if le is not None:
        result[f"{prefix}_label_encoder/classes_"] = np.asarray(le.classes_)
    for i, pred_list in enumerate(
        estimator._predictors  # type: ignore[attr-defined]
    ):
        for j, pred in enumerate(pred_list):
            pstate = pred.__getstate__()
            pprefix = f"{prefix}_predictors/{i}/{j}/"
            nodes = pstate["nodes"]
            assert nodes.dtype.names is not None
            for field in nodes.dtype.names:
                result[f"{pprefix}nodes_{field}"] = nodes[field].copy()
            result[f"{pprefix}binned_left_cat_bitsets"] = pstate[
                "binned_left_cat_bitsets"
            ]
            result[f"{pprefix}raw_left_cat_bitsets"] = pstate["raw_left_cat_bitsets"]
    return result


def restore_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
    fitted_state: dict[str, Any] | None = None,
) -> None:
    """Restore arrays onto an HGB estimator."""
    from sklearn.ensemble._hist_gradient_boosting.binning import (
        _BinMapper,  # type: ignore[attr-defined]
    )
    from sklearn.ensemble._hist_gradient_boosting.predictor import (
        TreePredictor,  # type: ignore[attr-defined]
    )
    from sklearn.preprocessing import LabelEncoder

    assert isinstance(
        estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
    )
    assert fitted_state is not None, "HGB restoration requires fitted_state"

    for attr in (
        "n_trees_per_iteration_",
        "do_early_stopping_",
        "n_features_in_",
    ):
        key = f"{prefix}{attr}"
        if key in fitted_state:
            setattr(estimator, attr, fitted_state[key])
    if f"{prefix}_random_seed" in fitted_state:
        estimator._random_seed = np.uint64(  # type: ignore[attr-defined]
            fitted_state[f"{prefix}_random_seed"]
        )

    for attr in (
        "classes_",
        "train_score_",
        "validation_score_",
        "_baseline_prediction",
    ):
        key = f"{prefix}{attr}"
        if key in arrays:
            setattr(estimator, attr, arrays[key])
    if isinstance(estimator, HistGradientBoostingClassifier) and hasattr(
        estimator, "classes_"
    ):
        estimator.n_classes_ = len(estimator.classes_)  # type: ignore[attr-defined]

    bm = _BinMapper(
        n_bins=fitted_state[f"{prefix}_bin_mapper/n_bins"],
        subsample=fitted_state[f"{prefix}_bin_mapper/subsample"],
        random_state=fitted_state[f"{prefix}_bin_mapper/random_state"],
        n_threads=fitted_state[f"{prefix}_bin_mapper/n_threads"],
    )
    bm.missing_values_bin_idx_ = fitted_state[
        f"{prefix}_bin_mapper/missing_values_bin_idx_"
    ]
    bm.is_categorical = None  # type: ignore[attr-defined]
    bm.known_categories = None  # type: ignore[attr-defined]
    n_features: int = fitted_state[f"{prefix}_bin_mapper/n_features_"]
    bm.bin_thresholds_ = [
        arrays[f"{prefix}_bin_mapper/bin_thresholds_{k}"] for k in range(n_features)
    ]
    bm.is_categorical_ = arrays[f"{prefix}_bin_mapper/is_categorical_"]
    bm.n_bins_non_missing_ = arrays[f"{prefix}_bin_mapper/n_bins_non_missing_"]
    estimator._bin_mapper = bm  # type: ignore[attr-defined]

    le_key = f"{prefix}_label_encoder/classes_"
    if le_key in arrays:
        le = LabelEncoder()
        le.classes_ = arrays[le_key]
        estimator._label_encoder = le  # type: ignore[attr-defined]
    else:
        estimator._label_encoder = None  # type: ignore[attr-defined]

    dtype_spec = fitted_state.get(f"{prefix}_predictors_node_dtype")
    if dtype_spec is not None:
        node_dtype: np.dtype = np.dtype(
            [(name, dtype_str) for name, dtype_str in dtype_spec]
        )
    else:
        node_dtype = np.dtype(
            [
                ("value", "<f8"),
                ("count", "<u4"),
                ("feature_idx", "<i8"),
                ("num_threshold", "<f8"),
                ("missing_go_to_left", "u1"),
                ("left", "<u4"),
                ("right", "<u4"),
                ("gain", "<f8"),
                ("depth", "<u4"),
                ("is_leaf", "u1"),
                ("bin_threshold", "u1"),
                ("is_categorical", "u1"),
                ("bitset_idx", "<u4"),
            ]
        )
    predictors_shape: list[int] = fitted_state[f"{prefix}_predictors_shape"]
    assert node_dtype.names is not None
    predictors = []
    for i, n_trees in enumerate(predictors_shape):
        pred_list = []
        for j in range(n_trees):
            pprefix = f"{prefix}_predictors/{i}/{j}/"
            first_field = node_dtype.names[0]
            n_nodes = arrays[f"{pprefix}nodes_{first_field}"].shape[0]
            nodes = np.empty(n_nodes, dtype=node_dtype)
            for field in node_dtype.names:
                nodes[field] = arrays[f"{pprefix}nodes_{field}"]
            pred = TreePredictor(
                nodes,
                arrays[f"{pprefix}binned_left_cat_bitsets"],
                arrays[f"{pprefix}raw_left_cat_bitsets"],
            )
            pred_list.append(pred)
        predictors.append(pred_list)
    estimator._predictors = predictors  # type: ignore[attr-defined]

    estimator._loss = estimator._get_loss(sample_weight=None)  # type: ignore[attr-defined]
    estimator._preprocessor = None  # type: ignore[attr-defined]
    estimator._scorer = None  # type: ignore[attr-defined]
    estimator._use_validation_data = False  # type: ignore[attr-defined]
    if isinstance(estimator, HistGradientBoostingClassifier):
        estimator._is_categorical_remapped = None  # type: ignore[attr-defined]
        estimator.is_categorical_ = None  # type: ignore[attr-defined]
