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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import Kernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Supported inner sklearn scalers. Extend when new scaler types are wrapped.
BaseScaler = StandardScaler | MinMaxScaler | RobustScaler


def get_sklearn_public_path(cls: type[BaseEstimator | Kernel]) -> str:
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


# ---------------------------------------------------------------------------
# Kernel serialization
# ---------------------------------------------------------------------------


def _serialize_kernel(kernel: Kernel) -> dict[str, Any]:
    """Recursively convert a kernel tree to a JSON-safe dict."""
    from sklearn.gaussian_process.kernels import KernelOperator

    data: dict[str, Any] = {"type": get_sklearn_public_path(type(kernel))}
    if isinstance(kernel, KernelOperator):
        data["k1"] = _serialize_kernel(kernel.k1)
        data["k2"] = _serialize_kernel(kernel.k2)
    else:
        params = kernel.get_params(deep=False)
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
            elif isinstance(v, tuple):
                data[k] = list(v)
            else:
                data[k] = v
    return data


def _deserialize_kernel(data: dict[str, Any]) -> Kernel:
    """Reconstruct a kernel tree from a serialised dict."""
    from sklearn.gaussian_process.kernels import KernelOperator

    data = dict(data)
    type_path: str = data.pop("type")
    module_path, _, class_name = type_path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)

    if issubclass(cls, KernelOperator):
        k1 = _deserialize_kernel(data["k1"])
        k2 = _deserialize_kernel(data["k2"])
        return cls(k1=k1, k2=k2)

    # Leaf kernel — convert list params back to correct types
    try:
        defaults = cls().get_params(deep=False)
    except TypeError:
        defaults = {}

    init_params: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, list):
            if isinstance(defaults.get(k), tuple):
                init_params[k] = tuple(v)
            else:
                init_params[k] = np.asarray(v)
        else:
            init_params[k] = v
    return cls(**init_params)


# ---------------------------------------------------------------------------
# Param serialization (get/set)
# ---------------------------------------------------------------------------


def get_model_params(model: BaseEstimator) -> dict[str, Any]:
    """Recursively extract hyperparameters from a fitted sklearn estimator."""

    def _get_params(model):
        params = model.get_params(deep=False)
        for k in list(params):
            if hasattr(model, f"named_{k}"):
                params[k] = getattr(model, f"named_{k}")
        return params

    def _serialize(obj):
        if isinstance(obj, list):
            return [_serialize(o) for o in obj]
        elif isinstance(obj, tuple):
            return tuple(_serialize(o) for o in obj)
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        elif isinstance(obj, Kernel):
            return _serialize_kernel(obj)
        elif isinstance(obj, BaseEstimator):
            if isinstance(obj, GaussianProcessClassifier):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    return _serialize(
                        _get_params(obj) | {"type": get_sklearn_public_path(type(obj))}
                    )
            return _serialize(
                _get_params(obj) | {"type": get_sklearn_public_path(type(obj))}
            )
        else:
            return obj

    return _serialize(_get_params(model))  # type: ignore


def set_model_params(model: BaseEstimator, params: dict[str, Any]) -> BaseEstimator:
    """Recursively set hyperparameters on an sklearn estimator."""
    params.pop("type", None)
    keys = list(params.keys())
    for key in keys:
        if hasattr(model, f"named_{key}"):
            attr = getattr(model, f"named_{key}")
        else:
            attr = getattr(model, key)
        if isinstance(attr, BaseEstimator):
            set_model_params(attr, params.pop(key))
        elif isinstance(attr, dict):
            values = params[key]
            for k, v in attr.items():
                if isinstance(v, BaseEstimator):
                    if k in values:
                        set_model_params(v, values.pop(k))
            if not values:
                params.pop(key)
    return model.set_params(**params)  # type: ignore


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
# Array extraction / restoration
# ---------------------------------------------------------------------------

_SKLEARN_ARRAY_ATTRS = (
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

# Non-array fitted attributes on MLPRegressor that must be persisted in JSON
# (beyond coefs_/intercepts_ which go to safetensors).
_MLP_FITTED_STATE_ATTRS = (
    "n_layers_",
    "n_outputs_",
    "out_activation_",
    "n_iter_",
    "t_",
    "loss_",
    "best_loss_",
    "n_iter_no_change",
)


def _collect_fitted_state(estimator: BaseEstimator, prefix: str = "") -> dict[str, Any]:
    """Collect non-array fitted attributes from estimator instances.

    Walks Pipeline steps recursively. Handles MLPRegressor, GPR, GPC,
    and HistGradientBoosting estimators.
    """
    state: dict[str, Any] = {}

    if isinstance(estimator, Pipeline):
        for step_name, step in estimator.named_steps.items():
            step_prefix = f"{prefix}{step_name}/" if prefix else f"{step_name}/"
            state.update(_collect_fitted_state(step, prefix=step_prefix))
        return state

    if isinstance(estimator, MLPRegressor):
        for attr in _MLP_FITTED_STATE_ATTRS:
            if hasattr(estimator, attr):
                state[f"{prefix}{attr}"] = getattr(estimator, attr)

    elif isinstance(estimator, GaussianProcessRegressor | GaussianProcessClassifier):
        if hasattr(estimator, "log_marginal_likelihood_value_"):
            state[f"{prefix}log_marginal_likelihood_value_"] = (
                estimator.log_marginal_likelihood_value_
            )
        if isinstance(estimator, GaussianProcessRegressor):
            if hasattr(estimator, "kernel_"):
                state[f"{prefix}kernel_"] = _serialize_kernel(
                    estimator.kernel_  # type: ignore[arg-type]
                )
        else:
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

    elif isinstance(
        estimator,
        HistGradientBoostingClassifier | HistGradientBoostingRegressor,
    ):
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
                [name, sample_nodes.dtype[name].str]
                for name in sample_nodes.dtype.names
            ]

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

    if isinstance(estimator, MLPRegressor):
        for attr in _MLP_FITTED_STATE_ATTRS:
            key = f"{prefix}{attr}"
            if key in fitted_state:
                setattr(estimator, attr, fitted_state[key])

    elif isinstance(estimator, GaussianProcessRegressor | GaussianProcessClassifier):
        lml_key = f"{prefix}log_marginal_likelihood_value_"
        if lml_key in fitted_state:
            estimator.log_marginal_likelihood_value_ = fitted_state[lml_key]
        if isinstance(estimator, GaussianProcessRegressor):
            kernel_key = f"{prefix}kernel_"
            if kernel_key in fitted_state:
                estimator.kernel_ = _deserialize_kernel(fitted_state[kernel_key])
        else:
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


def _arrays_from_estimator(
    estimator: BaseEstimator, prefix: str = ""
) -> dict[str, np.ndarray]:
    """Recursively extract numpy arrays from a fitted sklearn estimator."""
    arrays: dict[str, np.ndarray] = {}

    def _key(name: str) -> str:
        return f"{prefix}{name}"

    if isinstance(estimator, Pipeline):
        for step_name, step in estimator.named_steps.items():
            step_prefix = f"{prefix}{step_name}/" if prefix else f"{step_name}/"
            arrays.update(_arrays_from_estimator(step, prefix=step_prefix))
        return arrays

    if isinstance(estimator, MLPRegressor):
        if hasattr(estimator, "coefs_"):
            for i, w in enumerate(estimator.coefs_):
                arrays[f"{prefix}coefs_{i}"] = np.asarray(w)
        if hasattr(estimator, "intercepts_"):
            for i, b in enumerate(estimator.intercepts_):
                arrays[f"{prefix}intercepts_{i}"] = np.asarray(b)
        return arrays

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

    if isinstance(estimator, GaussianProcessClassifier):
        if hasattr(estimator, "classes_"):
            arrays[_key("classes_")] = np.asarray(estimator.classes_)
        if hasattr(estimator, "base_estimator_"):
            be_prefix = f"{prefix}base_estimator_/"
            arrays.update(
                _arrays_from_estimator(estimator.base_estimator_, prefix=be_prefix)
            )
        return arrays

    if isinstance(
        estimator,
        HistGradientBoostingClassifier | HistGradientBoostingRegressor,
    ):
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
                result[f"{pprefix}raw_left_cat_bitsets"] = pstate[
                    "raw_left_cat_bitsets"
                ]
        return result

    supported = any(hasattr(estimator, attr) for attr in _SKLEARN_ARRAY_ATTRS)
    if not supported:
        raise NotImplementedError(
            f"_arrays_from_estimator does not support estimator type "
            f"'{type(estimator).__name__}'. Only linear models, scalers, "
            f"MLPRegressor, GaussianProcessRegressor, "
            f"GaussianProcessClassifier, HistGradientBoosting*, "
            f"and Pipelines of those are supported."
        )

    for attr in _SKLEARN_ARRAY_ATTRS:
        if hasattr(estimator, attr):
            val = getattr(estimator, attr)
            arrays[f"{prefix}{attr}"] = np.asarray(val)

    return arrays


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

    if isinstance(estimator, MLPRegressor):
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
            estimator.coefs_ = coefs
        if intercepts:
            estimator.intercepts_ = intercepts
        return

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

    if isinstance(estimator, GaussianProcessClassifier):
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
            _restore_estimator_arrays(
                estimator.base_estimator_, arrays, prefix=be_prefix
            )
        return

    if isinstance(
        estimator,
        HistGradientBoostingClassifier | HistGradientBoostingRegressor,
    ):
        from sklearn.ensemble._hist_gradient_boosting.binning import (
            _BinMapper,  # type: ignore[attr-defined]
        )
        from sklearn.ensemble._hist_gradient_boosting.predictor import (
            TreePredictor,  # type: ignore[attr-defined]
        )
        from sklearn.preprocessing import LabelEncoder

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
        return

    for attr in _SKLEARN_ARRAY_ATTRS:
        key = f"{prefix}{attr}"
        if key in arrays:
            setattr(estimator, attr, arrays[key])


def _rebuild_estimator_from_params(params: dict[str, Any]) -> BaseEstimator:
    """Instantiate a sklearn estimator tree from a params dict."""
    params = dict(params)
    type_path: str = params.pop("type")
    module_path, _, class_name = type_path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)

    is_pipeline = issubclass(cls, Pipeline)

    init_params: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, dict) and "type" in v and "kernels." in v["type"]:
            init_params[k] = _deserialize_kernel(v)
        elif isinstance(v, dict) and "type" in v:
            init_params[k] = _rebuild_estimator_from_params(v)
        elif is_pipeline and k == "steps" and isinstance(v, dict):
            init_params[k] = [
                (name, _rebuild_estimator_from_params(step_params))
                for name, step_params in v.items()
            ]
        elif isinstance(v, list):
            rebuilt = []
            for item in v:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[1], dict)
                ):
                    rebuilt.append((item[0], _rebuild_estimator_from_params(item[1])))
                else:
                    rebuilt.append(item)
            init_params[k] = rebuilt
        else:
            init_params[k] = v

    init_params = _fix_json_param_types(cls, init_params)
    return cls(**init_params)


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
            columns=self.targets,
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
        set_model_params(self.model, params)
        return self

    def get_params(self) -> dict[str, Any]:
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
