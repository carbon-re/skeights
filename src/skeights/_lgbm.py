"""LightGBM serialization.

Supports two formats:
- ``native-text``: stores the booster model string (LightGBM's own text
  format) in the JSON state.  This is the safe fallback.
- ``columnar-tensors`` (default): decomposes the tree ensemble into
  parallel numpy arrays stored in safetensors, with only small scalar
  config in JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Field definitions for columnar extraction
# ---------------------------------------------------------------------------

# (field_name_in_model_string, array_key, dtype)
_SPLIT_FIELDS = [
    ("split_feature", "split_feature", np.int32),
    ("threshold", "threshold", np.float64),
    ("decision_type", "decision_type", np.uint8),
    ("left_child", "left_child", np.int32),
    ("right_child", "right_child", np.int32),
    ("split_gain", "split_gain", np.float32),
    ("internal_value", "internal_value", np.float64),
    ("internal_weight", "internal_weight", np.float64),
    ("internal_count", "internal_count", np.int32),
]

_LEAF_FIELDS = [
    ("leaf_value", "leaf_value", np.float64),
    ("leaf_weight", "leaf_weight", np.float64),
    ("leaf_count", "leaf_count", np.int32),
]

# Fields that need full-precision formatting in the model string
_FULL_PRECISION_FIELDS = {"threshold", "leaf_value"}


# ---------------------------------------------------------------------------
# Columnar extraction: booster -> arrays + metadata
# ---------------------------------------------------------------------------


def _parse_model_string(model_str: str) -> dict[str, Any]:
    """Parse a LightGBM model string into header, trees, and parameters."""
    lines = model_str.split("\n")
    header: dict[str, str] = {}
    trees: list[dict[str, str]] = []
    params: dict[str, str] = {}
    current_tree: dict[str, str] | None = None
    section = "header"

    for line in lines:
        if line.startswith("Tree="):
            if current_tree is not None:
                trees.append(current_tree)
            current_tree = {}
            section = "tree"
            continue
        if line == "end of trees":
            if current_tree is not None:
                trees.append(current_tree)
                current_tree = None
            section = "post"
            continue
        if line == "parameters:":
            section = "params"
            continue
        if line == "end of parameters":
            section = "post"
            continue
        if line == "" or line.startswith("feature_importances:"):
            if line.startswith("feature_importances:"):
                section = "skip"
            continue

        if section == "header" and "=" in line:
            k, v = line.split("=", 1)
            header[k] = v
        elif section == "tree" and current_tree is not None and "=" in line:
            k, v = line.split("=", 1)
            current_tree[k] = v
        elif section == "params" and line.startswith("[") and line.endswith("]"):
            inner = line[1:-1]
            if ": " in inner:
                k, v = inner.split(": ", 1)
                params[k] = v

    return {"header": header, "trees": trees, "params": params}


def _extract_columnar(
    booster: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract columnar arrays and metadata from a LightGBM booster."""
    parsed = _parse_model_string(booster.model_to_string())
    header = parsed["header"]
    trees = parsed["trees"]
    params = parsed["params"]

    # Accumulators: one list of arrays per field
    split_acc: dict[str, list[np.ndarray]] = {f[1]: [] for f in _SPLIT_FIELDS}
    leaf_acc: dict[str, list[np.ndarray]] = {f[1]: [] for f in _LEAF_FIELDS}

    split_offsets = [0]
    leaf_offsets = [0]
    shrinkages: list[float] = []
    num_cats: list[int] = []
    is_linears: list[int] = []

    for tree in trees:
        num_leaves = int(tree["num_leaves"])
        num_splits = num_leaves - 1

        for text_key, arr_key, dtype in _SPLIT_FIELDS:
            if num_splits > 0:
                split_acc[arr_key].append(
                    np.fromstring(tree[text_key], dtype=dtype, sep=" ")
                )
            else:
                split_acc[arr_key].append(np.array([], dtype=dtype))

        for text_key, arr_key, dtype in _LEAF_FIELDS:
            leaf_acc[arr_key].append(
                np.fromstring(tree[text_key], dtype=dtype, sep=" ")
            )

        split_offsets.append(split_offsets[-1] + num_splits)
        leaf_offsets.append(leaf_offsets[-1] + num_leaves)
        shrinkages.append(float(tree.get("shrinkage", "1")))
        num_cats.append(int(tree.get("num_cat", "0")))
        is_linears.append(int(tree.get("is_linear", "0")))

    def _concat(arrs: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(arrs) if arrs and arrs[0].size > 0 else arrs[0]

    arrays: dict[str, np.ndarray] = {
        "split_offsets": np.array(split_offsets, dtype=np.int32),
        "leaf_offsets": np.array(leaf_offsets, dtype=np.int32),
    }
    for _, arr_key, _ in _SPLIT_FIELDS:
        arrays[arr_key] = _concat(split_acc[arr_key])
    for _, arr_key, _ in _LEAF_FIELDS:
        arrays[arr_key] = _concat(leaf_acc[arr_key])
    arrays["shrinkage"] = np.array(shrinkages, dtype=np.float64)
    arrays["num_cat"] = np.array(num_cats, dtype=np.int32)
    arrays["is_linear"] = np.array(is_linears, dtype=np.int32)

    meta: dict[str, Any] = {
        "objective": header.get("objective", ""),
        "num_class": int(header.get("num_class", "1")),
        "num_tree_per_iteration": int(header.get("num_tree_per_iteration", "1")),
        "label_index": int(header.get("label_index", "0")),
        "max_feature_idx": int(header.get("max_feature_idx", "0")),
        "average_output": header.get("average_output", "False") == "True",
        "feature_names": header.get("feature_names", "").split(" "),
        "feature_infos": header.get("feature_infos", ""),
        "version": header.get("version", "v4"),
        "params": params,
    }

    return arrays, meta


# ---------------------------------------------------------------------------
# Columnar reconstruction: arrays + metadata -> model string -> booster
# ---------------------------------------------------------------------------


def _format_array(arr: np.ndarray, key: str) -> str:
    """Format an array as a space-separated string."""
    if key in _FULL_PRECISION_FIELDS:
        return " ".join(f"{x:.17g}" for x in arr)
    return " ".join(str(x) for x in arr)


def _rebuild_model_string(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> str:
    """Rebuild a LightGBM model string from columnar arrays and metadata."""
    split_offsets = arrays["split_offsets"]
    leaf_offsets = arrays["leaf_offsets"]
    n_trees = len(split_offsets) - 1

    tree_sizes: list[int] = []
    tree_sections: list[str] = []

    split_field_names = [
        ("split_feature", "split_feature"),
        ("split_gain", "split_gain"),
        ("threshold", "threshold"),
        ("decision_type", "decision_type"),
        ("left_child", "left_child"),
        ("right_child", "right_child"),
    ]
    internal_field_names = [
        ("internal_value", "internal_value"),
        ("internal_weight", "internal_weight"),
        ("internal_count", "internal_count"),
    ]
    leaf_field_names = [
        ("leaf_value", "leaf_value"),
        ("leaf_weight", "leaf_weight"),
        ("leaf_count", "leaf_count"),
    ]

    for t in range(n_trees):
        s_start, s_end = int(split_offsets[t]), int(split_offsets[t + 1])
        l_start, l_end = int(leaf_offsets[t]), int(leaf_offsets[t + 1])
        n_splits = s_end - s_start
        n_leaves = l_end - l_start

        lines = [
            f"Tree={t}",
            f"num_leaves={n_leaves}",
            f"num_cat={int(arrays['num_cat'][t])}",
        ]

        for text_key, arr_key in split_field_names:
            if n_splits > 0:
                lines.append(
                    f"{text_key}="
                    + _format_array(arrays[arr_key][s_start:s_end], arr_key)
                )
            else:
                lines.append(f"{text_key}=")

        for text_key, arr_key in leaf_field_names:
            lines.append(
                f"{text_key}=" + _format_array(arrays[arr_key][l_start:l_end], arr_key)
            )

        for text_key, arr_key in internal_field_names:
            if n_splits > 0:
                lines.append(
                    f"{text_key}="
                    + _format_array(arrays[arr_key][s_start:s_end], arr_key)
                )
            else:
                lines.append(f"{text_key}=")

        lines.append(f"is_linear={int(arrays['is_linear'][t])}")
        lines.append(f"shrinkage={arrays['shrinkage'][t]}")

        section = "\n".join(lines)
        tree_sections.append(section)
        tree_sizes.append(len(section) + 2)

    feature_names = meta["feature_names"]
    header_lines = [
        "tree",
        f"version={meta['version']}",
        f"num_class={meta['num_class']}",
        f"num_tree_per_iteration={meta['num_tree_per_iteration']}",
        f"label_index={meta['label_index']}",
        f"max_feature_idx={meta['max_feature_idx']}",
        f"objective={meta['objective']}",
        f"feature_names={' '.join(feature_names)}",
        f"feature_infos={meta['feature_infos']}",
        f"tree_sizes={' '.join(str(s) for s in tree_sizes)}",
    ]

    params = meta.get("params", {})
    param_lines = ["parameters:"]
    for k, v in params.items():
        param_lines.append(f"[{k}: {v}]")
    param_lines.append("")
    param_lines.append("end of parameters")

    parts = [
        "\n".join(header_lines),
        "",
        "\n\n".join(tree_sections),
        "",
        "end of trees",
        "",
        "feature_importances:",
        "",
        "\n".join(param_lines),
        "",
        "pandas_categorical:null",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def collect_state(
    estimator: BaseEstimator, prefix: str, format: str | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {}

    if format == "native":
        state[f"{prefix}__format__"] = {
            "library": "lightgbm",
            "format": "native-text",
            "schema_version": 1,
        }
        state[f"{prefix}model_str"] = estimator._Booster.model_to_string()  # type: ignore[attr-defined]
    else:
        state[f"{prefix}__format__"] = {
            "library": "lightgbm",
            "format": "columnar-tensors",
            "schema_version": 1,
        }
        _, meta = _extract_columnar(estimator._Booster)  # type: ignore[attr-defined]
        state[f"{prefix}tree"] = meta

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

    fmt = fitted_state.get(f"{prefix}__format__", {}).get("format", "native-text")

    if fmt != "columnar-tensors":
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


def extract_arrays(
    estimator: BaseEstimator, prefix: str, format: str | None = None
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    if format != "native":
        columnar, _ = _extract_columnar(estimator._Booster)  # type: ignore[attr-defined]
        for k, v in columnar.items():
            arrays[f"{prefix}tree/{k}"] = v

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
    if fitted_state is None:
        return

    fmt = fitted_state.get(f"{prefix}__format__", {}).get("format", "native-text")
    if fmt != "columnar-tensors":
        return

    import lightgbm as lgb

    tree_arrays: dict[str, np.ndarray] = {}
    tree_prefix = f"{prefix}tree/"
    for k, v in arrays.items():
        if k.startswith(tree_prefix):
            tree_arrays[k[len(tree_prefix) :]] = v

    meta = fitted_state[f"{prefix}tree"]
    model_str = _rebuild_model_string(tree_arrays, meta)
    estimator._Booster = lgb.Booster(model_str=model_str)  # type: ignore[attr-defined]
