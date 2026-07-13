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
        elif section == "skip":
            # feature_importances lines -- skip
            pass

    return {"header": header, "trees": trees, "params": params}


def _extract_columnar(
    booster: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract columnar arrays and metadata from a LightGBM booster."""
    parsed = _parse_model_string(booster.model_to_string())
    header = parsed["header"]
    trees = parsed["trees"]
    params = parsed["params"]

    # Per-tree arrays collected, then concatenated
    all_split_feature: list[np.ndarray] = []
    all_threshold: list[np.ndarray] = []
    all_decision_type: list[np.ndarray] = []
    all_left_child: list[np.ndarray] = []
    all_right_child: list[np.ndarray] = []
    all_split_gain: list[np.ndarray] = []
    all_internal_value: list[np.ndarray] = []
    all_internal_weight: list[np.ndarray] = []
    all_internal_count: list[np.ndarray] = []

    all_leaf_value: list[np.ndarray] = []
    all_leaf_weight: list[np.ndarray] = []
    all_leaf_count: list[np.ndarray] = []

    split_offsets = [0]
    leaf_offsets = [0]
    shrinkages: list[float] = []
    num_cats: list[int] = []
    is_linears: list[int] = []

    for tree in trees:
        num_leaves = int(tree["num_leaves"])
        num_splits = num_leaves - 1

        if num_splits > 0:
            sf = np.fromstring(tree["split_feature"], dtype=np.int32, sep=" ")
            th = np.fromstring(tree["threshold"], dtype=np.float64, sep=" ")
            dt = np.fromstring(tree["decision_type"], dtype=np.uint8, sep=" ")
            lc = np.fromstring(tree["left_child"], dtype=np.int32, sep=" ")
            rc = np.fromstring(tree["right_child"], dtype=np.int32, sep=" ")
            sg = np.fromstring(tree["split_gain"], dtype=np.float32, sep=" ")
            iv = np.fromstring(tree["internal_value"], dtype=np.float64, sep=" ")
            iw = np.fromstring(tree["internal_weight"], dtype=np.float64, sep=" ")
            ic = np.fromstring(tree["internal_count"], dtype=np.int32, sep=" ")
        else:
            # Single-leaf tree (no splits)
            sf = np.array([], dtype=np.int32)
            th = np.array([], dtype=np.float64)
            dt = np.array([], dtype=np.uint8)
            lc = np.array([], dtype=np.int32)
            rc = np.array([], dtype=np.int32)
            sg = np.array([], dtype=np.float32)
            iv = np.array([], dtype=np.float64)
            iw = np.array([], dtype=np.float64)
            ic = np.array([], dtype=np.int32)

        lv = np.fromstring(tree["leaf_value"], dtype=np.float64, sep=" ")
        lw = np.fromstring(tree["leaf_weight"], dtype=np.float64, sep=" ")
        lcount = np.fromstring(tree["leaf_count"], dtype=np.int32, sep=" ")

        all_split_feature.append(sf)
        all_threshold.append(th)
        all_decision_type.append(dt)
        all_left_child.append(lc)
        all_right_child.append(rc)
        all_split_gain.append(sg)
        all_internal_value.append(iv)
        all_internal_weight.append(iw)
        all_internal_count.append(ic)
        all_leaf_value.append(lv)
        all_leaf_weight.append(lw)
        all_leaf_count.append(lcount)

        split_offsets.append(split_offsets[-1] + num_splits)
        leaf_offsets.append(leaf_offsets[-1] + num_leaves)
        shrinkages.append(float(tree.get("shrinkage", "1")))
        num_cats.append(int(tree.get("num_cat", "0")))
        is_linears.append(int(tree.get("is_linear", "0")))

    arrays: dict[str, np.ndarray] = {}
    arrays["split_offsets"] = np.array(split_offsets, dtype=np.int32)
    arrays["leaf_offsets"] = np.array(leaf_offsets, dtype=np.int32)
    arrays["split_feature"] = (
        np.concatenate(all_split_feature)
        if all_split_feature
        else np.array([], dtype=np.int32)
    )
    arrays["threshold"] = (
        np.concatenate(all_threshold)
        if all_threshold
        else np.array([], dtype=np.float64)
    )
    arrays["decision_type"] = (
        np.concatenate(all_decision_type)
        if all_decision_type
        else np.array([], dtype=np.uint8)
    )
    arrays["left_child"] = (
        np.concatenate(all_left_child)
        if all_left_child
        else np.array([], dtype=np.int32)
    )
    arrays["right_child"] = (
        np.concatenate(all_right_child)
        if all_right_child
        else np.array([], dtype=np.int32)
    )
    arrays["split_gain"] = (
        np.concatenate(all_split_gain)
        if all_split_gain
        else np.array([], dtype=np.float32)
    )
    arrays["internal_value"] = (
        np.concatenate(all_internal_value)
        if all_internal_value
        else np.array([], dtype=np.float64)
    )
    arrays["internal_weight"] = (
        np.concatenate(all_internal_weight)
        if all_internal_weight
        else np.array([], dtype=np.float64)
    )
    arrays["internal_count"] = (
        np.concatenate(all_internal_count)
        if all_internal_count
        else np.array([], dtype=np.int32)
    )
    arrays["leaf_value"] = (
        np.concatenate(all_leaf_value)
        if all_leaf_value
        else np.array([], dtype=np.float64)
    )
    arrays["leaf_weight"] = (
        np.concatenate(all_leaf_weight)
        if all_leaf_weight
        else np.array([], dtype=np.float64)
    )
    arrays["leaf_count"] = (
        np.concatenate(all_leaf_count)
        if all_leaf_count
        else np.array([], dtype=np.int32)
    )
    arrays["shrinkage"] = np.array(shrinkages, dtype=np.float64)
    arrays["num_cat"] = np.array(num_cats, dtype=np.int32)
    arrays["is_linear"] = np.array(is_linears, dtype=np.int32)

    # Feature info for reconstruction
    feature_names = header.get("feature_names", "").split(" ")
    feature_infos_str = header.get("feature_infos", "")

    meta: dict[str, Any] = {
        "objective": header.get("objective", ""),
        "num_class": int(header.get("num_class", "1")),
        "num_tree_per_iteration": int(header.get("num_tree_per_iteration", "1")),
        "label_index": int(header.get("label_index", "0")),
        "max_feature_idx": int(header.get("max_feature_idx", "0")),
        "average_output": header.get("average_output", "False") == "True",
        "feature_names": feature_names,
        "feature_infos": feature_infos_str,
        "version": header.get("version", "v4"),
        "params": params,
    }

    return arrays, meta


# ---------------------------------------------------------------------------
# Columnar reconstruction: arrays + metadata -> model string -> booster
# ---------------------------------------------------------------------------


def _rebuild_model_string(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> str:
    """Rebuild a LightGBM model string from columnar arrays and metadata."""
    split_offsets = arrays["split_offsets"]
    leaf_offsets = arrays["leaf_offsets"]
    n_trees = len(split_offsets) - 1

    # Compute tree_sizes (approximate, LightGBM uses this for seeking)
    tree_sizes: list[int] = []
    tree_sections: list[str] = []

    for t in range(n_trees):
        s_start, s_end = int(split_offsets[t]), int(split_offsets[t + 1])
        l_start, l_end = int(leaf_offsets[t]), int(leaf_offsets[t + 1])
        n_splits = s_end - s_start
        n_leaves = l_end - l_start

        lines = [f"Tree={t}"]
        lines.append(f"num_leaves={n_leaves}")
        lines.append(f"num_cat={int(arrays['num_cat'][t])}")

        if n_splits > 0:
            lines.append(
                "split_feature="
                + " ".join(str(x) for x in arrays["split_feature"][s_start:s_end])
            )
            lines.append(
                "split_gain="
                + " ".join(f"{x}" for x in arrays["split_gain"][s_start:s_end])
            )
            lines.append(
                "threshold="
                + " ".join(f"{x:.17g}" for x in arrays["threshold"][s_start:s_end])
            )
            lines.append(
                "decision_type="
                + " ".join(str(x) for x in arrays["decision_type"][s_start:s_end])
            )
            lines.append(
                "left_child="
                + " ".join(str(x) for x in arrays["left_child"][s_start:s_end])
            )
            lines.append(
                "right_child="
                + " ".join(str(x) for x in arrays["right_child"][s_start:s_end])
            )
        else:
            lines.append("split_feature=")
            lines.append("split_gain=")
            lines.append("threshold=")
            lines.append("decision_type=")
            lines.append("left_child=")
            lines.append("right_child=")

        lines.append(
            "leaf_value="
            + " ".join(f"{x:.17g}" for x in arrays["leaf_value"][l_start:l_end])
        )
        lines.append(
            "leaf_weight="
            + " ".join(str(x) for x in arrays["leaf_weight"][l_start:l_end])
        )
        lines.append(
            "leaf_count="
            + " ".join(str(x) for x in arrays["leaf_count"][l_start:l_end])
        )

        if n_splits > 0:
            lines.append(
                "internal_value="
                + " ".join(f"{x}" for x in arrays["internal_value"][s_start:s_end])
            )
            lines.append(
                "internal_weight="
                + " ".join(str(x) for x in arrays["internal_weight"][s_start:s_end])
            )
            lines.append(
                "internal_count="
                + " ".join(str(x) for x in arrays["internal_count"][s_start:s_end])
            )
        else:
            lines.append("internal_value=")
            lines.append("internal_weight=")
            lines.append("internal_count=")

        lines.append(f"is_linear={int(arrays['is_linear'][t])}")
        lines.append(f"shrinkage={arrays['shrinkage'][t]}")

        section = "\n".join(lines)
        tree_sections.append(section)
        tree_sizes.append(len(section) + 2)  # +2 for surrounding newlines

    # Header
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

    # Parameters section
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

    if fmt == "columnar-tensors":
        # Reconstruction happens in restore_arrays where we have access
        # to the safetensors arrays. Just mark that we need it.
        pass
    else:
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

    # Collect the tree arrays from the safetensors namespace
    tree_arrays: dict[str, np.ndarray] = {}
    tree_prefix = f"{prefix}tree/"
    for k, v in arrays.items():
        if k.startswith(tree_prefix):
            tree_arrays[k[len(tree_prefix) :]] = v

    meta = fitted_state[f"{prefix}tree"]
    model_str = _rebuild_model_string(tree_arrays, meta)
    estimator._Booster = lgb.Booster(model_str=model_str)  # type: ignore[attr-defined]
