"""XGBoost serialization.

Supports two formats:
- ``native-json``: stores the booster model as XGBoost's own JSON dict
  in the state.  This is the safe fallback.
- ``columnar-tensors`` (default): decomposes the tree ensemble into
  parallel numpy arrays stored in safetensors, with only small scalar
  config in JSON.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Field definitions for columnar extraction
# ---------------------------------------------------------------------------

# (json_key, array_key, dtype)
_NODE_FIELDS = [
    ("split_indices", "split_indices", np.int32),
    ("split_conditions", "split_conditions", np.float32),
    ("left_children", "left_children", np.int32),
    ("right_children", "right_children", np.int32),
    ("parents", "parents", np.int32),
    ("default_left", "default_left", np.uint8),
    ("base_weights", "base_weights", np.float32),
    ("sum_hessian", "sum_hessian", np.float32),
    ("loss_changes", "loss_changes", np.float32),
    ("split_type", "split_type", np.uint8),
]

_CAT_FIELDS = [
    "categories",
    "categories_segments",
    "categories_sizes",
    "categories_nodes",
]


# ---------------------------------------------------------------------------
# Columnar extraction: booster JSON -> arrays + metadata
# ---------------------------------------------------------------------------


def _extract_columnar(
    booster: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract columnar arrays and metadata from an XGBoost booster."""
    raw = json.loads(booster.save_raw(raw_format="json").decode())
    learner = raw["learner"]
    gbm = learner["gradient_booster"]
    model = gbm["model"]
    trees = model["trees"]

    # Accumulators: one list of arrays per field
    node_acc: dict[str, list[np.ndarray]] = {f[1]: [] for f in _NODE_FIELDS}
    offsets = [0]
    tree_params: list[dict[str, str]] = []

    # Categorical split data
    cat_acc: dict[str, list[int]] = {f: [] for f in _CAT_FIELDS}
    cat_offset = 0

    for tree in trees:
        num_nodes = int(tree["tree_param"]["num_nodes"])
        offsets.append(offsets[-1] + num_nodes)
        tree_params.append(tree["tree_param"])

        for json_key, arr_key, dtype in _NODE_FIELDS:
            node_acc[arr_key].append(np.array(tree[json_key], dtype=dtype))

        cats = tree.get("categories", [])
        cat_acc["categories"].extend(cats)
        cat_acc["categories_nodes"].extend(tree.get("categories_nodes", []))
        cat_acc["categories_segments"].extend(
            s + cat_offset for s in tree.get("categories_segments", [])
        )
        cat_acc["categories_sizes"].extend(tree.get("categories_sizes", []))
        cat_offset += len(cats)

    arrays: dict[str, np.ndarray] = {
        "offsets": np.array(offsets, dtype=np.int32),
    }
    for _, arr_key, _ in _NODE_FIELDS:
        arrays[arr_key] = np.concatenate(node_acc[arr_key])
    for cat_key in _CAT_FIELDS:
        vals = cat_acc[cat_key]
        arrays[cat_key] = (
            np.array(vals, dtype=np.int32) if vals else np.array([], dtype=np.int32)
        )

    meta: dict[str, Any] = {
        "learner_model_param": learner["learner_model_param"],
        "objective": learner["objective"],
        "attributes": learner.get("attributes", {}),
        "feature_names": learner.get("feature_names", []),
        "feature_types": learner.get("feature_types", []),
        "gbtree_model_param": model["gbtree_model_param"],
        "iteration_indptr": model["iteration_indptr"],
        "tree_info": model["tree_info"],
        "tree_params": tree_params,
        "gbm_name": gbm["name"],
        "version": raw["version"],
    }

    return arrays, meta


# ---------------------------------------------------------------------------
# Columnar reconstruction: arrays + metadata -> JSON -> booster
# ---------------------------------------------------------------------------


def _rebuild_model_json(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild XGBoost model JSON from columnar arrays and metadata."""
    offsets = arrays["offsets"]
    n_trees = len(offsets) - 1
    tree_params = meta["tree_params"]

    trees = []
    for t in range(n_trees):
        start, end = int(offsets[t]), int(offsets[t + 1])
        tree_dict: dict[str, Any] = {"id": t, "tree_param": tree_params[t]}
        for json_key, arr_key, _ in _NODE_FIELDS:
            tree_dict[json_key] = arrays[arr_key][start:end].tolist()
        for cat_key in _CAT_FIELDS:
            tree_dict[cat_key] = []
        trees.append(tree_dict)

    return {
        "learner": {
            "attributes": meta.get("attributes", {}),
            "feature_names": meta["feature_names"],
            "feature_types": meta["feature_types"],
            "gradient_booster": {
                "model": {
                    "cats": {
                        "enc": [],
                        "feature_segments": [],
                        "sorted_idx": [],
                    },
                    "gbtree_model_param": meta["gbtree_model_param"],
                    "iteration_indptr": meta["iteration_indptr"],
                    "tree_info": meta["tree_info"],
                    "trees": trees,
                },
                "name": meta["gbm_name"],
            },
            "learner_model_param": meta["learner_model_param"],
            "objective": meta["objective"],
        },
        "version": meta["version"],
    }


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def collect_state(
    estimator: BaseEstimator, prefix: str, format: str | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    booster = estimator.get_booster()  # type: ignore[attr-defined]

    if format == "native":
        state[f"{prefix}__format__"] = {
            "library": "xgboost",
            "format": "native-json",
            "schema_version": 1,
        }
        model_json = json.loads(booster.save_raw(raw_format="json").decode())
        state[f"{prefix}model_json"] = model_json
    else:
        state[f"{prefix}__format__"] = {
            "library": "xgboost",
            "format": "columnar-tensors",
            "schema_version": 1,
        }
        _, meta = _extract_columnar(booster)
        state[f"{prefix}tree"] = meta

    if _is_classifier(estimator):
        state[f"{prefix}n_classes"] = estimator.n_classes_  # type: ignore[attr-defined]

    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    import xgboost as xgb

    fmt = fitted_state.get(f"{prefix}__format__", {}).get("format", "native-json")

    if fmt != "columnar-tensors":
        model_json = fitted_state[f"{prefix}model_json"]
        model_bytes = bytearray(json.dumps(model_json).encode())
        booster = xgb.Booster()
        booster.load_model(model_bytes)
        estimator._Booster = booster  # type: ignore[attr-defined]

    if _is_classifier(estimator):
        estimator.n_classes_ = fitted_state[f"{prefix}n_classes"]  # type: ignore[attr-defined]


def extract_arrays(
    estimator: BaseEstimator, prefix: str, format: str | None = None
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    if format != "native":
        booster = estimator.get_booster()  # type: ignore[attr-defined]
        columnar, _ = _extract_columnar(booster)
        for k, v in columnar.items():
            arrays[f"{prefix}tree/{k}"] = v

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
    if fitted_state is None:
        return

    fmt = fitted_state.get(f"{prefix}__format__", {}).get("format", "native-json")
    if fmt != "columnar-tensors":
        return

    import xgboost as xgb

    tree_arrays: dict[str, np.ndarray] = {}
    tree_prefix = f"{prefix}tree/"
    for k, v in arrays.items():
        if k.startswith(tree_prefix):
            tree_arrays[k[len(tree_prefix) :]] = v

    meta = fitted_state[f"{prefix}tree"]
    model_json = _rebuild_model_json(tree_arrays, meta)
    model_bytes = bytearray(json.dumps(model_json).encode())
    booster = xgb.Booster()
    booster.load_model(model_bytes)
    estimator._Booster = booster  # type: ignore[attr-defined]
