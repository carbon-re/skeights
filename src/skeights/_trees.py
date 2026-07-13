"""Tree-based estimator serialization (DecisionTree, RandomForest, GradientBoosting)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skeights._utils import get_sklearn_public_path

# Tree-based ensemble types.
_TreeEstimator = DecisionTreeRegressor | DecisionTreeClassifier
_RandomForest = RandomForestRegressor | RandomForestClassifier
_GradientBoosting = GradientBoostingRegressor | GradientBoostingClassifier
_TreeEnsemble = _RandomForest | _GradientBoosting


# ---------------------------------------------------------------------------
# Low-level Tree helpers
# ---------------------------------------------------------------------------


def _arrays_from_tree(tree_obj: object, prefix: str) -> dict[str, np.ndarray]:
    """Extract arrays from a sklearn Tree object (tree_ attribute)."""
    state = tree_obj.__getstate__()  # type: ignore[union-attr]
    arrays: dict[str, np.ndarray] = {}
    nodes = state["nodes"]  # type: ignore[index]
    assert nodes.dtype.names is not None
    for field in nodes.dtype.names:
        arrays[f"{prefix}nodes_{field}"] = nodes[field].copy()
    arrays[f"{prefix}values"] = state["values"]  # type: ignore[index]
    return arrays


def _state_from_tree(tree_obj: object, prefix: str) -> dict[str, Any]:
    """Extract scalar state from a sklearn Tree object."""
    state = tree_obj.__getstate__()  # type: ignore[union-attr]
    result: dict[str, Any] = {}
    result[f"{prefix}max_depth"] = state["max_depth"]  # type: ignore[index]
    result[f"{prefix}node_count"] = state["node_count"]  # type: ignore[index]
    # Tree constructor needs n_features, n_classes, n_outputs — always read
    # from the object (available as attrs even when __getstate__ omits them).
    result[f"{prefix}n_features"] = tree_obj.n_features  # type: ignore[union-attr]
    result[f"{prefix}n_outputs"] = tree_obj.n_outputs  # type: ignore[union-attr]
    n_classes = tree_obj.n_classes  # type: ignore[union-attr]
    result[f"{prefix}n_classes"] = (
        list(n_classes) if hasattr(n_classes, "tolist") else n_classes
    )
    nodes = state["nodes"]  # type: ignore[index]
    result[f"{prefix}nodes_dtype"] = [
        [name, nodes.dtype[name].str]
        for name in nodes.dtype.names  # type: ignore[union-attr]
    ]
    return result


def _make_tree(
    arrays: dict[str, np.ndarray],
    fitted_state: dict[str, Any],
    prefix: str,
) -> object:
    """Create and restore a sklearn Tree object from arrays and state."""
    from sklearn.tree._tree import Tree

    n_features = fitted_state[f"{prefix}n_features"]
    n_classes = fitted_state[f"{prefix}n_classes"]
    n_outputs = fitted_state[f"{prefix}n_outputs"]
    n_classes_arr = np.array(n_classes) if isinstance(n_classes, list) else n_classes

    tree = Tree(
        n_features=n_features,
        n_classes=n_classes_arr,
        n_outputs=n_outputs,
    )

    dtype_spec = fitted_state[f"{prefix}nodes_dtype"]
    node_dtype = np.dtype([(name, dstr) for name, dstr in dtype_spec])
    assert node_dtype.names is not None
    first_field = node_dtype.names[0]
    n_nodes = arrays[f"{prefix}nodes_{first_field}"].shape[0]
    nodes = np.empty(n_nodes, dtype=node_dtype)
    for field in node_dtype.names:
        nodes[field] = arrays[f"{prefix}nodes_{field}"]

    tree_state: dict[str, Any] = {
        "max_depth": fitted_state[f"{prefix}max_depth"],
        "node_count": fitted_state[f"{prefix}node_count"],
        "nodes": nodes,
        "values": arrays[f"{prefix}values"],
    }
    tree.__setstate__(tree_state)
    return tree


def _make_tree_estimator(
    arrays: dict[str, np.ndarray],
    fitted_state: dict[str, Any],
    prefix: str,
    is_classifier: bool,
    n_features_in: int,
    n_outputs: int,
) -> _TreeEstimator:
    """Create a fitted DecisionTree estimator from serialized state."""
    tree_est: _TreeEstimator = (
        DecisionTreeClassifier() if is_classifier else DecisionTreeRegressor()
    )
    tree_est.tree_ = _make_tree(arrays, fitted_state, prefix)  # type: ignore[attr-defined]
    tree_est.n_features_in_ = n_features_in  # type: ignore[attr-defined]
    tree_est.n_outputs_ = n_outputs  # type: ignore[attr-defined]
    # Classifiers need n_classes_ and classes_ for predict_proba
    if is_classifier:
        n_classes = fitted_state[f"{prefix}n_classes"]
        tree_est.n_classes_ = (
            max(n_classes) if isinstance(n_classes, list) else n_classes
        )  # type: ignore[attr-defined]
    return tree_est


def _arrays_from_tree_ensemble(
    estimator: _TreeEnsemble, prefix: str
) -> dict[str, np.ndarray]:
    """Extract arrays from all trees in a RandomForest or GradientBoosting."""
    arrays: dict[str, np.ndarray] = {}

    if isinstance(estimator, _RandomForest):
        trees = estimator.estimators_
    else:
        # GB: estimators_ is ndarray (n_estimators, n_trees_per_iter)
        trees = [
            estimator.estimators_[i][j]
            for i in range(estimator.estimators_.shape[0])
            for j in range(estimator.estimators_.shape[1])
        ]

    for i, tree_est in enumerate(trees):
        arrays.update(_arrays_from_tree(tree_est.tree_, f"{prefix}_trees/{i}/"))

    # Classifier classes
    if hasattr(estimator, "classes_"):
        arrays[f"{prefix}classes_"] = np.asarray(estimator.classes_)  # type: ignore[union-attr]

    # GradientBoosting: train_score_ and init_ arrays
    if isinstance(estimator, _GradientBoosting):
        if hasattr(estimator, "train_score_"):
            arrays[f"{prefix}train_score_"] = np.asarray(estimator.train_score_)
        # init_ is a DummyRegressor/DummyClassifier with constant_/class_prior_
        if hasattr(estimator, "init_") and hasattr(estimator.init_, "constant_"):
            arrays[f"{prefix}_init/constant_"] = np.asarray(estimator.init_.constant_)  # type: ignore[union-attr]
        if hasattr(estimator, "init_") and hasattr(estimator.init_, "class_prior_"):
            arrays[f"{prefix}_init/class_prior_"] = np.asarray(
                estimator.init_.class_prior_  # type: ignore[union-attr]
            )
        if hasattr(estimator, "init_") and hasattr(estimator.init_, "classes_"):
            arrays[f"{prefix}_init/classes_"] = np.asarray(estimator.init_.classes_)  # type: ignore[union-attr]

    return arrays


def _state_from_tree_ensemble(estimator: _TreeEnsemble, prefix: str) -> dict[str, Any]:
    """Collect fitted state from a tree ensemble."""
    state: dict[str, Any] = {}

    if isinstance(estimator, _RandomForest):
        trees = estimator.estimators_
        state[f"{prefix}n_trees"] = len(trees)
    else:
        shape = estimator.estimators_.shape
        state[f"{prefix}estimators_shape"] = list(shape)
        trees = [
            estimator.estimators_[i][j]
            for i in range(shape[0])
            for j in range(shape[1])
        ]
        state[f"{prefix}n_estimators_"] = estimator.n_estimators_
        if hasattr(estimator, "init_"):
            state[f"{prefix}_init/type"] = get_sklearn_public_path(
                type(estimator.init_)
            )

    for i, tree_est in enumerate(trees):
        state.update(_state_from_tree(tree_est.tree_, f"{prefix}_trees/{i}/"))

    for attr in ("n_features_in_", "n_outputs_", "n_classes_"):
        if hasattr(estimator, attr):
            val = getattr(estimator, attr)
            state[f"{prefix}{attr}"] = val

    return state


def _restore_tree_ensemble(
    estimator: _TreeEnsemble,
    arrays: dict[str, np.ndarray],
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    """Restore a tree ensemble from arrays and state."""
    # Restore ensemble-level attrs
    for attr in ("n_features_in_", "n_outputs_", "n_classes_"):
        key = f"{prefix}{attr}"
        if key in fitted_state:
            setattr(estimator, attr, fitted_state[key])

    classes_key = f"{prefix}classes_"
    if classes_key in arrays:
        estimator.classes_ = arrays[classes_key]  # type: ignore[union-attr]
        if not hasattr(estimator, "n_classes_"):
            estimator.n_classes_ = len(estimator.classes_)  # type: ignore[attr-defined]

    is_clf = isinstance(estimator, (RandomForestClassifier, GradientBoostingClassifier))
    n_feat = fitted_state[f"{prefix}n_features_in_"]
    n_out = fitted_state.get(f"{prefix}n_outputs_", 1)

    if isinstance(estimator, _RandomForest):
        n_trees = fitted_state[f"{prefix}n_trees"]
        estimator.estimators_ = [
            _make_tree_estimator(
                arrays,
                fitted_state,
                f"{prefix}_trees/{i}/",
                is_clf,
                n_feat,
                n_out,
            )
            for i in range(n_trees)
        ]
    else:
        # GradientBoosting
        shape = fitted_state[f"{prefix}estimators_shape"]
        estimator.n_estimators_ = fitted_state[f"{prefix}n_estimators_"]
        ts_key = f"{prefix}train_score_"
        if ts_key in arrays:
            estimator.train_score_ = arrays[ts_key]

        # Restore init_
        init_type_key = f"{prefix}_init/type"
        if init_type_key in fitted_state:
            from skeights._utils import safe_import

            init_path = fitted_state[init_type_key]
            init_cls = safe_import(init_path)
            init = init_cls()
            init.n_outputs_ = n_out  # type: ignore[attr-defined]
            init.n_features_in_ = n_feat  # type: ignore[attr-defined]
            # Set _strategy (private attr derived from strategy param)
            init._strategy = init.strategy  # type: ignore[attr-defined]
            const_key = f"{prefix}_init/constant_"
            if const_key in arrays:
                init.constant_ = arrays[const_key]  # type: ignore[attr-defined]
            prior_key = f"{prefix}_init/class_prior_"
            if prior_key in arrays:
                init.class_prior_ = arrays[prior_key]  # type: ignore[attr-defined]
            init_classes_key = f"{prefix}_init/classes_"
            if init_classes_key in arrays:
                init.classes_ = arrays[init_classes_key]  # type: ignore[attr-defined]
                init.n_classes_ = len(arrays[init_classes_key])  # type: ignore[attr-defined]
            estimator.init_ = init

        # Create tree grid -- GB trees are always regressors (even for classification)
        estimator.estimators_ = np.empty(shape, dtype=object)
        idx = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                estimator.estimators_[i][j] = _make_tree_estimator(
                    arrays,
                    fitted_state,
                    f"{prefix}_trees/{idx}/",
                    is_classifier=False,
                    n_features_in=n_feat,
                    n_outputs=n_out,
                )
                idx += 1

        # Reinstate internal loss object needed for _raw_predict
        estimator._loss = estimator._get_loss(sample_weight=None)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Public dispatch API (called from _core.py)
# ---------------------------------------------------------------------------


def collect_state(estimator: BaseEstimator, prefix: str) -> dict[str, Any]:
    """Collect fitted state for tree-based estimators."""
    state: dict[str, Any] = {}
    if isinstance(estimator, (_RandomForest, _GradientBoosting)):
        return _state_from_tree_ensemble(estimator, prefix)  # type: ignore[arg-type]
    if isinstance(estimator, (DecisionTreeRegressor, DecisionTreeClassifier)):
        state.update(_state_from_tree(estimator.tree_, f"{prefix}_tree/"))
        for attr in ("n_features_in_", "n_outputs_", "n_classes_"):
            if hasattr(estimator, attr):
                val = getattr(estimator, attr)
                state[f"{prefix}{attr}"] = (
                    int(val) if isinstance(val, (int, np.integer)) else val
                )
        if hasattr(estimator, "classes_"):
            pass  # stored as array
        return state
    return state


def restore_state(
    estimator: BaseEstimator,
    fitted_state: dict[str, Any],
    prefix: str,
) -> None:
    """Restore fitted state for tree-based estimators (no-op: handled in arrays)."""
    # Tree ensembles and standalone trees are handled in the arrays path.
    return


def extract_arrays(estimator: BaseEstimator, prefix: str) -> dict[str, np.ndarray]:
    """Extract arrays from tree-based estimators."""
    if isinstance(estimator, (_RandomForest, _GradientBoosting)):
        return _arrays_from_tree_ensemble(estimator, prefix)  # type: ignore[arg-type]
    if isinstance(estimator, (DecisionTreeRegressor, DecisionTreeClassifier)):
        arrays: dict[str, np.ndarray] = {}
        arrays.update(_arrays_from_tree(estimator.tree_, f"{prefix}_tree/"))
        if hasattr(estimator, "classes_"):
            arrays[f"{prefix}classes_"] = np.asarray(estimator.classes_)
        return arrays
    return {}


def restore_arrays(
    estimator: BaseEstimator,
    arrays: dict[str, np.ndarray],
    prefix: str,
    fitted_state: dict[str, Any] | None = None,
) -> None:
    """Restore arrays for tree-based estimators."""
    if isinstance(estimator, (_RandomForest, _GradientBoosting)):
        assert fitted_state is not None, (
            "Tree ensemble restoration requires fitted_state"
        )
        _restore_tree_ensemble(estimator, arrays, fitted_state, prefix)  # type: ignore[arg-type]
        return
    if isinstance(estimator, (DecisionTreeRegressor, DecisionTreeClassifier)):
        assert fitted_state is not None
        estimator.tree_ = _make_tree(  # type: ignore[attr-defined]
            arrays, fitted_state, f"{prefix}_tree/"
        )
        for attr in ("n_features_in_", "n_outputs_", "n_classes_"):
            key = f"{prefix}{attr}"
            if key in fitted_state:
                setattr(estimator, attr, fitted_state[key])
        classes_key = f"{prefix}classes_"
        if classes_key in arrays:
            estimator.classes_ = arrays[classes_key]  # type: ignore[attr-defined]


def handles(estimator: BaseEstimator) -> bool:
    """Return True if this module handles the given estimator type."""
    return isinstance(
        estimator,
        (
            DecisionTreeRegressor,
            DecisionTreeClassifier,
            RandomForestRegressor,
            RandomForestClassifier,
            GradientBoostingRegressor,
            GradientBoostingClassifier,
        ),
    )
