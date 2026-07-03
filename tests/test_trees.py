"""DecisionTree, RandomForest, and GradientBoosting tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skeights._core import _arrays_from_estimator, _collect_fitted_state

from .conftest import assert_serializable, round_trip


@pytest.mark.parametrize(
    "estimator, use_regression",
    [
        (DecisionTreeRegressor(max_depth=3, random_state=0), True),
        (DecisionTreeClassifier(max_depth=3, random_state=0), False),
    ],
    ids=["regressor", "classifier"],
)
def test_decision_tree_round_trip(
    estimator, use_regression, regression_data, binary_data
):
    X, y = regression_data if use_regression else binary_data
    target = y.iloc[:, 0]
    estimator.fit(X, target)
    restored = round_trip(estimator)
    np.testing.assert_allclose(estimator.predict(X), restored.predict(X), atol=1e-10)
    # Verify tree structure matches
    assert restored.tree_.node_count == estimator.tree_.node_count
    assert restored.tree_.max_depth == estimator.tree_.max_depth


@pytest.mark.parametrize(
    "estimator",
    [
        ensemble.RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0),
        ensemble.RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0),
    ],
    ids=["regressor", "classifier"],
)
def test_random_forest_round_trip(estimator, regression_data, binary_data):
    is_clf = isinstance(estimator, ensemble.RandomForestClassifier)
    X, y = binary_data if is_clf else regression_data
    estimator.fit(X, y.iloc[:, 0])
    restored = round_trip(estimator)
    np.testing.assert_allclose(estimator.predict(X), restored.predict(X), atol=1e-10)
    assert len(restored.estimators_) == len(estimator.estimators_)
    for orig_tree, rest_tree in zip(estimator.estimators_, restored.estimators_):
        assert rest_tree.tree_.node_count == orig_tree.tree_.node_count


@pytest.mark.parametrize(
    "estimator",
    [
        ensemble.GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=0),
        ensemble.GradientBoostingClassifier(
            n_estimators=5, max_depth=3, random_state=0
        ),
    ],
    ids=["regressor", "classifier"],
)
def test_gradient_boosting_round_trip(estimator, regression_data, binary_data):
    is_clf = isinstance(estimator, ensemble.GradientBoostingClassifier)
    X, y = binary_data if is_clf else regression_data
    estimator.fit(X, y.iloc[:, 0])
    restored = round_trip(estimator)
    np.testing.assert_allclose(estimator.predict(X), restored.predict(X), atol=1e-10)


def test_rf_out_of_sample(regression_data_split):
    X_train, y_train, X_test, _ = regression_data_split
    model = ensemble.RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)
    model.fit(X_train, y_train["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict(X_test), restored.predict(X_test), atol=1e-10
    )


def test_gb_out_of_sample(regression_data_split):
    X_train, y_train, X_test, _ = regression_data_split
    model = ensemble.GradientBoostingRegressor(
        n_estimators=10, max_depth=3, random_state=0
    )
    model.fit(X_train, y_train["y"])
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict(X_test), restored.predict(X_test), atol=1e-10
    )


def test_rf_arrays_contain_tree_nodes(regression_data):
    X, y = regression_data
    model = ensemble.RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y["y"])
    arrays = _arrays_from_estimator(model)
    assert "_trees/0/values" in arrays
    assert "_trees/1/values" in arrays
    assert "_trees/2/values" in arrays
    assert any("nodes_" in k for k in arrays)


def test_rf_state_contains_tree_count(regression_data):
    X, y = regression_data
    model = ensemble.RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y["y"])
    assert _collect_fitted_state(model)["n_trees"] == 3


def test_gb_state_contains_init_type(regression_data):
    X, y = regression_data
    model = ensemble.GradientBoostingRegressor(
        n_estimators=3, max_depth=2, random_state=0
    )
    model.fit(X, y["y"])
    fitted = _collect_fitted_state(model)
    assert "_init/type" in fitted
    assert "DummyRegressor" in fitted["_init/type"]


@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeRegressor(max_depth=3, random_state=0),
        ensemble.RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0),
        ensemble.GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0),
    ],
    ids=["dt", "rf", "gb"],
)
def test_tree_serialization_formats(estimator, regression_data):
    X, y = regression_data
    estimator.fit(X, y["y"])
    assert_serializable(estimator)
