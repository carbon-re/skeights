"""DecisionTree, RandomForest, and GradientBoosting round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .conftest import round_trip


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
