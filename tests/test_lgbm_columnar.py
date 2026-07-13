"""Comprehensive round-trip tests for LightGBM columnar tensor serialization.

Tests prediction equivalence across regression, binary classification,
and multiclass classification, for both columnar-tensors and native-text
formats, including edge cases.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

import skeights  # noqa: E402
from skeights._core import _arrays_from_estimator, _collect_fitted_state  # noqa: E402
from skeights._utils import json_default  # noqa: E402

from .conftest import round_trip  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_Xy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    y = X[:, 0] * 3.0 + X[:, 3] * -1.5 + rng.standard_normal(200) * 0.1
    return X, y


@pytest.fixture
def binary_Xy():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_Xy():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 10))
    y = np.digitize(X[:, 0], bins=[-0.5, 0.5])  # 3 classes
    return X, y


# ---------------------------------------------------------------------------
# Model configs to parametrise over
# ---------------------------------------------------------------------------

REGRESSOR_CONFIGS = [
    pytest.param(
        {"n_estimators": 1, "max_depth": 1, "verbose": -1},
        id="single-stump",
    ),
    pytest.param(
        {"n_estimators": 5, "max_depth": 3, "verbose": -1},
        id="shallow",
    ),
    pytest.param(
        {"n_estimators": 50, "max_depth": 6, "verbose": -1},
        id="deep",
    ),
    pytest.param(
        {"n_estimators": 10, "max_depth": -1, "num_leaves": 63, "verbose": -1},
        id="leafwise",
    ),
]

CLASSIFIER_CONFIGS = [
    pytest.param(
        {"n_estimators": 5, "max_depth": 3, "verbose": -1},
        id="shallow",
    ),
    pytest.param(
        {"n_estimators": 20, "max_depth": 5, "verbose": -1},
        id="medium",
    ),
]

FORMATS = [
    pytest.param(None, id="columnar"),
    pytest.param("native", id="native"),
]


# ---------------------------------------------------------------------------
# Round-trip tests: predict equivalence
# ---------------------------------------------------------------------------


class TestRegressorRoundTrip:
    @pytest.mark.parametrize("config", REGRESSOR_CONFIGS)
    @pytest.mark.parametrize("format", FORMATS)
    def test_predict(self, regression_Xy, config, format):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(**config)
        model.fit(X, y)

        state, arrays = skeights.serialize(model, format=format)
        restored = skeights.deserialize(state, arrays)

        np.testing.assert_allclose(
            model.predict(X), restored.predict(X), atol=0, rtol=0,
        )

    @pytest.mark.parametrize("config", REGRESSOR_CONFIGS)
    def test_raw_score(self, regression_Xy, config):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(**config)
        model.fit(X, y)

        state, arrays = skeights.serialize(model)
        restored = skeights.deserialize(state, arrays)

        np.testing.assert_allclose(
            model.predict(X, raw_score=True),
            restored.predict(X, raw_score=True),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize("format", FORMATS)
    def test_disk_round_trip(self, regression_Xy, format, tmp_path):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=10, max_depth=4, verbose=-1)
        model.fit(X, y)

        skeights.save(
            model,
            tmp_path / "m.safetensors",
            tmp_path / "m.json",
            format=format,
        )
        loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")

        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X), atol=0, rtol=0,
        )


class TestBinaryClassifierRoundTrip:
    @pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
    @pytest.mark.parametrize("format", FORMATS)
    def test_predict(self, binary_Xy, config, format):
        X, y = binary_Xy
        model = lgb.LGBMClassifier(**config)
        model.fit(X, y)

        state, arrays = skeights.serialize(model, format=format)
        restored = skeights.deserialize(state, arrays)

        np.testing.assert_array_equal(model.predict(X), restored.predict(X))

    @pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
    @pytest.mark.parametrize("format", FORMATS)
    def test_predict_proba(self, binary_Xy, config, format):
        X, y = binary_Xy
        model = lgb.LGBMClassifier(**config)
        model.fit(X, y)

        state, arrays = skeights.serialize(model, format=format)
        restored = skeights.deserialize(state, arrays)

        np.testing.assert_allclose(
            model.predict_proba(X), restored.predict_proba(X), atol=0, rtol=0,
        )

    @pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
    def test_raw_score(self, binary_Xy, config):
        X, y = binary_Xy
        model = lgb.LGBMClassifier(**config)
        model.fit(X, y)

        state, arrays = skeights.serialize(model)
        restored = skeights.deserialize(state, arrays)

        np.testing.assert_allclose(
            model.predict(X, raw_score=True),
            restored.predict(X, raw_score=True),
            atol=0, rtol=0,
        )


class TestMulticlassClassifierRoundTrip:
    @pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
    @pytest.mark.parametrize("format", FORMATS)
    def test_predict(self, multiclass_Xy, config, format):
        X, y = multiclass_Xy
        model = lgb.LGBMClassifier(**config)
        model.fit(X, y)

        state, arrays = skeights.serialize(model, format=format)
        restored = skeights.deserialize(state, arrays)

        np.testing.assert_array_equal(model.predict(X), restored.predict(X))

    @pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
    @pytest.mark.parametrize("format", FORMATS)
    def test_predict_proba(self, multiclass_Xy, config, format):
        X, y = multiclass_Xy
        model = lgb.LGBMClassifier(**config)
        model.fit(X, y)

        state, arrays = skeights.serialize(model, format=format)
        restored = skeights.deserialize(state, arrays)

        np.testing.assert_allclose(
            model.predict_proba(X), restored.predict_proba(X), atol=0, rtol=0,
        )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestColumnarStructure:
    def test_format_tag(self, regression_Xy):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=3, max_depth=2, verbose=-1)
        model.fit(X, y)
        state = _collect_fitted_state(model)
        assert state["__format__"]["format"] == "columnar-tensors"
        assert state["__format__"]["library"] == "lightgbm"
        assert state["__format__"]["schema_version"] == 1

    def test_native_format_tag(self, regression_Xy):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=3, max_depth=2, verbose=-1)
        model.fit(X, y)
        state = _collect_fitted_state(model, format="native")
        assert state["__format__"]["format"] == "native-text"

    def test_tree_metadata_in_state(self, regression_Xy):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=3, max_depth=2, verbose=-1)
        model.fit(X, y)
        state = _collect_fitted_state(model)
        tree = state["tree"]
        assert tree["objective"] == "regression"
        assert tree["num_class"] == 1
        assert tree["num_tree_per_iteration"] == 1
        assert isinstance(tree["feature_names"], list)

    def test_arrays_contain_tree_tensors(self, regression_Xy):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=3, max_depth=2, verbose=-1)
        model.fit(X, y)
        arrays = _arrays_from_estimator(model)
        expected_keys = {
            "tree/split_offsets", "tree/leaf_offsets",
            "tree/split_feature", "tree/threshold",
            "tree/left_child", "tree/right_child",
            "tree/decision_type", "tree/split_gain",
            "tree/internal_value", "tree/internal_weight", "tree/internal_count",
            "tree/leaf_value", "tree/leaf_weight", "tree/leaf_count",
            "tree/shrinkage", "tree/num_cat", "tree/is_linear",
            "feature_importances_",
        }
        assert expected_keys.issubset(set(arrays.keys()))

    def test_offsets_shape(self, regression_Xy):
        X, y = regression_Xy
        n_trees = 5
        model = lgb.LGBMRegressor(n_estimators=n_trees, max_depth=2, verbose=-1)
        model.fit(X, y)
        arrays = _arrays_from_estimator(model)
        assert arrays["tree/split_offsets"].shape == (n_trees + 1,)
        assert arrays["tree/leaf_offsets"].shape == (n_trees + 1,)
        assert arrays["tree/shrinkage"].shape == (n_trees,)

    def test_state_is_json_serializable(self, regression_Xy):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
        model.fit(X, y)
        state = _collect_fitted_state(model)
        json.dumps(state, default=json_default)

    def test_no_model_blob_in_columnar_state(self, regression_Xy):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
        model.fit(X, y)
        state = _collect_fitted_state(model)
        assert "model_str" not in state


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_tree(self, regression_Xy):
        X, y = regression_Xy
        model = lgb.LGBMRegressor(n_estimators=1, max_depth=1, verbose=-1)
        model.fit(X, y)
        restored = round_trip(model)
        np.testing.assert_allclose(
            model.predict(X), restored.predict(X), atol=0, rtol=0,
        )

    def test_missing_values(self, regression_Xy):
        X, y = regression_Xy
        X_missing = X.copy()
        rng = np.random.default_rng(99)
        mask = rng.random(X_missing.shape) < 0.1
        X_missing[mask] = np.nan

        model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, verbose=-1)
        model.fit(X_missing, y)
        restored = round_trip(model)
        np.testing.assert_allclose(
            model.predict(X_missing), restored.predict(X_missing), atol=0, rtol=0,
        )

    def test_high_cardinality_features(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((500, 50))
        y = X[:, :5].sum(axis=1) + rng.standard_normal(500) * 0.1
        model = lgb.LGBMRegressor(n_estimators=20, max_depth=5, verbose=-1)
        model.fit(X, y)
        restored = round_trip(model)
        np.testing.assert_allclose(
            model.predict(X), restored.predict(X), atol=0, rtol=0,
        )

    def test_pipeline_with_lgbm(self, regression_Xy):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = regression_Xy
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)),
        ])
        pipe.fit(X, y)
        restored = round_trip(pipe)
        np.testing.assert_allclose(
            pipe.predict(X), restored.predict(X), atol=0, rtol=0,
        )
