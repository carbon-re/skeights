"""XGBoost round-trip and structural tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

import skeights  # noqa: E402
from skeights._core import _arrays_from_estimator, _collect_fitted_state  # noqa: E402
from skeights._utils import json_default  # noqa: E402

from .conftest import round_trip  # noqa: E402

# ---------------------------------------------------------------------------
# Data fixtures
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
# Parametrisation
# ---------------------------------------------------------------------------

REGRESSOR_CONFIGS = [
    pytest.param(
        {"n_estimators": 1, "max_depth": 1, "random_state": 0},
        id="single-stump",
    ),
    pytest.param(
        {"n_estimators": 5, "max_depth": 3, "random_state": 0},
        id="shallow",
    ),
    pytest.param(
        {"n_estimators": 50, "max_depth": 6, "random_state": 0},
        id="deep",
    ),
]

CLASSIFIER_CONFIGS = [
    pytest.param(
        {"n_estimators": 5, "max_depth": 3, "random_state": 0},
        id="shallow",
    ),
    pytest.param(
        {"n_estimators": 20, "max_depth": 5, "random_state": 0},
        id="medium",
    ),
]

FORMATS = [
    pytest.param(None, id="columnar"),
    pytest.param("native", id="native"),
]


# ---------------------------------------------------------------------------
# Round-trip tests: regressors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", REGRESSOR_CONFIGS)
@pytest.mark.parametrize("format", FORMATS)
def test_xgb_regressor_round_trip(regression_Xy, config, format):
    X, y = regression_Xy
    model = xgb.XGBRegressor(**config)
    model.fit(X, y)
    state, arrays = skeights.serialize(model, format=format)
    restored = skeights.deserialize(state, arrays)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=0, rtol=0)


@pytest.mark.parametrize("config", REGRESSOR_CONFIGS)
def test_xgb_regressor_raw_score(regression_Xy, config):
    X, y = regression_Xy
    model = xgb.XGBRegressor(**config)
    model.fit(X, y)
    state, arrays = skeights.serialize(model)
    restored = skeights.deserialize(state, arrays)
    orig_raw = model.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    restored_raw = restored.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    np.testing.assert_allclose(orig_raw, restored_raw, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Round-trip tests: classifiers (binary + multiclass)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_fixture", ["binary_Xy", "multiclass_Xy"], ids=["binary", "multiclass"]
)
@pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
@pytest.mark.parametrize("format", FORMATS)
def test_xgb_classifier_predict(request, data_fixture, config, format):
    X, y = request.getfixturevalue(data_fixture)
    model = xgb.XGBClassifier(**config)
    model.fit(X, y)
    state, arrays = skeights.serialize(model, format=format)
    restored = skeights.deserialize(state, arrays)
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))


@pytest.mark.parametrize(
    "data_fixture", ["binary_Xy", "multiclass_Xy"], ids=["binary", "multiclass"]
)
@pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
@pytest.mark.parametrize("format", FORMATS)
def test_xgb_classifier_predict_proba(request, data_fixture, config, format):
    X, y = request.getfixturevalue(data_fixture)
    model = xgb.XGBClassifier(**config)
    model.fit(X, y)
    state, arrays = skeights.serialize(model, format=format)
    restored = skeights.deserialize(state, arrays)
    np.testing.assert_allclose(
        model.predict_proba(X), restored.predict_proba(X), atol=0, rtol=0
    )


@pytest.mark.parametrize("config", CLASSIFIER_CONFIGS)
def test_xgb_classifier_raw_margin(binary_Xy, config):
    X, y = binary_Xy
    model = xgb.XGBClassifier(**config)
    model.fit(X, y)
    state, arrays = skeights.serialize(model)
    restored = skeights.deserialize(state, arrays)
    orig_raw = model.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    restored_raw = restored.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    np.testing.assert_allclose(orig_raw, restored_raw, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Disk round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format", FORMATS)
def test_xgb_disk_round_trip(regression_Xy, format, tmp_path):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=10, max_depth=4, random_state=0)
    model.fit(X, y)
    skeights.save(model, tmp_path / "m.safetensors", tmp_path / "m.json", format=format)
    loaded = skeights.load(tmp_path / "m.safetensors", tmp_path / "m.json")
    np.testing.assert_allclose(model.predict(X), loaded.predict(X), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_xgb_format_tag(regression_Xy):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y)
    state = _collect_fitted_state(model)
    assert state["__format__"]["format"] == "columnar-tensors"
    assert state["__format__"]["library"] == "xgboost"


def test_xgb_native_format_tag(regression_Xy):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y)
    state = _collect_fitted_state(model, format="native")
    assert state["__format__"]["format"] == "native-json"


def test_xgb_tree_metadata_in_state(regression_Xy):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y)
    state = _collect_fitted_state(model)
    tree = state["tree"]
    assert "objective" in tree
    assert "learner_model_param" in tree
    assert "version" in tree


def test_xgb_arrays_contain_tree_tensors(regression_Xy):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(X, y)
    arrays = _arrays_from_estimator(model)
    expected_keys = {
        "tree/offsets",
        "tree/split_indices",
        "tree/split_conditions",
        "tree/left_children",
        "tree/right_children",
        "tree/parents",
        "tree/default_left",
        "tree/base_weights",
        "tree/sum_hessian",
        "tree/loss_changes",
        "tree/split_type",
        "feature_importances_",
    }
    assert expected_keys.issubset(set(arrays.keys()))


def test_xgb_no_model_blob_in_columnar_state(regression_Xy):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    state = _collect_fitted_state(model)
    assert "model_json" not in state


def test_xgb_state_is_json_serializable(regression_Xy):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    state = _collect_fitted_state(model)
    json.dumps(state, default=json_default)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_xgb_single_tree(regression_Xy):
    X, y = regression_Xy
    model = xgb.XGBRegressor(n_estimators=1, max_depth=1, random_state=0)
    model.fit(X, y)
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=0, rtol=0)


def test_xgb_missing_values(regression_Xy):
    X, y = regression_Xy
    X_missing = X.copy()
    rng = np.random.default_rng(99)
    mask = rng.random(X_missing.shape) < 0.1
    X_missing[mask] = np.nan
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_missing, y)
    restored = round_trip(model)
    np.testing.assert_allclose(
        model.predict(X_missing), restored.predict(X_missing), atol=0, rtol=0
    )


def test_xgb_high_cardinality_features():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((500, 50))
    y = X[:, :5].sum(axis=1) + rng.standard_normal(500) * 0.1
    model = xgb.XGBRegressor(n_estimators=20, max_depth=5, random_state=0)
    model.fit(X, y)
    restored = round_trip(model)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), atol=0, rtol=0)


def test_xgb_pipeline(regression_Xy):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = regression_Xy
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=0),
            ),
        ]
    )
    pipe.fit(X, y)
    restored = round_trip(pipe)
    np.testing.assert_allclose(pipe.predict(X), restored.predict(X), atol=0, rtol=0)
