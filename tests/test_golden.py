"""Golden tests: load fixtures and verify prediction equivalence.

Three fixture types:
- Legacy (headerless): saved on sklearn 1.5 before format tags existed.
  Tests backward compatibility with old artifacts.
- Columnar: saved with the default columnar-tensors format.
- Native: saved with explicit format="native" (format tag present).

If these tests fail on a newer sklearn/lightgbm/xgboost, the
serialization format is not forward-compatible for that estimator type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import skeights

FIXTURES_DIR = Path(__file__).parent / "fixtures"

MODELS = [
    "ridge",
    "pipeline",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "mlp",
]


def _can_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


# Legacy headerless fixtures (backward compat)
OPTIONAL_MODELS = [
    pytest.param(
        "lgbm",
        marks=pytest.mark.skipif(
            not _can_import("lightgbm"),
            reason="lightgbm not installed",
        ),
    ),
    pytest.param(
        "xgb",
        marks=pytest.mark.skipif(
            not _can_import("xgboost"),
            reason="xgboost not installed",
        ),
    ),
]

# Columnar and native fixtures
TREE_FORMAT_MODELS = [
    pytest.param(
        "lgbm_columnar",
        marks=pytest.mark.skipif(
            not _can_import("lightgbm"),
            reason="lightgbm not installed",
        ),
    ),
    pytest.param(
        "lgbm_native",
        marks=pytest.mark.skipif(
            not _can_import("lightgbm"),
            reason="lightgbm not installed",
        ),
    ),
    pytest.param(
        "xgb_columnar",
        marks=pytest.mark.skipif(
            not _can_import("xgboost"),
            reason="xgboost not installed",
        ),
    ),
    pytest.param(
        "xgb_native",
        marks=pytest.mark.skipif(
            not _can_import("xgboost"),
            reason="xgboost not installed",
        ),
    ),
]


@pytest.fixture
def X_test() -> np.ndarray:
    return np.load(FIXTURES_DIR / "X_test.npy")


@pytest.mark.parametrize("model_name", MODELS + OPTIONAL_MODELS + TREE_FORMAT_MODELS)
def test_golden_predictions(model_name: str, X_test: np.ndarray):
    loaded = skeights.load(
        FIXTURES_DIR / f"{model_name}.safetensors",
        FIXTURES_DIR / f"{model_name}.json",
    )
    expected = np.load(FIXTURES_DIR / f"{model_name}_preds.npy")
    actual = loaded.predict(X_test)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
