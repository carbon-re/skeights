"""Generate golden test fixtures.

Run this script to regenerate fixtures when intentionally changing
the serialization format. Commit the results to tests/fixtures/.

    python tests/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import skeights

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixed data for reproducibility
rng = np.random.default_rng(42)
X_train = rng.standard_normal((80, 3))
y_train = X_train[:, 0] * 2 + X_train[:, 1] - 0.5 * X_train[:, 2]
X_test = rng.standard_normal((20, 3))

MODELS: dict[str, object] = {
    "ridge": Ridge(alpha=0.1),
    "pipeline": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=0.1)),
        ]
    ),
    "random_forest": RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=5, max_depth=3, random_state=0
    ),
    "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=5, random_state=0),
    "mlp": MLPRegressor(hidden_layer_sizes=(8,), random_state=0, max_iter=500),
}

try:
    import lightgbm as lgb

    MODELS["lgbm"] = lgb.LGBMRegressor(n_estimators=5, max_depth=3, verbose=-1)
except ImportError:
    pass

try:
    import xgboost as xgb

    MODELS["xgb"] = xgb.XGBRegressor(n_estimators=5, max_depth=3)
except ImportError:
    pass


def main() -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)

    # Save test data
    np.save(FIXTURES_DIR / "X_test.npy", X_test)

    for name, model in MODELS.items():
        model.fit(X_train, y_train)  # type: ignore[union-attr]
        skeights.save(
            model,  # type: ignore[arg-type]
            FIXTURES_DIR / f"{name}.safetensors",
            FIXTURES_DIR / f"{name}.json",
        )
        preds = model.predict(X_test)  # type: ignore[union-attr]
        np.save(FIXTURES_DIR / f"{name}_preds.npy", preds)
        print(f"  {name}: saved ({preds.shape})")

    print(f"\nGenerated on sklearn {sklearn.__version__}, numpy {np.__version__}")


if __name__ == "__main__":
    main()
