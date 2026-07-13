"""Measure JSON vs safetensors file sizes for tree models."""

import os
import tempfile

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingRegressor

import skeights

X_reg, y_reg = make_regression(n_samples=500, n_features=20, random_state=42)
X_cls, y_cls = make_classification(n_samples=500, n_features=20, random_state=42)

results = []


def measure(name, model, X, y):
    model.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        skeights.save(model, f"{d}/m.safetensors", f"{d}/m.json")
        json_size = os.path.getsize(f"{d}/m.json")
        st_size = os.path.getsize(f"{d}/m.safetensors")
        results.append((name, json_size, st_size))


# XGBoost
try:
    import xgboost as xgb

    measure("XGBRegressor(100 trees)", xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=0), X_reg, y_reg)
    measure("XGBClassifier(100 trees)", xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=0), X_cls, y_cls)
except ImportError:
    print("xgboost not installed, skipping")

# LightGBM
try:
    import lightgbm as lgb

    measure("LGBMRegressor(100 trees)", lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1), X_reg, y_reg)
    measure("LGBMClassifier(100 trees)", lgb.LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1), X_cls, y_cls)
except ImportError:
    print("lightgbm not installed, skipping")

# sklearn baseline (already uses safetensors properly)
measure("sklearn GBRegressor(100 trees)", GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=0), X_reg, y_reg)

print(f"{'Model':<35} {'JSON':>10} {'Safetensors':>12} {'Total':>10}")
print("-" * 70)
for name, js, st in results:
    pct = js / (js + st) * 100
    print(f"{name:<35} {js:>9,} {st:>11,} {js + st:>9,}  (JSON={pct:.0f}%)")
