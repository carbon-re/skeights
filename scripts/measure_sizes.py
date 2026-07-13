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


def measure(name, model, X, y, has_format=True):
    model.fit(X, y)
    with tempfile.TemporaryDirectory() as d:
        # Columnar (default)
        skeights.save(model, f"{d}/c.safetensors", f"{d}/c.json")
        c_js = os.path.getsize(f"{d}/c.json")
        c_st = os.path.getsize(f"{d}/c.safetensors")

        if has_format:
            # Native
            skeights.save(model, f"{d}/n.safetensors", f"{d}/n.json", format="native")
            n_js = os.path.getsize(f"{d}/n.json")
            n_st = os.path.getsize(f"{d}/n.safetensors")
        else:
            n_js, n_st = c_js, c_st

        results.append((name, n_js, n_st, c_js, c_st))


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

# sklearn baseline (no format option)
measure("sklearn GBReg(100 trees)", GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=0), X_reg, y_reg, has_format=False)

print(f"{'Model':<30} {'Native Total':>14} {'Columnar Total':>16} {'Reduction':>10}")
print(f"{'':30} {'(JSON + ST)':>14} {'(JSON + ST)':>16}")
print("-" * 73)
for name, n_js, n_st, c_js, c_st in results:
    n_total = n_js + n_st
    c_total = c_js + c_st
    if n_total != c_total:
        pct = f"{(1 - c_total / n_total) * 100:.0f}%"
        print(f"{name:<30} {n_total:>13,} {c_total:>15,} {pct:>10}")
        print(f"{'':30}  (J:{n_js:>8,})  (J:{c_js:>8,})")
    else:
        print(f"{name:<30} {n_total:>13,} {c_total:>15,} {'n/a':>10}")
