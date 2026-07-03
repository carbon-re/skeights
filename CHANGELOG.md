# Changelog

## 0.1.1

- Fix logo not rendering on PyPI (use absolute URL)

## 0.1.0

Initial release.

- `save()` / `load()` for fitted sklearn estimators
- Supported: linear models, MLP, DecisionTree, RandomForest,
  GradientBoosting, HistGradientBoosting, GaussianProcess,
  TransformedTargetRegressor, scalers, pipelines
- Optional: LightGBM, XGBoost
- Output: safetensors (arrays) + JSON (hyperparameters, state)
- Version tracking: warns when loading with a different
  sklearn/lightgbm/xgboost version
- CI: Python 3.11-3.13, sklearn 1.5/1.6/latest, nightly
