# Supported Estimators

## Supported

| Category | Estimators |
|---|---|
| Linear models | `Ridge`, `Lasso`, `LinearRegression`, `LogisticRegression`, and other linear models |
| Neural networks | `MLPRegressor`, `MLPClassifier` |
| Trees | `DecisionTreeRegressor`, `DecisionTreeClassifier` |
| Tree ensembles (sklearn) | `RandomForestRegressor`, `RandomForestClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` |
| LightGBM | `LGBMRegressor`, `LGBMClassifier` (columnar tensors or native text) |
| XGBoost | `XGBRegressor`, `XGBClassifier` (columnar tensors or native JSON) |
| Gaussian processes | `GaussianProcessRegressor`, `GaussianProcessClassifier` (including composite kernels) |
| Preprocessing | `StandardScaler`, `MinMaxScaler`, `RobustScaler` |
| Composition | `Pipeline`, `TransformedTargetRegressor` |

## Not yet implemented

| Category | Estimators |
|---|---|
| Boosting | `CatBoost` |
| Meta-estimators | `VotingClassifier`, `StackingRegressor`, etc. |
| Decomposition | `PCA` and other decomposition transforms |

Open an issue or PR if you need any of these.

## Not planned

| Goal | Use instead |
|---|---|
| Cross-version sklearn migration | [sklearn-migrator](https://github.com/anvaldes/sklearn-migrator) |
| Broad secure persistence | [skops](https://github.com/skops-dev/skops) |
