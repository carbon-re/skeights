# Supported Estimators

| Status | Estimators |
|---|---|
| **Supported** | `Ridge`, `Lasso`, `LinearRegression`, `LogisticRegression`, and other linear models. `MLPRegressor`/`MLPClassifier`. `DecisionTreeRegressor`/`Classifier`, `RandomForestRegressor`/`Classifier`, `GradientBoostingRegressor`/`Classifier`, `HistGradientBoostingRegressor`/`Classifier`. `LGBMRegressor`/`Classifier` (columnar tensors or native text), `XGBRegressor`/`Classifier` (columnar tensors or native JSON). `GaussianProcessRegressor`/`Classifier` (including composite kernels). `TransformedTargetRegressor`. `StandardScaler`, `MinMaxScaler`, `RobustScaler`. `Pipeline` composed of any of the above. |
| **Not yet implemented** | `CatBoost`, other ensemble meta-estimators (`VotingClassifier`, `StackingRegressor`, etc.), `PCA` and other decomposition transforms. Open an issue or PR if you need any of these. |
| **Not planned** | Cross-version sklearn migration (use [sklearn-migrator](https://github.com/anvaldes/sklearn-migrator)). General-purpose secure persistence with broad estimator coverage (use [skops](https://github.com/skops-dev/skops)). |
