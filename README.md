<p align="center">
  <img src="logo.png" alt="skeights logo" width="200">
</p>

# skeights

Serialize fitted scikit-learn models to [safetensors](https://github.com/huggingface/safetensors) + JSON.

No pickle. No joblib. Just weights and config.

## Why?

Pickle is the default way to save sklearn models, but it's:
- **Insecure**: arbitrary code execution on load
- **Fragile**: breaks across sklearn versions, Python versions, and platforms
- **Opaque**: you can't inspect what's inside

[skops](https://github.com/skops-dev/skops) solves the security
problem by replacing pickle with a safe binary format, but the
output is still a single opaque blob; you can't easily inspect
the hyperparameters or diff two versions of a model.

skeights separates structure from weights:
- **`.json`**: hyperparameters and scalar fitted state,
  human-readable and diffable
- **`.safetensors`**: numeric arrays (weights, fitted params)
  in a safe, fast, widely-supported format

## Install

```bash
pip install skeights
```

## Usage

```python
import skeights
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fit your model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1)),
])
pipe.fit(X_train, y_train)

# Wrap and save
model = skeights.SklearnModel(pipe)
model.fit(X_train, y_train)
skeights.save(model, "model.safetensors", "model.json")

# Load and predict
loaded = skeights.load("model.safetensors", "model.json")
predictions = loaded.predict(X_test)
```

## Supported estimators

- **Linear models**: Ridge, Lasso, LinearRegression, LogisticRegression, etc.
- **MLPRegressor / Classifier**: multi-layer perceptron
- **DecisionTreeRegressor / Classifier**: full tree serialization
- **RandomForestRegressor / Classifier**: full tree serialization
- **GradientBoostingRegressor / Classifier**: including init estimator
- **HistGradientBoostingRegressor / Classifier**: including bin mapper state
- **GaussianProcessRegressor / Classifier**: including composite kernels
- **TransformedTargetRegressor**: target scaling wrappers
- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler
- **Pipelines**: any Pipeline composed of supported estimators

## License

MIT
