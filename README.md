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

skeights decomposes a fitted estimator into two files:
- **`.safetensors`**: numeric arrays (weights, fitted params) in a safe, fast format
- **`.json`**: hyperparameters and scalar fitted state, human-readable

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
- **MLPRegressor**: multi-layer perceptron
- **GaussianProcessRegressor / Classifier**: including composite kernels
- **HistGradientBoostingRegressor / Classifier**: including bin mapper state
- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler
- **Pipelines**: any Pipeline composed of supported estimators

## License

MIT
