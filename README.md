<p align="center">
  <img src="https://raw.githubusercontent.com/carbon-re/skeights/main/logo.png" alt="skeights logo" width="200">
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

# Fit your model as usual
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1)),
])
pipe.fit(X_train, y_train)

# Save
skeights.save(pipe, "model.safetensors", "model.json")

# Load and predict
loaded = skeights.load("model.safetensors", "model.json")
predictions = loaded.predict(X_test)
```

## API

| Function | Description |
|---|---|
| `save(estimator, arrays_path, state_path, format=None)` | Serialize to safetensors + JSON files |
| `load(arrays_path, state_path)` | Load from files, return fitted estimator |
| `serialize(estimator, format=None)` | Return `(state_dict, arrays_dict)` in memory |
| `deserialize(state, arrays)` | Reconstruct estimator from dicts |
| `get_model_params(estimator)` | Recursively extract hyperparameters (handles Pipelines, kernels, TTR) |
| `set_model_params(estimator, params)` | Recursively set hyperparameters |

## Supported estimators

- **Linear models**: Ridge, Lasso, LinearRegression, LogisticRegression, etc.
- **MLPRegressor / Classifier**: multi-layer perceptron
- **DecisionTreeRegressor / Classifier**: full tree serialization
- **RandomForestRegressor / Classifier**: full tree serialization
- **GradientBoostingRegressor / Classifier**: including init estimator
- **HistGradientBoostingRegressor / Classifier**: including bin mapper state
- **LGBMRegressor / Classifier**: columnar tensors (default) or native text
- **XGBRegressor / Classifier**: columnar tensors (default) or native JSON
- **GaussianProcessRegressor / Classifier**: including composite kernels
- **TransformedTargetRegressor**: target scaling wrappers
- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler
- **Pipelines**: any Pipeline composed of supported estimators

## Tree model formats

LightGBM and XGBoost models are serialized as columnar tensors by
default: split features, thresholds, child pointers, and leaf values
are stored as typed numpy arrays in safetensors, with only small
scalar config (objective, feature names, etc.) in JSON.

This gives 30-85% smaller files compared to the native format and
makes the JSON human-readable. To use the native format instead:

```python
skeights.save(model, "model.safetensors", "model.json", format="native")
```

Artifacts saved with older versions of skeights (before columnar
support) are loaded transparently -- no migration needed.

## Compatibility

skeights requires scikit-learn >= 1.5 and tests against 1.5,
1.6, and latest in CI.

Saved models are forward-compatible on a best-effort basis: we
test loading sklearn 1.5 fixtures on newer versions, but don't
guarantee cross-version compatibility.

When loading a model saved with a different sklearn version,
skeights will emit a warning.

## License

MIT
