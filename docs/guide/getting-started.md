# Getting Started

## Installation

```bash
pip install skeights
```

For LightGBM or XGBoost support:

```bash
pip install skeights[lightgbm]
pip install skeights[xgboost]
pip install skeights[all]  # both
```

### Compatibility

Requires scikit-learn >= 1.5, LightGBM >= 4.4 (optional),
XGBoost >= 2.1 (optional). CI tests against scikit-learn 1.5,
1.6, and latest; LightGBM 4.4 and latest; XGBoost 2.1 and
latest.

Saved models are forward-compatible on a best-effort basis: we
test loading fixtures saved on older library versions with newer
ones, but don't guarantee cross-version compatibility. skeights
will emit a warning when loading a model saved with a different
library version.

## Save and load

```python
import skeights
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1)),
])
pipe.fit(X_train, y_train)

# Save to files
skeights.save(pipe, "model.safetensors", "model.json")

# Load back
loaded = skeights.load("model.safetensors", "model.json")
predictions = loaded.predict(X_test)
```

## In-memory serialization

If you don't want to write to disk (e.g. for storing in a database
or sending over the network):

```python
state, arrays = skeights.serialize(pipe)

# Later...
loaded = skeights.deserialize(state, arrays)
```

## Inspecting hyperparameters

```python
params = skeights.get_model_params(pipe)
# Returns a nested dict of all hyperparameters

skeights.set_model_params(pipe, {"model__alpha": 0.5})
```
