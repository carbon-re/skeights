<p align="center">
  <img src="https://raw.githubusercontent.com/carbon-re/skeights/main/logo.png" alt="skeights logo" width="200">
</p>

# skeights

Serialize fitted scikit-learn models to [safetensors](https://github.com/huggingface/safetensors) + JSON.

No pickle. No joblib. Just weights and config.

## Install

```bash
pip install skeights
```

## Quick start

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

# Save
skeights.save(pipe, "model.safetensors", "model.json")

# Load and predict
loaded = skeights.load("model.safetensors", "model.json")
predictions = loaded.predict(X_test)
```

## Documentation

Full docs, API reference, and supported estimators: **[carbon-re.github.io/skeights](https://carbon-re.github.io/skeights)**

## License

MIT
