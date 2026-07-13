<p align="center">
  <img src="https://raw.githubusercontent.com/carbon-re/skeights/main/logo.png" alt="skeights logo" width="200">
</p>

# skeights

Serialize fitted scikit-learn models to [safetensors](https://github.com/huggingface/safetensors) + JSON.

No pickle. No joblib. Just weights and config.

## Why?

Pickle is the default way to save sklearn models, but it's insecure
(arbitrary code execution on load), fragile (breaks across versions),
and opaque (you can't inspect what's inside without loading it).

skeights splits a model into two layers:

- **`.json`**: hyperparameters, fitted scalars, and structural config.
  Human-readable, greppable, diffable. You can inspect how a model
  is configured without deserializing it or running any code.
- **`.safetensors`**: the numeric bulk -- coefficients, tree split
  arrays, leaf values -- as dense typed arrays in the
  [safetensors](https://github.com/huggingface/safetensors) format.

The JSON layer makes configuration inspectable. The tensor layer is
where the compactness comes from: typed binary arrays instead of
numbers encoded as text, which matters most for large tree ensembles.
The full model is not "human-readable" -- the numeric contents live
in the tensors -- but the parts you actually want to inspect are.

Why safetensors specifically: it is memory-mappable,
language-agnostic, and widely adopted across the ML ecosystem. The
weight payload is readable outside Python and outside skeights.
Loading safetensors does not execute arbitrary code.

**Security note**: skeights does not use pickle or joblib. The JSON
state file names the Python classes to instantiate (e.g.
`sklearn.linear_model.Ridge`), but the loader only allows imports
from `sklearn`, `lightgbm`, and `xgboost`. Arbitrary module imports
from crafted JSON files are blocked.

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

| Status | Estimators |
|---|---|
| **Supported** | Ridge, Lasso, LinearRegression, LogisticRegression, and other linear models. MLPRegressor/Classifier. DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting (regressors and classifiers). LGBMRegressor/Classifier (columnar tensors or native text), XGBRegressor/Classifier (columnar tensors or native JSON). GaussianProcessRegressor/Classifier (including composite kernels). TransformedTargetRegressor. StandardScaler, MinMaxScaler, RobustScaler. Pipelines composed of any of the above. |
| **Not yet implemented** | CatBoost, other ensemble meta-estimators (VotingClassifier, StackingRegressor, etc.), PCA and other decomposition transforms. Open an issue or PR if you need any of these. |
| **Not planned** | Cross-version sklearn migration (use [sklearn-migrator](https://github.com/ibis-ssl/sklearn-migrator)). General-purpose secure persistence with broad estimator coverage (use [skops](https://github.com/skops-dev/skops)). |

## When to use something else

There are other tools for serializing sklearn models. Here is when
to reach for them instead.

**[skops](https://github.com/skops-dev/skops)** is the actively
maintained, scikit-learn-adjacent option for secure persistence,
referenced in sklearn's own docs. It covers pipelines, XGBoost,
LightGBM, has compression, model inspection, and Hugging Face Hub
integration. Reach for skops if you want the broadest, most
battle-tested secure persistence and do not specifically need the
safetensors format or the readable-config / compact-weights split.
skeights' differentiator over skops is narrow: safetensors-native
storage (standard, memory-mappable, cross-language) with a
structure/weights separation that makes config inspectable and
tree ensembles compact.

**[sklearn-migrator](https://github.com/ibis-ssl/sklearn-migrator)**
is purpose-built for loading models across different sklearn
versions, with a peer-reviewed paper behind it. Reach for it if
cross-version migration is your problem. skeights does not guarantee
cross-version support. Note that sklearn-migrator does not cover
pipelines, which skeights does.

**[sklearn-json](https://github.com/mlrequest/sklearn-json)** stores
models as JSON. It has not been updated in several years.

> **Note**: the feature descriptions of skops and sklearn-migrator
> above reflect their state as of mid-2025. Check their current
> docs for the latest.

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
