# API Reference

## `save`

```python
def save(
    estimator: BaseEstimator,
    arrays_path: str | Path,
    state_path: str | Path,
    format: str | None = None,
) -> None: ...
```

Serialize a fitted estimator to a pair of files:

- `arrays_path`: safetensors file containing numpy arrays (weights, coefficients, etc.)
- `state_path`: JSON file containing hyperparameters and scalar fitted state.
- `format`: optional, `"native"` to use the library's own format for LightGBM/XGBoost models. Default uses columnar tensors.

## `load`

```python
def load(
    arrays_path: str | Path,
    state_path: str | Path,
) -> BaseEstimator: ...
```

Load a fitted estimator from files previously created by `save`.
Reconstructs the full model hierarchy (pipelines, nested estimators)
from the JSON config, restores all fitted arrays, and auto-detects
the serialization format (e.g. columnar vs native for tree models).
The returned estimator is ready for inference or further training.

## `serialize`

```python
def serialize(
    estimator: BaseEstimator,
    format: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]: ...
```

Like `save`, but returns `(state_dict, arrays_dict)` in memory instead of writing to files.

## `deserialize`

```python
def deserialize(
    state: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> BaseEstimator: ...
```

Reconstruct a fitted estimator from the dicts returned by `serialize`.

## `get_model_params`

```python
def get_model_params(
    estimator: BaseEstimator,
) -> dict[str, Any]: ...
```

Recursively extract hyperparameters from an estimator. For composite
models like `Pipeline` or `TransformedTargetRegressor`, it traverses
the model hierarchy and returns a nested dict where each sub-model's
parameters are grouped under its step name.

For example, given `Pipeline([("scaler", StandardScaler()), ("model", Ridge())])`,
the keys `"scaler"` and `"model"` match the step names from the pipeline:

```python
{"steps": {
    "scaler": {"with_mean": True, "type": "sklearn.preprocessing.StandardScaler", ...},
    "model":  {"alpha": 0.1, "type": "sklearn.linear_model.Ridge", ...}
}, ...}
```

## `set_model_params`

```python
def set_model_params(
    estimator: BaseEstimator,
    params: dict[str, Any],
) -> BaseEstimator: ...
```

Set hyperparameters on an estimator using the same nested structure
returned by `get_model_params`. For composite models, it traverses
the hierarchy and applies parameters to each sub-model. You only
need to include the parameters you want to change.

```python
skeights.set_model_params(pipe, {"steps": {"model": {"alpha": 0.5}}})
```

## `get_sklearn_public_path`

```python
def get_sklearn_public_path(
    cls: type,
) -> str: ...
```

Return the stable public import path for a scikit-learn class. Mostly
useful internally when developing handlers, but exposed for external use.
sklearn places classes in private submodules (e.g.
`sklearn.preprocessing._data.StandardScaler`) but re-exports them
from public packages (`sklearn.preprocessing.StandardScaler`). This
function resolves the shortest public path, falling back to the full
private path if no public re-export exists.

```python
from sklearn.preprocessing import StandardScaler
skeights.get_sklearn_public_path(StandardScaler)
# "sklearn.preprocessing.StandardScaler"
```
