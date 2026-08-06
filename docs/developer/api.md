# API Reference

## Core functions

| Function | Description |
|---|---|
| `save(estimator, arrays_path, state_path, format=None)` | Serialize to safetensors + JSON files |
| `load(arrays_path, state_path)` | Load from files, return fitted estimator |
| `serialize(estimator, format=None)` | Return `(state_dict, arrays_dict)` in memory |
| `deserialize(state, arrays)` | Reconstruct estimator from dicts |
| `get_model_params(estimator)` | Recursively extract hyperparameters |
| `set_model_params(estimator, params)` | Recursively set hyperparameters |

## `save`

```python
skeights.save(
    estimator: BaseEstimator,
    arrays_path: str,
    state_path: str,
    format: str | None = None,
) -> None
```

Serialize a fitted estimator to a pair of files:

- `arrays_path`: safetensors file containing numpy arrays (weights, coefficients, etc.)
- `state_path`: JSON file containing hyperparameters and scalar fitted state.
- `format`: optional, `"native"` to use the library's own format for LightGBM/XGBoost models. Default uses columnar tensors.

## `load`

```python
skeights.load(
    arrays_path: str,
    state_path: str,
) -> BaseEstimator
```

Load a fitted estimator from files previously created by `save`.

## `serialize`

```python
skeights.serialize(
    estimator: BaseEstimator,
    format: str | None = None,
) -> tuple[dict, dict]
```

Like `save`, but returns `(state_dict, arrays_dict)` in memory instead of writing to files.

## `deserialize`

```python
skeights.deserialize(
    state: dict,
    arrays: dict,
) -> BaseEstimator
```

Reconstruct a fitted estimator from the dicts returned by `serialize`.

## `get_model_params`

```python
skeights.get_model_params(
    estimator: BaseEstimator,
) -> dict
```

Recursively extract hyperparameters from an estimator (including pipeline steps).

## `set_model_params`

```python
skeights.set_model_params(
    estimator: BaseEstimator,
    params: dict,
) -> None
```

Recursively set hyperparameters on an estimator.
