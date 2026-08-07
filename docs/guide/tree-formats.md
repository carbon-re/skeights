# Tree Model Formats

LightGBM and XGBoost models are serialized as columnar tensors by
default: split features, thresholds, child pointers, and leaf values
are stored as typed numpy arrays in safetensors, with only small
scalar config (objective, feature names, etc.) in JSON.

This reduces file size by 30-85% compared to the native format and
makes the JSON human-readable.

## Using the native format

To use the library's own format instead:

```python
skeights.save(model, "model.safetensors", "model.json", format="native")
```

- **LightGBM native**: stores the booster model string (LightGBM's text format) in JSON.
- **XGBoost native**: stores the booster model as XGBoost's own JSON dict.

## Loading

The loader automatically detects whether a model was saved in
columnar or native format and handles both transparently. You
do not need to specify the format when loading.
