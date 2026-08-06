# Contributing

## Development setup

Clone the repo and install in editable mode with dev dependencies:

```bash
git clone https://github.com/carbon-re/skeights.git
cd skeights
pip install -e ".[dev]"
```

Run the tests to check everything works:

```bash
pytest
```

## Running the docs locally

Install the docs dependency and start the dev server:

```bash
pip install mkdocs-material
mkdocs serve
```

This launches a local preview at `http://127.0.0.1:8000/skeights/`
that auto-reloads when you edit any docs files.

## Adding support for a new estimator

skeights uses a handler pattern to support different estimator types.
Each handler is a class that knows how to decompose a specific
estimator into JSON state + numpy arrays, and how to put them back
together.

### The contract

Subclass `skeights.EstimatorHandler` and implement five methods:

```python
from skeights._handler import EstimatorHandler

class MyHandler(EstimatorHandler):

    def handles(self, estimator):
        """Return True if this handler owns the given estimator type."""
        return isinstance(estimator, MyEstimatorClass)

    def collect_state(self, estimator, prefix, format=None):
        """Extract JSON-safe scalar state (hyperparameters, metadata).

        Returns a dict of scalar values that will be saved to JSON.
        Use the prefix to namespace keys (important for pipelines).
        """
        ...

    def restore_state(self, estimator, fitted_state, prefix):
        """Restore scalar state onto an estimator skeleton."""
        ...

    def extract_arrays(self, estimator, prefix, format=None):
        """Extract numpy arrays (weights, coefficients, etc).

        Returns a dict of numpy arrays that will be saved to safetensors.
        """
        ...

    def restore_arrays(self, estimator, arrays, prefix, fitted_state=None):
        """Restore numpy arrays onto an estimator."""
        ...
```

### Registering the handler

Add your handler to the list in `_core.py`:

```python
def _get_handlers() -> list[EstimatorHandler]:
    from skeights._mymodule import MyHandler
    # ... existing handlers ...
    return [
        # ... existing handlers ...
        MyHandler(),
    ]
```

### Key concepts

**Prefix**: every key in the state dict and arrays dict is prefixed
with a path like `scaler/` or `model/`. This is how pipelines work:
each step gets its own namespace. Always use `f"{prefix}{key}"`
when building keys.

**State vs arrays**: anything that's a scalar, string, or small list
goes in state (saved as JSON). Anything that's a numpy array goes in
arrays (saved as safetensors). The split is important because JSON
is human-readable and diffable, while safetensors is efficient for
large numeric data.

**Format**: the `format` parameter lets handlers support multiple
serialization strategies. For example, LightGBM and XGBoost support
`format="native"` (library's own format) and the default columnar
tensors format.

### Example: a simple handler

`_mlp.py` is the simplest handler to use as a reference. It
serializes MLP regressors and classifiers by:

1. Saving scalar state (`n_layers_`, `n_outputs_`, etc.) to JSON
2. Saving weight matrices (`coefs_`, `intercepts_`) as numpy arrays

### Testing

Add round-trip tests in `tests/` that:

1. Fit a model
2. Save it with `skeights.save()`
3. Load it with `skeights.load()`
4. Check predictions match the original

See `tests/test_mlp.py` for a straightforward example.
