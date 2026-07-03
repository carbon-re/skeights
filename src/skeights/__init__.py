"""skeights — serialize fitted scikit-learn models to safetensors + JSON."""

__version__ = "0.2.0"

from skeights._io import deserialize, load, save, serialize
from skeights._params import get_model_params, set_model_params
from skeights._utils import get_sklearn_public_path

__all__ = [
    "deserialize",
    "get_model_params",
    "get_sklearn_public_path",
    "load",
    "save",
    "serialize",
    "set_model_params",
]
