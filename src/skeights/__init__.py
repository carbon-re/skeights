"""skeights — serialize fitted scikit-learn models to safetensors + JSON."""

__version__ = "0.1.0"

from skeights._io import load, save
from skeights._params import get_model_params, set_model_params
from skeights._utils import get_sklearn_public_path

__all__ = [
    "get_model_params",
    "get_sklearn_public_path",
    "load",
    "save",
    "set_model_params",
]
