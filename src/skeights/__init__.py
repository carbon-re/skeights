"""skeights — serialize fitted scikit-learn models to safetensors + JSON."""

try:
    from skeights._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

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
