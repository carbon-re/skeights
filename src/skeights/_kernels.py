"""Kernel serialization for Gaussian Process estimators."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import Kernel

from skeights._utils import get_sklearn_public_path


def _serialize_kernel(kernel: Kernel) -> dict[str, Any]:
    """Recursively convert a kernel tree to a JSON-safe dict."""
    from sklearn.gaussian_process.kernels import KernelOperator

    data: dict[str, Any] = {"type": get_sklearn_public_path(type(kernel))}
    if isinstance(kernel, KernelOperator):
        data["k1"] = _serialize_kernel(kernel.k1)
        data["k2"] = _serialize_kernel(kernel.k2)
    else:
        params = kernel.get_params(deep=False)
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
            elif isinstance(v, tuple):
                data[k] = list(v)
            else:
                data[k] = v
    return data


def _deserialize_kernel(data: dict[str, Any]) -> Kernel:
    """Reconstruct a kernel tree from a serialised dict."""
    from sklearn.gaussian_process.kernels import KernelOperator

    data = dict(data)
    type_path: str = data.pop("type")
    module_path, _, class_name = type_path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)

    if issubclass(cls, KernelOperator):
        k1 = _deserialize_kernel(data["k1"])
        k2 = _deserialize_kernel(data["k2"])
        return cls(k1=k1, k2=k2)

    # Leaf kernel -- convert list params back to correct types
    try:
        defaults = cls().get_params(deep=False)
    except TypeError:
        defaults = {}

    init_params: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, list):
            if isinstance(defaults.get(k), tuple):
                init_params[k] = tuple(v)
            else:
                init_params[k] = np.asarray(v)
        else:
            init_params[k] = v
    return cls(**init_params)
