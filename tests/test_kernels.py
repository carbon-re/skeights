"""Kernel serialization round-trip tests."""

from __future__ import annotations

import pytest
from sklearn.gaussian_process.kernels import RBF, Matern, Product, Sum, WhiteKernel

from skeights._kernels import _deserialize_kernel, _serialize_kernel


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(length_scale=2.5),
        WhiteKernel(noise_level=0.1) + RBF(length_scale=1.5),
        RBF(length_scale=1.0) * Matern(length_scale=2.0, nu=1.5),
        Sum(Product(RBF(), Matern()), WhiteKernel()),
    ],
)
def test_kernel_round_trip(kernel):
    restored = _deserialize_kernel(_serialize_kernel(kernel))
    assert type(restored) is type(kernel)
