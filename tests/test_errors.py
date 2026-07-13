"""Error handling tests."""

from __future__ import annotations

import pytest
from sklearn.neighbors import KNeighborsRegressor

from skeights._core import _arrays_from_estimator
from skeights._utils import safe_import


def test_unsupported_estimator_raises(regression_data):
    X, y = regression_data
    model = KNeighborsRegressor()
    model.fit(X, y["y"])
    with pytest.raises(NotImplementedError):
        _arrays_from_estimator(model)


def test_safe_import_allows_sklearn():
    cls = safe_import("sklearn.linear_model.Ridge")
    from sklearn.linear_model import Ridge

    assert cls is Ridge


def test_safe_import_blocks_arbitrary_module():
    with pytest.raises(ValueError, match="not in the allowed list"):
        safe_import("os.system")


def test_safe_import_blocks_subprocess():
    with pytest.raises(ValueError, match="not in the allowed list"):
        safe_import("subprocess.Popen")
