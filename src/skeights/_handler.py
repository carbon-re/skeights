"""Base class for estimator handlers.

To add support for a new model type, subclass ``EstimatorHandler`` and
implement the five abstract methods.  Then register an instance in
``_core._get_handlers()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class EstimatorHandler(ABC):
    """Interface that every estimator handler must implement.

    A handler knows how to decompose a fitted scikit-learn (or compatible)
    estimator into:

    * A JSON-safe dict of scalar state (hyperparameters, metadata).
    * A flat dict of numpy arrays (weights, fitted arrays).

    And how to restore those pieces back onto an unfitted estimator skeleton.
    """

    @abstractmethod
    def handles(self, estimator: BaseEstimator) -> bool:
        """Return True if this handler owns the given estimator type."""
        ...

    @abstractmethod
    def collect_state(
        self,
        estimator: BaseEstimator,
        prefix: str,
        format: str | None = None,
    ) -> dict[str, Any]:
        """Extract JSON-safe scalar state from a fitted estimator."""
        ...

    @abstractmethod
    def restore_state(
        self,
        estimator: BaseEstimator,
        fitted_state: dict[str, Any],
        prefix: str,
    ) -> None:
        """Restore scalar state onto an estimator."""
        ...

    @abstractmethod
    def extract_arrays(
        self,
        estimator: BaseEstimator,
        prefix: str,
        format: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract numpy arrays from a fitted estimator."""
        ...

    @abstractmethod
    def restore_arrays(
        self,
        estimator: BaseEstimator,
        arrays: dict[str, np.ndarray],
        prefix: str,
        fitted_state: dict[str, Any] | None = None,
    ) -> None:
        """Restore numpy arrays onto an estimator."""
        ...
