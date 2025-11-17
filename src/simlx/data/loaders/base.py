"""Base classes for overlap-aware data loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class OverlapAwareDataLoader(ABC):
    """Abstract interface for data loaders that yield overlapping patches."""

    @abstractmethod
    def __iter__(self) -> Iterable[Any]:
        """Yield mini-batches respecting the configured patch-overlap scheme."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
