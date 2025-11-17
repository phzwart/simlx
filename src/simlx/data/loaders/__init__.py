"""Data loader factory and interfaces.

This package will expose utilities for constructing datasets and dataloaders
with partial patch overlap support.
"""

from .base import OverlapAwareDataLoader

__all__ = [
    "OverlapAwareDataLoader",
]
