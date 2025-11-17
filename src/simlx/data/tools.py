"""Reusable utilities for constructing overlapping views of data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchOverlap:
    """Simple container describing overlap ratios along height and width."""

    height: float
    width: float

    def clamp(self) -> PatchOverlap:
        """Clamp overlap ratios to the inclusive range [0, 1]."""
        return PatchOverlap(
            height=max(0.0, min(1.0, self.height)),
            width=max(0.0, min(1.0, self.width)),
        )


def compute_overlap_region(
    patch_a: tuple[int, int, int, int],
    patch_b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the intersection between two 2D patches.

    Patches are expressed as ``(top, left, bottom, right)``.
    """

    top = max(patch_a[0], patch_b[0])
    left = max(patch_a[1], patch_b[1])
    bottom = min(patch_a[2], patch_b[2])
    right = min(patch_a[3], patch_b[3])
    return top, left, bottom, right
