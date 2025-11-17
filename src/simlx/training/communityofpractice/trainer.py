"""Peer-learning trainer that coordinates a swarm of models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ...models.base import BaseModel


@dataclass
class PeerGroup:
    """Group of models with designated teacher and students."""

    teacher: BaseModel
    students: Sequence[BaseModel]


class CommunityOfPracticeTrainer:
    """Coordinate peer learning across partially overlapping patches."""

    def __init__(self, peers: Iterable[PeerGroup]) -> None:
        self.peers = list(peers)

    def step(self, batch) -> None:  # pragma: no cover - to be implemented
        """Run a single optimization step for the peer group."""
        raise NotImplementedError
