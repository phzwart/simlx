"""Training strategies for simlx."""

from .barlow_twins import BarlowTwinsTrainer, BarlowTwinsTrainerConfig
from .communityofpractice import CommunityOfPracticeTrainer

__all__ = [
    "BarlowTwinsTrainer",
    "BarlowTwinsTrainerConfig",
    "CommunityOfPracticeTrainer",
]
