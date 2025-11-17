"""Example script for training Barlow Twins model.

This script demonstrates how to use the Barlow Twins trainer with a simple dataset.
You'll need to provide your own data augmentation pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from simlx.training import BarlowTwinsTrainer, BarlowTwinsTrainerConfig
from simlx.tracking import create_tracker


class DummyAugmentedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dummy dataset that returns two augmented views of random images.

    Replace this with your actual dataset that applies augmentations.
    """

    def __init__(self, num_samples: int = 1000, image_size: tuple[int, int] = (224, 224)) -> None:
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate random image (replace with actual data loading)
        image = torch.randn(3, *self.image_size)

        # Apply augmentations (replace with actual augmentation pipeline)
        # For Barlow Twins, you typically use:
        # - Random cropping and resizing
        # - Random horizontal flipping
        # - Color jittering
        # - Random grayscale conversion
        # - Gaussian blur
        # - Solarization (optional)

        # For this example, we'll just add some noise as a placeholder
        aug1 = image + torch.randn_like(image) * 0.1
        aug2 = image + torch.randn_like(image) * 0.1

        return aug1, aug2


def main() -> None:
    """Main training function."""
    # Create dataset
    dataset = DummyAugmentedDataset(num_samples=10000, image_size=(224, 224))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create experiment tracker (optional)
    tracker = create_tracker("mlflow", experiment_name="barlow_twins_experiments")

    # Configure trainer
    config = BarlowTwinsTrainerConfig(
        encoder_name="resnet18",
        encoder_output_dim=512,
        projection_dim=128,
        projection_hidden_dim=2048,
        learning_rate=1e-3,
        weight_decay=1e-6,
        optimizer="adam",
        lr_scheduler="cosine",
        num_epochs=100,
        batch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lambda_param=5e-3,
        scale_loss=1.0 / 128,
        log_interval=10,
        checkpoint_interval=100,
        checkpoint_dir="./checkpoints",
        tracker=tracker,
    )

    # Create and run trainer
    trainer = BarlowTwinsTrainer(config)
    trainer.fit(dataloader)

    print("Training completed!")


if __name__ == "__main__":
    main()

