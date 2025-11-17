"""Barlow Twins trainer for self-supervised learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simlx.models.barlow_twins import BarlowTwins, BarlowTwinsConfig
from simlx.tracking.base import ExperimentTracker, NoOpTracker
from simlx.utils.logging import configure_logging


@dataclass
class BarlowTwinsTrainerConfig:
    """Configuration for :class:`BarlowTwinsTrainer`."""

    # Model config
    encoder_name: str = "resnet18"
    encoder_output_dim: int = 512
    projection_dim: int = 128
    projection_hidden_dim: int = 2048

    # Training config
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    optimizer: str = "adam"
    lr_scheduler: str | None = None
    lr_scheduler_kwargs: dict[str, Any] | None = None
    num_epochs: int = 100
    batch_size: int = 256
    device: str = "cuda"

    # Barlow Twins specific
    lambda_param: float = 5e-3  # Weight for invariance term
    scale_loss: float = 1.0 / 128  # Scale factor for loss (1/projection_dim)

    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 100
    checkpoint_dir: Path | str | None = None

    # Early stopping (based on plateau, not L2 norm)
    early_stopping_patience: int | None = None  # None = disabled
    early_stopping_min_delta: float = 0.0  # Minimum change to qualify as improvement
    early_stopping_mode: str = "min"  # "min" for loss, "max" for metrics
    early_stopping_metric: str = "loss"  # Metric to monitor for plateau

    # Tracking
    tracker: ExperimentTracker | None = None


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss function.

    The loss encourages:
    1. Invariance: diagonal elements of cross-correlation matrix close to 1
    2. Redundancy reduction: off-diagonal elements close to 0
    """

    def __init__(self, lambda_param: float = 5e-3, scale_loss: float = 1.0 / 128) -> None:
        super().__init__()
        self.lambda_param = lambda_param
        self.scale_loss = scale_loss

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute Barlow Twins loss.

        Args:
            z_a: Projections from first augmented view [batch_size, projection_dim]
            z_b: Projections from second augmented view [batch_size, projection_dim]

        Returns:
            Scalar loss value
        """
        batch_size = z_a.size(0)
        projection_dim = z_a.size(1)

        # Normalize projections
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-8)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-8)

        # Compute cross-correlation matrix
        # C = (1/batch_size) * z_a_norm^T @ z_b_norm
        c = torch.mm(z_a_norm.t(), z_b_norm) / batch_size

        # Invariance term: diagonal elements should be close to 1
        # Sum of (1 - C_ii)^2
        invariance_loss = torch.sum((torch.diag(c) - 1.0) ** 2)

        # Redundancy reduction term: off-diagonal elements should be close to 0
        # Sum of C_ij^2 for i != j
        mask = ~torch.eye(projection_dim, dtype=torch.bool, device=c.device)
        redundancy_loss = torch.sum(c[mask] ** 2)

        # Total loss
        loss = invariance_loss + self.lambda_param * redundancy_loss
        return loss * self.scale_loss


class BarlowTwinsTrainer:
    """Trainer for Barlow Twins self-supervised learning."""

    def __init__(self, config: BarlowTwinsTrainerConfig) -> None:
        self.config = config
        self.logger = configure_logging(name="BarlowTwinsTrainer")
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.tracker = config.tracker or NoOpTracker()

        # Create model
        model_config = BarlowTwinsConfig(
            encoder_name=config.encoder_name,
            encoder_output_dim=config.encoder_output_dim,
            projection_dim=config.projection_dim,
            projection_hidden_dim=config.projection_hidden_dim,
        )
        self.model = BarlowTwins(config=model_config).to(self.device)

        # Create loss function
        self.criterion = BarlowTwinsLoss(
            lambda_param=config.lambda_param,
            scale_loss=config.scale_loss,
        )

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Early stopping state
        self.best_metric_value: float | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

        # Checkpoint directory
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.optimizer.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler based on config."""
        if self.config.lr_scheduler is None:
            return None

        scheduler_name = self.config.lr_scheduler.lower()
        scheduler_kwargs = self.config.lr_scheduler_kwargs or {}

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                **scheduler_kwargs,
            )
        if scheduler_name == "step":
            step_size = scheduler_kwargs.pop("step_size", 30)
            gamma = scheduler_kwargs.pop("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
                **scheduler_kwargs,
            )
        if scheduler_name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_kwargs,
            )

        raise ValueError(f"Unsupported scheduler: {self.config.lr_scheduler}")

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Tuple of (augmented_view_1, augmented_view_2) tensors

        Returns:
            Dictionary of metrics for this step
        """
        x_a, x_b = batch
        x_a = x_a.to(self.device)
        x_b = x_b.to(self.device)

        # Forward pass
        z_a = self.model(x_a)
        z_b = self.model(x_b)

        # Compute loss
        loss = self.criterion(z_a, z_b)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute additional metrics
        with torch.no_grad():
            # Normalize for correlation computation
            z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-8)
            z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-8)
            c = torch.mm(z_a_norm.t(), z_b_norm) / z_a.size(0)

            # Mean diagonal (invariance)
            mean_diag = torch.diag(c).mean().item()

            # Mean off-diagonal (redundancy)
            mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)
            mean_off_diag = c[mask].abs().mean().item()

        metrics = {
            "loss": loss.item(),
            "mean_diagonal": mean_diag,
            "mean_off_diagonal": mean_off_diag,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        self.global_step += 1
        return metrics

    def train_epoch(self, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding (augmented_view_1, augmented_view_2) batches

        Returns:
            Dictionary of epoch-averaged metrics
        """
        self.model.train()
        epoch_metrics: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(dataloader):
            step_metrics = self.train_step(batch)

            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Logging
            if self.global_step % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch:03d} | Step {self.global_step:05d} | "
                    f"Loss: {step_metrics['loss']:.4f} | "
                    f"Diag: {step_metrics['mean_diagonal']:.4f} | "
                    f"Off-Diag: {step_metrics['mean_off_diagonal']:.4f}"
                )

                # Log to tracker
                self.tracker.log_metrics(step_metrics, step=self.global_step)

            # Checkpointing
            if (
                self.checkpoint_dir
                and self.config.checkpoint_interval > 0
                and self.global_step % self.config.checkpoint_interval == 0
            ):
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        # Average metrics over epoch
        avg_metrics = {key: sum(values) / len(values) for key, values in epoch_metrics.items()}
        return avg_metrics

    def fit(self, train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Train the model for the configured number of epochs.

        Args:
            train_dataloader: DataLoader for training data
        """
        self.tracker.start_run(run_name=f"barlow_twins_{self.config.encoder_name}")

        # Log hyperparameters
        config_dict = {
            "encoder_name": self.config.encoder_name,
            "encoder_output_dim": self.config.encoder_output_dim,
            "projection_dim": self.config.projection_dim,
            "projection_hidden_dim": self.config.projection_hidden_dim,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "optimizer": self.config.optimizer,
            "lr_scheduler": self.config.lr_scheduler or "none",
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "lambda_param": self.config.lambda_param,
            "scale_loss": self.config.scale_loss,
        }
        self.tracker.log_params(config_dict)

        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self.train_epoch(train_dataloader)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_metrics["loss"])
                else:
                    self.scheduler.step()

            # Check for early stopping based on plateau (not L2 norm)
            if self.config.early_stopping_patience is not None:
                self._check_early_stopping(epoch_metrics)

            # Log epoch metrics
            self.logger.info(
                f"Epoch {epoch:03d} completed | "
                f"Avg Loss: {epoch_metrics['loss']:.4f} | "
                f"Avg Diag: {epoch_metrics['mean_diagonal']:.4f} | "
                f"Avg Off-Diag: {epoch_metrics['mean_off_diagonal']:.4f}"
            )
            if self.config.early_stopping_patience is not None:
                self.logger.info(
                    f"  Early stopping: {self.epochs_without_improvement}/"
                    f"{self.config.early_stopping_patience} epochs without improvement"
                )

            # Log epoch metrics to tracker
            epoch_metrics_with_prefix = {f"epoch_{k}": v for k, v in epoch_metrics.items()}
            self.tracker.log_metrics(epoch_metrics_with_prefix, step=epoch)

            # Save checkpoint at end of epoch
            if self.checkpoint_dir:
                self.save_checkpoint(f"checkpoint_epoch_{epoch:03d}.pt")

            # Stop if early stopping triggered
            if self.should_stop:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Best {self.config.early_stopping_metric}: {self.best_metric_value:.6f}"
                )
                break

        self.tracker.end_run()
        self.logger.info("Training completed")

    def _check_early_stopping(self, epoch_metrics: dict[str, float]) -> None:
        """Check if training should stop based on plateau detection.
        
        Uses plateau detection (no improvement) rather than absolute L2 norm,
        since latent spaces will be aligned in a combined encoder afterwards.
        
        Args:
            epoch_metrics: Dictionary of metrics from the current epoch
        """
        if self.config.early_stopping_patience is None:
            return

        metric_name = self.config.early_stopping_metric
        if metric_name not in epoch_metrics:
            self.logger.warning(
                f"Early stopping metric '{metric_name}' not found in epoch_metrics. "
                f"Available: {list(epoch_metrics.keys())}"
            )
            return

        current_value = epoch_metrics[metric_name]
        is_better = False

        if self.best_metric_value is None:
            # First epoch - always better
            is_better = True
        elif self.config.early_stopping_mode == "min":
            # For loss: lower is better
            is_better = current_value < (self.best_metric_value - self.config.early_stopping_min_delta)
        elif self.config.early_stopping_mode == "max":
            # For metrics: higher is better
            is_better = current_value > (self.best_metric_value + self.config.early_stopping_min_delta)
        else:
            raise ValueError(
                f"Invalid early_stopping_mode: {self.config.early_stopping_mode}. "
                f"Must be 'min' or 'max'"
            )

        if is_better:
            self.best_metric_value = current_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.config.early_stopping_patience:
            self.should_stop = True

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        if self.checkpoint_dir is None:
            return

        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Log checkpoint to tracker
        self.tracker.log_model(checkpoint_path, f"checkpoints/{filename}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

