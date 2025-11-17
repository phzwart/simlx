"""Utilities for working with projection heads in Matryoshka U-Net.

Example:
    >>> projections = compute_svd_projections(model, train_loader, torch.device("cuda"))
    >>> replace_random_projections(model, projections)
    >>> metrics = analyze_projection_quality(model, val_loader)
    >>> print(metrics["bottleneck"]["variance_explained"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from simlx.models.matryoshka_unet import MatryoshkaUNet
from simlx.models.random_projection import RandomGaussianProjection


def compute_svd_projections(
    model: MatryoshkaUNet,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute SVD-based projection matrices for random projection heads.

    Args:
        model: Matryoshka U-Net configured with random projection heads.
        dataloader: Iterable supplying input tensors or batches containing tensors.
        device: Device on which to run the forward passes while collecting features.
        max_batches: Optional limit on the number of batches to process.

    Returns:
        Mapping from scale name to projection matrix shaped
        ``[out_features, in_features]``.
    """

    feature_sets = _collect_head_inputs(model, dataloader, device, max_batches)
    projections: Dict[str, torch.Tensor] = {}

    for scale_name, features in feature_sets.items():
        module = model.scale_heads[scale_name].head
        if not isinstance(module, RandomGaussianProjection):
            continue
        centered = features - features.mean(dim=0, keepdim=True)
        available_samples = centered.shape[0]
        out_features = module.out_features
        if available_samples < out_features:
            raise RuntimeError(
                f"Insufficient samples ({available_samples}) to compute SVD for {scale_name}. "
                f"Need at least {out_features}."
            )
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        projections[scale_name] = vh[:out_features, :].to(device)

    return projections


def replace_random_projections(
    model: MatryoshkaUNet,
    projections: Dict[str, torch.Tensor],
) -> None:
    """Replace random projection heads with fixed projection matrices.

    Args:
        model: Target Matryoshka U-Net instance.
        projections: Mapping produced by :func:`compute_svd_projections`.
    """

    for scale_name, head in model.scale_heads.items():
        module = head.head
        if isinstance(module, RandomGaussianProjection):
            if scale_name not in projections:
                raise KeyError(f"No projection provided for scale '{scale_name}'.")
            module.set_projection(projections[scale_name])
            module.eval()
    model.eval()


def analyze_projection_quality(
    model: MatryoshkaUNet,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict[str, float]]:
    """Analyze projection quality metrics for each random projection head.

    Args:
        model: Matryoshka U-Net instance to evaluate.
        dataloader: Iterable providing evaluation batches.
        device: Device on which to collect features.

    Returns:
        Nested mapping ``scale -> metric -> value`` including variance explained,
        reconstruction error, and effective rank.
    """

    feature_sets = _collect_head_inputs(model, dataloader, device, None)
    metrics: Dict[str, Dict[str, float]] = {}

    for scale_name, features in feature_sets.items():
        module = model.scale_heads[scale_name].head
        if not isinstance(module, RandomGaussianProjection):
            continue
        weight = module.weight
        if weight is None:
            continue
        weight = weight.detach().to(features.device, features.dtype)

        centered = features - features.mean(dim=0, keepdim=True)
        total_variance = torch.sum(centered**2).item()

        projected = centered @ weight.t()
        pinv = torch.linalg.pinv(weight)
        reconstructed = projected @ pinv.t()
        residual = centered - reconstructed
        residual_norm = torch.sum(residual**2).item()

        if total_variance > 0:
            variance_explained = 1.0 - residual_norm / total_variance
        else:
            variance_explained = 0.0

        reconstruction_error = residual_norm / centered.shape[0] if centered.shape[0] > 0 else 0.0

        singular_values = torch.linalg.svdvals(projected)
        if torch.count_nonzero(singular_values).item() == 0:
            effective_rank = 0.0
        else:
            probs = singular_values / singular_values.sum()
            entropy = -(probs * torch.log(probs + 1e-12)).sum()
            effective_rank = torch.exp(entropy).item()

        metrics[scale_name] = {
            "variance_explained": float(max(0.0, min(1.0, variance_explained))),
            "reconstruction_error": float(reconstruction_error),
            "effective_rank": float(effective_rank),
        }

    return metrics


def _extract_batch_input(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("image", "inputs", "input", "data", "x"):
            if key in batch:
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    return value
        raise KeyError(
            "Could not find tensor input in batch dictionary. Expected one of keys: "
            "'image', 'inputs', 'input', 'data', or 'x'."
        )
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Received empty batch sequence.")
        first = batch[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(
        "Unsupported batch format. Provide a tensor, dict with tensor values, "
        "or a tuple/list whose first element is a tensor."
    )


def _collect_head_inputs(
    model: MatryoshkaUNet,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> Dict[str, torch.Tensor]:
    random_heads: Dict[str, RandomGaussianProjection] = {
        scale_name: head.head
        for scale_name, head in model.scale_heads.items()
        if isinstance(head.head, RandomGaussianProjection)
    }
    if not random_heads:
        raise RuntimeError("Model does not contain any RandomGaussianProjection heads.")

    features: Dict[str, List[torch.Tensor]] = {name: [] for name in random_heads}
    hooks: List[RemovableHandle] = []
    random_states: List[tuple[RandomGaussianProjection, bool]] = []
    was_training = model.training
    original_device = _get_model_device(model)

    model.eval()
    if original_device is None or original_device != device:
        model.to(device)

    for scale_name, module in random_heads.items():
        random_states.append((module, module.training))
        module.train(True)

        def _hook(
            mod: RandomGaussianProjection,
            inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
            *,
            target_scale: str = scale_name,
        ) -> None:
            if not inputs:
                return
            tensor = inputs[0].detach().to("cpu", copy=False)
            flattened = tensor.movedim(1, -1).reshape(-1, tensor.shape[1]).to(torch.float32)
            features[target_scale].append(flattened)

        hooks.append(module.register_forward_hook(_hook))  # type: ignore[arg-type]

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = _extract_batch_input(batch).to(device)
                model(inputs)
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
    finally:
        for hook in hooks:
            hook.remove()
        for module, state in random_states:
            module.train(state)
        if was_training:
            model.train()
        if original_device is not None and original_device != device:
            model.to(original_device)

    aggregated: Dict[str, torch.Tensor] = {}
    for scale_name, tensors in features.items():
        if not tensors:
            raise RuntimeError(f"No features collected for scale '{scale_name}'.")
        aggregated[scale_name] = torch.cat(tensors, dim=0)
    return aggregated


def _get_model_device(model: torch.nn.Module) -> Optional[torch.device]:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return None

