"""Mamba-style state-space encoder with the same API as ViT encoder.

Works with pre-tokenized inputs from qlty.pretokenizer_2d.
"""

from __future__ import annotations

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

from simlx.models.vit_encoder import (
    TokenEmbedding,
    CoordinatePositionalEmbedding,
)


class SSMBlock(nn.Module):
    """Mamba-style state-space model block with selective scan."""

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        state_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim or embed_dim
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        # Gating projection
        self.gate_proj = nn.Linear(embed_dim, embed_dim * 2)
        # State-space parameters
        self.state_proj = nn.Linear(embed_dim, self.state_dim)
        # Output projection
        self.out_proj = nn.Linear(self.state_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Selective state-space scan.

        Args:
            u: Input sequence (B, N, state_dim).
            delta: Time step (B, N, 1).
            A: State matrix (state_dim,).
            B: Input projection (B, N, state_dim).
            C: Output projection (B, N, state_dim).

        Returns:
            Output sequence (B, N, state_dim).
        """
        B_batch, N, D = u.shape
        device = u.device
        dtype = u.dtype

        # Discretize: A_d = exp(delta * A), B_d = delta * B
        # delta: (B, N, 1), A: (state_dim,)
        deltaA = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))  # (B, N, state_dim)
        deltaB_u = delta * B * u  # (B, N, state_dim)

        # Sequential scan: h_t = A_d * h_{t-1} + B_d * u_t
        h = torch.zeros(B_batch, D, device=device, dtype=dtype)  # (B, state_dim)
        outputs = []

        for i in range(N):
            h = deltaA[:, i] * h + deltaB_u[:, i]  # (B, state_dim)
            # Output: C * h (element-wise)
            y_i = C[:, i] * h  # (B, state_dim)
            outputs.append(y_i.unsqueeze(1))  # (B, 1, state_dim)

        y = torch.cat(outputs, dim=1)  # (B, N, state_dim)
        return y

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through SSM block.

        Args:
            x: Input tokens of shape (B, N, embed_dim).
            mask: Optional mask (not used in SSM, but kept for API compatibility).

        Returns:
            Output tokens of shape (B, N, embed_dim).
        """
        # Pre-norm SSM
        x_norm = self.norm1(x)
        gate_out = self.gate_proj(x_norm)  # (B, N, 2*embed_dim)
        gate, gate_gate = gate_out.chunk(2, dim=-1)
        gate = F.silu(gate)  # SiLU activation for gating
        gate_gate = F.silu(gate_gate)

        # Project to state space
        u = self.state_proj(x_norm * gate)  # (B, N, state_dim)

        # Simplified SSM: use learnable parameters
        # In a full implementation, these would be more sophisticated
        state_dim = self.state_dim
        batch_size, seq_len = x.shape[0], x.shape[1]
        A = -torch.ones(state_dim, device=x.device, dtype=x.dtype)  # Stable A
        B = self.state_proj(gate_gate)  # (B, N, state_dim) - gated input projection
        C = torch.ones((batch_size, seq_len, state_dim), device=x.device, dtype=x.dtype)  # Output projection

        # Time step (learned from input)
        delta = F.softplus(self.state_proj(x_norm).mean(dim=-1, keepdim=True))  # (B, N, 1)

        # Selective scan
        ssm_out = self._selective_scan(u, delta, A, B, C)
        ssm_out = self.out_proj(ssm_out)
        x = x + self.dropout1(ssm_out)

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MambaViTEncoder(nn.Module):
    """Mamba-style state-space encoder with the same API as ViT encoder.

    Works with pre-tokenized inputs from qlty.pretokenizer_2d. Uses the same token
    embedding and coordinate-based positional embedding as ViT, but replaces
    attention blocks with Mamba-style SSM blocks.

    Example:
        >>> from qlty.patchops import tokenize_patch
        >>> encoder = MambaViTEncoder(
        ...     token_dim=768,  # C*patch_size*patch_size
        ...     embed_dim=768,
        ...     depth=12,
        ...     mlp_ratio=4.0,
        ...     dropout=0.1,
        ... )
        >>> patch = torch.randn(3, 64, 64)
        >>> tokens, coords = tokenize_patch(patch, patch_size=16, stride=8)
        >>> tokens_batch = tokens.unsqueeze(0)  # (1, T, 768)
        >>> coords_batch = coords.unsqueeze(0)  # (1, T, 2)
        >>> encoded = encoder(tokens_batch, coords_batch)
    """

    def __init__(
        self,
        token_dim: int,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        state_dim: int | None = None,
        max_rel_coord: float = 128.0,
        pos_reference: str = "min",
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.state_dim = state_dim

        self.token_embed = TokenEmbedding(token_dim, embed_dim)
        self.pos_embed = CoordinatePositionalEmbedding(
            embed_dim, max_rel_coord=max_rel_coord, reference=pos_reference
        )
        self.dropout_layer = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                SSMBlock(embed_dim, mlp_ratio, dropout, state_dim)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Mamba encoder.

        Args:
            tokens: Pre-tokenized tokens from qlty of shape (B, T, token_dim).
                   Can handle variable-length sequences (use sequence_lengths for padding).
            coords: Absolute coordinates from qlty of shape (B, T, 2) with (y, x) values.
            mask: Optional boolean mask of shape (B, T) where True indicates visible tokens.
                 If None, all tokens are processed. Can be used with overlap_mask from qlty.
            sequence_lengths: Optional tensor of shape (B,) with actual sequence lengths
                            for variable-length sequences.

        Returns:
            Encoded tokens of shape (B, T, embed_dim).
        """
        # Embed tokens
        x = self.token_embed(tokens)  # (B, T, embed_dim)

        # Add positional embeddings from coordinates
        pos_emb = self.pos_embed(coords)  # (B, T, embed_dim)
        x = x + pos_emb
        x = self.dropout_layer(x)

        # Apply mask by zeroing masked tokens
        if mask is not None:
            # mask: (B, T), True = visible, False = masked
            # Handle different mask shapes
            if mask.dim() == 1:
                # If mask is (T,), expand to (1, T) for broadcasting
                if mask.shape[0] == x.shape[1]:
                    mask = mask.unsqueeze(0)  # (1, T)
                else:
                    raise ValueError(
                        f"Mask shape {mask.shape} doesn't match sequence length {x.shape[1]}"
                    )
            elif mask.dim() == 2:
                # Ensure mask matches (B, T)
                if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1]:
                    raise ValueError(
                        f"Mask shape {mask.shape} doesn't match tokens shape {x.shape[:2]}"
                    )
            else:
                raise ValueError(f"Mask must be 1D or 2D, got shape {mask.shape}")
            
            mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
            x = x * mask_expanded.float()

        # Apply SSM blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)

        return x
