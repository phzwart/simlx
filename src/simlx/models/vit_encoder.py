"""Vision Transformer encoder for masked representation learning with qlty pretokenizer."""

from __future__ import annotations

import math

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]


class TokenEmbedding(nn.Module):
    """Embedding layer for pre-tokenized tokens from qlty.pretokenizer_2d.

    Takes flattened token vectors (C*patch_size*patch_size) and projects to embed_dim.
    """

    def __init__(
        self,
        token_dim: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.proj = nn.Linear(token_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed pre-tokenized tokens.

        Args:
            tokens: Token tensor of shape (B, T, token_dim) or (T, token_dim).

        Returns:
            Embedded tokens of shape (B, T, embed_dim) or (T, embed_dim).
        """
        return self.proj(tokens)


class CoordinatePositionalEmbedding(nn.Module):
    """Coordinate-based 2D positional embeddings using relative coordinate differences.

    Uses sinusoidal positional encoding based on relative coordinate differences from
    qlty.pretokenizer_2d. Computes relative positions by subtracting a reference point
    (minimum coordinates) to make embeddings invariant to absolute patch position.
    """

    def __init__(
        self,
        embed_dim: int,
        max_rel_coord: float = 128.0,
        temperature: float = 10000.0,
        reference: str = "min",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_rel_coord = max_rel_coord
        self.temperature = temperature
        self.reference = reference  # "min", "mean", or "first"

        # Create position encoding tables
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

        # Generate sinusoidal encodings for relative y and x coordinate differences
        dim_t = torch.arange(embed_dim // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (embed_dim // 2))

        # Cache for coordinate ranges
        self.register_buffer("dim_t", dim_t)

    def _compute_relative_coords(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute relative coordinates from absolute coordinates.

        Args:
            coords: Absolute coordinates of shape (B, T, 2) with (y, x) values.

        Returns:
            Tuple of (rel_y, rel_x) tensors of shape (B, T, 1).
        """
        B, T, _ = coords.shape

        if self.reference == "min":
            # Relative to minimum coordinates in each batch
            y_min = coords[:, :, 0:1].min(dim=1, keepdim=True)[0]  # (B, 1, 1)
            x_min = coords[:, :, 1:2].min(dim=1, keepdim=True)[0]  # (B, 1, 1)
            rel_y = coords[:, :, 0:1] - y_min  # (B, T, 1)
            rel_x = coords[:, :, 1:2] - x_min  # (B, T, 1)
        elif self.reference == "mean":
            # Relative to mean coordinates in each batch
            y_mean = coords[:, :, 0:1].mean(dim=1, keepdim=True)  # (B, 1, 1)
            x_mean = coords[:, :, 1:2].mean(dim=1, keepdim=True)  # (B, 1, 1)
            rel_y = coords[:, :, 0:1] - y_mean  # (B, T, 1)
            rel_x = coords[:, :, 1:2] - x_mean  # (B, T, 1)
        elif self.reference == "first":
            # Relative to first token coordinates
            rel_y = coords[:, :, 0:1] - coords[:, 0:1, 0:1]  # (B, T, 1)
            rel_x = coords[:, :, 1:2] - coords[:, 0:1, 1:2]  # (B, T, 1)
        else:
            raise ValueError(f"Unsupported reference: {self.reference}")

        return rel_y, rel_x

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Generate positional embeddings from relative coordinate differences.

        Args:
            coords: Absolute coordinate tensor of shape (B, T, 2) or (T, 2) with (y, x) values.

        Returns:
            Positional embeddings of shape (B, T, embed_dim) or (T, embed_dim).
        """
        is_batched = coords.dim() == 3
        if not is_batched:
            coords = coords.unsqueeze(0)

        B, T, _ = coords.shape
        device = coords.device
        dtype = coords.dtype

        # Compute relative coordinates
        rel_y, rel_x = self._compute_relative_coords(coords)  # (B, T, 1) each

        # Sinusoidal encoding for relative y
        # dim_t: (embed_dim//2,), expand to (1, 1, embed_dim//2) for broadcasting
        dim_t_expanded = self.dim_t.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim//2)
        y_pos = rel_y / dim_t_expanded  # (B, T, embed_dim//2)
        y_emb = torch.zeros(B, T, self.embed_dim // 2, device=device, dtype=dtype)
        y_emb[:, :, 0::2] = torch.sin(y_pos[:, :, 0::2])
        y_emb[:, :, 1::2] = torch.cos(y_pos[:, :, 1::2])

        # Sinusoidal encoding for relative x
        x_pos = rel_x / dim_t_expanded  # (B, T, embed_dim//2)
        x_emb = torch.zeros(B, T, self.embed_dim // 2, device=device, dtype=dtype)
        x_emb[:, :, 0::2] = torch.sin(x_pos[:, :, 0::2])
        x_emb[:, :, 1::2] = torch.cos(x_pos[:, :, 1::2])

        # Concatenate y and x embeddings
        pos_emb = torch.cat([y_emb, x_emb], dim=-1)  # (B, T, embed_dim)

        if not is_batched:
            pos_emb = pos_emb.squeeze(0)

        return pos_emb


class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention and MLP, pre-norm architecture."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tokens of shape (B, N, embed_dim).
            mask: Optional attention mask of shape (B, N) or (B, N, N).

        Returns:
            Output tokens of shape (B, N, embed_dim).
        """
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # Convert boolean mask to attention mask
                attn_mask = mask.float()
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
                attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            elif mask.dim() == 3:
                attn_mask = mask
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for masked representation learning.

    Works with pre-tokenized inputs from qlty.pretokenizer_2d. Accepts tokens and
    coordinates instead of raw images.

    Example:
        >>> from qlty.patchops import tokenize_patch
        >>> encoder = ViTEncoder(
        ...     token_dim=768,  # C*patch_size*patch_size
        ...     embed_dim=768,
        ...     depth=12,
        ...     num_heads=12,
        ...     mlp_ratio=4.0,
        ...     dropout=0.1,
        ... )
        >>> patch = torch.randn(3, 64, 64)
        >>> tokens, coords = tokenize_patch(patch, patch_size=16, stride=8)
        >>> # tokens: (T, 768), coords: (T, 2)
        >>> tokens_batch = tokens.unsqueeze(0)  # (1, T, 768)
        >>> coords_batch = coords.unsqueeze(0)  # (1, T, 2)
        >>> encoded = encoder(tokens_batch, coords_batch)
    """

    def __init__(
        self,
        token_dim: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_rel_coord: float = 128.0,
        pos_reference: str = "min",
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.token_embed = TokenEmbedding(token_dim, embed_dim)
        self.pos_embed = CoordinatePositionalEmbedding(
            embed_dim, max_rel_coord=max_rel_coord, reference=pos_reference
        )
        self.dropout_layer = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
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
        """Forward pass through ViT encoder.

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

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)

        return x
