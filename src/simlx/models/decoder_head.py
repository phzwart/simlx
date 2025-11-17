"""Lightweight reconstruction head for masked prediction.

Works with encoded tokens from ViTEncoder or MambaViTEncoder that process
qlty.pretokenizer_2d outputs. Reconstructs original token dimensions
(C*patch_size*patch_size) from encoded representations.
"""

from __future__ import annotations

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]


class DecoderHead(nn.Module):
    """Reconstruction head for masked prediction.

    Takes encoded visible tokens (from ViTEncoder or MambaViTEncoder) and
    predicts reconstructions for all tokens (including masked ones). Outputs
    match the original token dimensions from qlty.pretokenizer_2d.

    Example:
        >>> from qlty.patchops import tokenize_patch, build_sequence_pair
        >>> from simlx.models.vit_encoder import ViTEncoder
        >>>
        >>> # MLP mode
        >>> encoder = ViTEncoder(token_dim=768, embed_dim=768)
        >>> decoder = DecoderHead(
        ...     embed_dim=768,
        ...     token_dim=768,  # C*patch_size*patch_size from qlty
        ...     mode="mlp",
        ...     hidden_dim=512,
        ... )
        >>> patch = torch.randn(3, 64, 64)
        >>> tokens, coords = tokenize_patch(patch, patch_size=16, stride=8)
        >>> tokens_batch = tokens.unsqueeze(0)  # (1, T, 768)
        >>> coords_batch = coords.unsqueeze(0)  # (1, T, 2)
        >>> encoded = encoder(tokens_batch, coords_batch)  # (1, T, 768)
        >>> # Use overlap_mask from build_sequence_pair for masking
        >>> reconstructions = decoder(encoded, mask=overlap_mask)

        >>> # Transformer mode
        >>> decoder = DecoderHead(
        ...     embed_dim=768,
        ...     token_dim=768,
        ...     mode="transformer",
        ...     decoder_depth=2,
        ...     num_heads=8,
        ... )
        >>> reconstructions = decoder(encoded, mask=overlap_mask)
    """

    def __init__(
        self,
        embed_dim: int,
        token_dim: int,
        mode: str = "mlp",
        hidden_dim: int | None = None,
        decoder_depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim  # type: ignore[assignment]
        self.token_dim = token_dim  # type: ignore[assignment]  # Original token dimension from qlty (C*patch_size*patch_size)
        self.mode = mode.lower()

        if self.mode == "mlp":
            hidden_dim = hidden_dim or embed_dim
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, token_dim),
            )
        elif self.mode == "transformer":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_blocks = nn.ModuleList([
                TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(decoder_depth)
            ])
            self.decoder_norm = nn.LayerNorm(embed_dim)
            self.decoder = nn.Linear(embed_dim, token_dim)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'mlp' or 'transformer'.")

    def forward(  # noqa: C901
        self,
        visible_tokens: torch.Tensor,
        num_total_tokens: int | None = None,
        mask: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict reconstructions for all tokens.

        Args:
            visible_tokens: Encoded tokens from encoder of shape (B, T, embed_dim).
                          Can be from ViTEncoder or MambaViTEncoder processing qlty tokens.
            num_total_tokens: Total number of tokens (visible + masked). Required if mask
                            is provided, otherwise inferred from visible_tokens.
            mask: Optional boolean mask of shape (B, T) where True indicates visible.
                 Can use overlap_mask from qlty.build_sequence_pair. If provided, used
                 to reconstruct full sequence with correct token positions.
            sequence_lengths: Optional tensor of shape (B,) with actual sequence lengths
                            for variable-length sequences from qlty.

        Returns:
            Reconstructions of shape (B, T, token_dim) matching original qlty token dimensions.
        """
        B, N_visible, D = visible_tokens.shape

        if mask is not None:
            num_total_tokens = mask.shape[1]
        elif num_total_tokens is None:
            num_total_tokens = N_visible

        if self.mode == "mlp":
            if mask is not None:
                # Reconstruct full sequence: place visible tokens, zero-fill masked
                all_tokens = torch.zeros(B, num_total_tokens, D, device=visible_tokens.device)
                visible_idx = 0
                for b in range(B):
                    for n in range(num_total_tokens):
                        if mask[b, n]:
                            all_tokens[b, n] = visible_tokens[b, visible_idx]
                            visible_idx += 1
                reconstructions = self.decoder(all_tokens)
            else:
                # All tokens visible
                reconstructions = self.decoder(visible_tokens)
        else:  # transformer mode
            if mask is not None:
                # Create full sequence with mask tokens
                all_tokens = torch.zeros(B, num_total_tokens, D, device=visible_tokens.device)
                mask_tokens = self.mask_token.expand(B, num_total_tokens, -1)
                visible_idx = 0
                for b in range(B):
                    for n in range(num_total_tokens):
                        if mask[b, n]:
                            all_tokens[b, n] = visible_tokens[b, visible_idx]
                            visible_idx += 1
                        else:
                            all_tokens[b, n] = mask_tokens[b, n]
            else:
                # All tokens visible, no mask tokens needed
                all_tokens = visible_tokens

            # Apply transformer decoder blocks
            for block in self.decoder_blocks:
                all_tokens = block(all_tokens)

            all_tokens = self.decoder_norm(all_tokens)
            reconstructions = self.decoder(all_tokens)

        return reconstructions


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and MLP, pre-norm architecture."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Input tokens of shape (B, N, embed_dim).

        Returns:
            Output tokens of shape (B, N, embed_dim).
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x
