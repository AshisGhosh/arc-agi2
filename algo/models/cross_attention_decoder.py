import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function: x * silu(gate(x))"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)  # Project back to original dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(x) * F.silu(self.gate(x)))


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for patch-based model.

    Test tokens (queries) attend to support tokens (keys/values).
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        # Query projection (for test tokens)
        self.q_proj = nn.Linear(model_dim, model_dim)

        # Key/Value projections (for support tokens)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

        # Output projection
        self.out_proj = nn.Linear(model_dim, model_dim)

        # Normalization
        self.norm1 = nn.RMSNorm(model_dim)
        self.norm2 = nn.RMSNorm(model_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = SwiGLU(model_dim, int(model_dim * 2.5))  # ~256 for d=128

    def forward(
        self, test_tokens: torch.Tensor, support_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.

        Args:
            test_tokens: [B, num_test_patches, model_dim] - query tokens
            support_tokens: [B, num_support_patches, model_dim] - key/value tokens

        Returns:
            updated_test_tokens: [B, num_test_patches, model_dim]
        """
        B, num_test, d = test_tokens.shape
        _, num_support, _ = support_tokens.shape

        # Pre-norm
        test_tokens_norm = self.norm1(test_tokens)

        # Project to Q, K, V
        Q = self.q_proj(test_tokens_norm)  # [B, num_test, d]
        K = self.k_proj(support_tokens)  # [B, num_support, d]
        V = self.v_proj(support_tokens)  # [B, num_support, d]

        # Reshape for multi-head attention
        Q = Q.view(B, num_test, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, num_test, head_dim]
        K = K.view(B, num_support, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, num_support, head_dim]
        V = V.view(B, num_support, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, num_support, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # [B, num_heads, num_test, num_support]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(
            attn_weights, V
        )  # [B, num_heads, num_test, head_dim]

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, num_test, d)
        )  # [B, num_test, d]

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection
        test_tokens = test_tokens + attn_output

        # FFN with residual
        test_tokens_norm = self.norm2(test_tokens)
        ffn_output = self.ffn(test_tokens_norm)
        test_tokens = test_tokens + ffn_output

        return test_tokens


class CrossAttentionDecoder(nn.Module):
    """
    Cross-attention decoder for patch-based model.

    Multiple layers of cross-attention between test tokens and support tokens.
    """

    def __init__(
        self, model_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers

        # Cross-attention layers
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(model_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, test_tokens: torch.Tensor, support_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention decoder.

        Args:
            test_tokens: [B, num_test_patches, model_dim] - query tokens
            support_tokens: [B, num_support_patches, model_dim] - key/value tokens

        Returns:
            updated_test_tokens: [B, num_test_patches, model_dim]
        """
        for layer in self.layers:
            test_tokens = layer(test_tokens, support_tokens)

        return test_tokens


class PatchOutputHead(nn.Module):
    """
    Output head for patch-based model using PixelShuffle upsampling.

    Uses PixelShuffle for high-quality upsampling from patch-level to pixel-level.
    This approach provides better spatial awareness and shape generation compared
    to nearest neighbor interpolation or simple ConvTranspose2d.
    """

    def __init__(self, model_dim: int, num_colors: int, patch_size: int = 3):
        super().__init__()
        self.model_dim = model_dim
        self.num_colors = num_colors
        self.patch_size = patch_size

        # Project to per-patch features with more channels for PixelShuffle
        # PixelShuffle requires channels to be divisible by patch_size^2
        self.patch_features = nn.Linear(model_dim, num_colors * patch_size * patch_size)

        # PixelShuffle upsampling
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

        # Optional refinement after PixelShuffle
        self.refinement = nn.Sequential(
            nn.Conv2d(num_colors, num_colors * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_colors * 2, num_colors, kernel_size=3, padding=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens to per-pixel logits using PixelShuffle.

        Args:
            patch_tokens: [B, num_patches, model_dim]

        Returns:
            logits: [B, num_colors, H, W] - per-pixel classification logits
        """
        B, num_patches, d = patch_tokens.shape

        # Project to per-patch features
        patch_features = self.patch_features(
            patch_tokens
        )  # [B, num_patches, num_colors * patch_size^2]

        # Reshape to spatial format
        num_patches_h = num_patches_w = int(num_patches**0.5)
        patch_features = patch_features.view(
            B,
            num_patches_h,
            num_patches_w,
            self.num_colors * self.patch_size * self.patch_size,
        )
        patch_features = patch_features.permute(
            0, 3, 1, 2
        )  # [B, num_colors * patch_size^2, num_patches_h, num_patches_w]

        # PixelShuffle upsampling from 10x10 to 30x30
        logits = self.pixel_shuffle(patch_features)  # [B, num_colors, 30, 30]

        # Apply refinement network
        logits = self.refinement(logits)  # [B, num_colors, 30, 30]

        return logits
