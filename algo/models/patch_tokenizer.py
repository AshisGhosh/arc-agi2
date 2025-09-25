import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
    """
    Patch tokenizer for converting 30x30 images to patch tokens.

    Converts images to onehot encoding, then patches them into 3x3 non-overlapping patches.
    Each patch becomes a token with 9*num_colors features (9 pixels * num_colors).

    For delta tokens, num_colors=20 to handle positive (0-9) and negative (10-19) deltas.
    """

    def __init__(self, patch_size: int = 3, num_colors: int = 10, model_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.model_dim = model_dim

        # Linear projection from patch features to model dimension
        patch_features = patch_size * patch_size * num_colors  # 3*3*num_colors
        self.patch_projection = nn.Linear(patch_features, model_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch tokens.

        Args:
            images: [B, H, W] with values 0-9 (color indices)

        Returns:
            tokens: [B, num_patches, model_dim]
        """
        B, H, W = images.shape

        # Ensure images are long type for onehot encoding
        images_long = images.long()

        # Convert to onehot encoding
        onehot = F.one_hot(
            images_long, num_classes=self.num_colors
        ).float()  # [B, H, W, 10]

        # Reshape to patches
        # Ensure dimensions are divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Image size {H}x{W} must be divisible by patch_size {self.patch_size}"
            )

        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Reshape to patches: [B, H, W, 10] -> [B, num_patches_h, patch_size, num_patches_w, patch_size, 10]
        patches = onehot.view(
            B,
            num_patches_h,
            self.patch_size,
            num_patches_w,
            self.patch_size,
            self.num_colors,
        )

        # Flatten patch dimensions: [B, num_patches_h, num_patches_w, patch_size*patch_size*10]
        patches = patches.permute(
            0, 1, 3, 2, 4, 5
        ).contiguous()  # [B, num_patches_h, num_patches_w, patch_size, patch_size, 10]
        patches = patches.view(
            B, num_patches, self.patch_size * self.patch_size * self.num_colors
        )

        # Project to model dimension
        tokens = self.patch_projection(patches)  # [B, num_patches, model_dim]

        return tokens


class PositionalEncoding(nn.Module):
    """
    Learned 2D absolute positional encoding for patch tokens.
    """

    def __init__(self, num_patches_h: int, num_patches_w: int, model_dim: int):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.model_dim = model_dim

        # Learned positional embeddings for each patch position
        self.pos_embedding = nn.Parameter(
            torch.randn(num_patches_h * num_patches_w, model_dim) * 0.02
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to tokens.

        Args:
            tokens: [B, num_patches, model_dim]

        Returns:
            tokens with positional encoding: [B, num_patches, model_dim]
        """
        return tokens + self.pos_embedding.unsqueeze(0)


def compute_delta_tokens(
    support_inputs: torch.Tensor, support_outputs: torch.Tensor
) -> torch.Tensor:
    """
    Compute delta tokens that preserve actual color change information.

    Encoding scheme:
    - No change (diff = 0): encode as 0
    - Positive change (diff > 0): encode as diff (1-9)
    - Negative change (diff < 0): encode as 10 + abs(diff) (10-19)

    This preserves the actual magnitude and direction of color changes.

    Examples:
    - input=5, output=5 → diff=0 → encode as 0
    - input=3, output=7 → diff=+4 → encode as 4
    - input=7, output=3 → diff=-4 → encode as 14 (10 + 4)
    - input=1, output=9 → diff=+8 → encode as 8
    - input=9, output=1 → diff=-8 → encode as 18 (10 + 8)

    Args:
        support_inputs: [B, H, W] with values 0-9
        support_outputs: [B, H, W] with values 0-9

    Returns:
        delta_colors: [B, H, W] with values 0-19 (preserving actual delta information)
    """
    # Ensure inputs are long type
    support_inputs_long = support_inputs.long()
    support_outputs_long = support_outputs.long()

    # Compute the actual color difference
    color_diff = support_outputs_long - support_inputs_long  # [B, H, W]

    # Create delta encoding that preserves actual change magnitude:
    # - No change: 0
    # - Positive change: use actual diff (1-9)
    # - Negative change: use 10 + abs(diff) (10-19)
    delta_colors = torch.where(
        color_diff == 0,
        torch.zeros_like(color_diff),  # No change: 0
        torch.where(
            color_diff > 0,
            color_diff,  # Positive change: use actual diff (1-9)
            10 + torch.abs(color_diff),  # Negative change: 10 + abs(diff) (10-19)
        ),
    )

    return delta_colors
