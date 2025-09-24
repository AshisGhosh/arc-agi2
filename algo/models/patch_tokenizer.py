import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
    """
    Patch tokenizer for converting 30x30 images to patch tokens.

    Converts images to onehot encoding, then patches them into 3x3 non-overlapping patches.
    Each patch becomes a token with 9*10=90 features (9 pixels * 10 colors).
    """

    def __init__(self, patch_size: int = 3, num_colors: int = 10, model_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.model_dim = model_dim

        # Linear projection from patch features to model dimension
        patch_features = patch_size * patch_size * num_colors  # 3*3*10 = 90
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
    Compute delta tokens: onehot(Y) - onehot(X) for support pairs.

    Args:
        support_inputs: [B, H, W] with values 0-9
        support_outputs: [B, H, W] with values 0-9

    Returns:
        delta_tokens: [B, num_patches, model_dim] - difference tokens
    """
    # Ensure inputs are long type
    support_inputs_long = support_inputs.long()
    support_outputs_long = support_outputs.long()

    # Convert to onehot
    input_onehot = F.one_hot(
        support_inputs_long, num_classes=10
    ).float()  # [B, H, W, 10]
    output_onehot = F.one_hot(
        support_outputs_long, num_classes=10
    ).float()  # [B, H, W, 10]

    # Compute difference
    delta_onehot = output_onehot - input_onehot  # [B, H, W, 10]

    # Convert back to color indices for tokenization
    # Find the color with maximum absolute value in each position
    delta_abs = torch.abs(delta_onehot)  # [B, H, W, 10]
    delta_colors = torch.argmax(delta_abs, dim=-1)  # [B, H, W]

    # Get the actual delta values for the selected colors
    delta_values = delta_onehot.gather(-1, delta_colors.unsqueeze(-1)).squeeze(
        -1
    )  # [B, H, W]

    # Create a simple mapping: if delta > 0, use color + 10, if delta < 0, use color
    # This gives us values 0-19, but we need 0-9, so we'll use modulo
    delta_colors = (
        delta_colors + (delta_values > 0).long() * 10
    )  # [B, H, W] with values 0-19
    delta_colors = delta_colors % 10  # [B, H, W] with values 0-9

    return delta_colors
