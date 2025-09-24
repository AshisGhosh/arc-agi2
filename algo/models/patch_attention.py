import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import random

from .base import BaseARCModel
from .patch_tokenizer import PatchTokenizer, PositionalEncoding, compute_delta_tokens
from .cross_attention_decoder import CrossAttentionDecoder, PatchOutputHead


class PatchCrossAttentionModel(BaseARCModel):
    """
    Patch-based cross-attention model for ARC tasks.

    Architecture:
    1. Tokenize 30x30 images into 3x3 patches (100 tokens)
    2. Support pairs: (X_s, Y_s) -> compute Δ = onehot(Y_s) - onehot(X_s)
    3. Cross-attention: test tokens attend to support tokens
    4. Output: per-pixel logits via PixelShuffle upsampling
    """

    def __init__(self, config):
        super().__init__(config)

        # Model hyperparameters
        self.patch_size = getattr(config, "patch_size", 3)
        self.model_dim = getattr(config, "model_dim", 128)
        self.num_heads = getattr(config, "num_heads", 4)
        self.num_layers = getattr(config, "num_layers", 3)
        self.dropout = getattr(config, "dropout", 0.1)
        self.num_colors = 10

        # Image dimensions
        self.image_size = 30
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 100 patches

        # Components
        self.patch_tokenizer = PatchTokenizer(
            patch_size=self.patch_size,
            num_colors=self.num_colors,
            model_dim=self.model_dim,
        )

        self.pos_encoding = PositionalEncoding(
            num_patches_h=self.image_size // self.patch_size,
            num_patches_w=self.image_size // self.patch_size,
            model_dim=self.model_dim,
        )

        self.decoder = CrossAttentionDecoder(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self.output_head = PatchOutputHead(
            model_dim=self.model_dim,
            num_colors=self.num_colors,
            patch_size=self.patch_size,
        )

        # RMSNorm for support tokens (one-time normalization)
        self.support_norm = nn.RMSNorm(self.model_dim)

    def apply_palette_permutation(
        self, images: torch.Tensor, permutation: List[int]
    ) -> torch.Tensor:
        """
        Apply palette permutation to images.

        Args:
            images: [B, H, W] with values 0-9
            permutation: List of 10 integers representing color mapping

        Returns:
            permuted_images: [B, H, W] with permuted colors
        """
        permuted_images = torch.zeros_like(images)
        for old_color, new_color in enumerate(permutation):
            mask = images == old_color
            permuted_images[mask] = new_color
        return permuted_images

    def generate_palette_permutation(self) -> List[int]:
        """Generate random palette permutation for episode."""
        colors = list(range(10))
        random.shuffle(colors)
        return colors

    def tokenize_support_pair(
        self,
        support_input: torch.Tensor,
        support_output: torch.Tensor,
        palette_permutation: List[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a support pair (X_s, Y_s) into X and Δ tokens.

        Args:
            support_input: [H, W] with values 0-9
            support_output: [H, W] with values 0-9
            palette_permutation: Optional color permutation

        Returns:
            Dictionary with 'x_tokens' and 'delta_tokens'
        """
        # Apply palette permutation if provided
        if palette_permutation is not None:
            support_input = self.apply_palette_permutation(
                support_input, palette_permutation
            )
            support_output = self.apply_palette_permutation(
                support_output, palette_permutation
            )

        # Tokenize input
        x_tokens = self.patch_tokenizer(
            support_input.unsqueeze(0)
        )  # [1, num_patches, model_dim]
        x_tokens = self.pos_encoding(x_tokens)  # Add positional encoding

        # Compute delta tokens
        delta_colors = compute_delta_tokens(
            support_input.unsqueeze(0), support_output.unsqueeze(0)
        )  # [1, H, W]
        delta_tokens = self.patch_tokenizer(delta_colors)  # [1, num_patches, model_dim]
        delta_tokens = self.pos_encoding(delta_tokens)  # Add positional encoding

        return {"x_tokens": x_tokens, "delta_tokens": delta_tokens}

    def forward(
        self,
        example1_input: torch.Tensor,
        example1_output: torch.Tensor,
        example2_input: torch.Tensor,
        example2_output: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for patch-based cross-attention model.

        Args:
            example1_input: [B, 30, 30] - First input (grayscale format)
            example1_output: [B, 30, 30] - First output (grayscale format)
            example2_input: [B, 30, 30] - Second input (grayscale format)
            example2_output: [B, 30, 30] - Second output (grayscale format)
            target_input: [total_queries, 30, 30] - Target input (grayscale format)

        Returns:
            logits: [total_queries, 10, 30, 30] - Per-pixel classification logits
        """
        B = example1_input.size(0)

        # Generate palette permutation for this episode
        palette_permutation = self.generate_palette_permutation()

        # Tokenize support pairs (already in grayscale format)
        support1_tokens = self.tokenize_support_pair(
            example1_input[0], example1_output[0], palette_permutation
        )
        support2_tokens = self.tokenize_support_pair(
            example2_input[0], example2_output[0], palette_permutation
        )

        # Build support context: [X₁, Δ₁, X₂, Δ₂]
        support_tokens = torch.cat(
            [
                support1_tokens["x_tokens"],  # [1, 100, d]
                support1_tokens["delta_tokens"],  # [1, 100, d]
                support2_tokens["x_tokens"],  # [1, 100, d]
                support2_tokens["delta_tokens"],  # [1, 100, d]
            ],
            dim=1,
        )  # [1, 400, d]

        # Expand support tokens to match batch size
        support_tokens = support_tokens.expand(B, -1, -1)  # [B, 400, d]

        # One-time RMSNorm on support tokens
        support_tokens = self.support_norm(support_tokens)

        # Process all query inputs
        all_logits = []
        for i in range(target_input.size(0)):
            # Tokenize test input (already in grayscale format)
            test_input_permuted = self.apply_palette_permutation(
                target_input[i], palette_permutation
            )
            test_tokens = self.patch_tokenizer(
                test_input_permuted.unsqueeze(0)
            )  # [1, 100, d]
            test_tokens = self.pos_encoding(test_tokens)  # Add positional encoding

            # Use support tokens for this specific query (expand to match test_tokens batch size)
            query_support_tokens = support_tokens[:1]  # [1, 400, d]

            # Cross-attention: test tokens attend to support tokens
            updated_test_tokens = self.decoder(
                test_tokens, query_support_tokens
            )  # [1, 100, d]

            # Generate output logits
            logits = self.output_head(updated_test_tokens)  # [1, 10, 30, 30]
            all_logits.append(logits)

        # Concatenate all logits
        final_logits = torch.cat(all_logits, dim=0)  # [total_queries, 10, 30, 30]

        return final_logits

    def _convert_resnet_to_grayscale(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Convert ResNet format (64x64 RGB) to grayscale 30x30.
        This is a simplified conversion for compatibility.

        Args:
            rgb_image: [B, 3, 64, 64] - RGB image in [-1, 1] range

        Returns:
            grayscale: [B, 30, 30] - Grayscale image with values 0-9
        """
        B, C, H, W = rgb_image.shape

        # Convert to grayscale (simplified)
        gray = torch.mean(rgb_image, dim=1)  # [B, 64, 64]

        # Resize to 30x30
        gray = F.interpolate(gray.unsqueeze(1), size=(30, 30), mode="nearest").squeeze(
            1
        )  # [B, 30, 30]

        # Convert from [-1, 1] to [0, 9] range (simplified)
        gray = (gray + 1) / 2  # [0, 1]
        gray = gray * 9  # [0, 9]
        gray = torch.round(gray).long()  # [0, 9]

        return gray

    def forward_rule_latent_training(
        self,
        rule_latent_inputs: torch.Tensor,
        all_train_inputs: torch.Tensor,
        num_train: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for batched rule latent training (compatibility with existing training loop).

        This method is required by the base class but not used in patch-based training.
        Returns dummy values for compatibility.

        Args:
            rule_latent_inputs: [B, 2, 2, 3, 64, 64] - ResNet format inputs
            all_train_inputs: [B, max_train, 1, 30, 30] - Training inputs
            num_train: [B] - Number of training examples per task

        Returns:
            Dictionary with dummy values for compatibility
        """
        # For patch-based model, we don't use rule latent training
        # Return dummy values to maintain compatibility
        B = rule_latent_inputs.size(0)
        dummy_rule_latents = torch.zeros(
            B, self.model_dim, device=rule_latent_inputs.device
        )

        return {
            "rule_latents": dummy_rule_latents,
            "training_logits": None,  # Not used in patch-based training
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging/debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "patch_cross_attention",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "patch_size": self.patch_size,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_patches": self.num_patches,
        }
