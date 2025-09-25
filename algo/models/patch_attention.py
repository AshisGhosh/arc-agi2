import torch
import torch.nn as nn
from typing import Dict, Any

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

        # Separate tokenizer for delta tokens (handles 20 colors: 0-9 for positive, 10-19 for negative)
        self.delta_tokenizer = PatchTokenizer(
            patch_size=self.patch_size,
            num_colors=20,  # 20 colors to handle sign information
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

    def tokenize_support_pair(
        self,
        support_input: torch.Tensor,
        support_output: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a support pair (X_s, Y_s) into X and Δ tokens.

        Args:
            support_input: [H, W] with values 0-9
            support_output: [H, W] with values 0-9

        Returns:
            Dictionary with 'x_tokens' and 'delta_tokens'
        """
        # Ensure inputs are on the same device as model parameters
        device = next(self.parameters()).device
        support_input = support_input.to(device)
        support_output = support_output.to(device)

        # Tokenize input
        x_tokens = self.patch_tokenizer(
            support_input.unsqueeze(0)
        )  # [1, num_patches, model_dim]
        x_tokens = self.pos_encoding(x_tokens)  # Add positional encoding

        # Compute delta tokens
        delta_colors = compute_delta_tokens(
            support_input.unsqueeze(0), support_output.unsqueeze(0)
        )  # [1, H, W] with values 0-19
        delta_tokens = self.delta_tokenizer(delta_colors)  # [1, num_patches, model_dim]
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
        # Ensure all inputs are on the same device as model parameters
        device = next(self.parameters()).device
        example1_input = example1_input.to(device)
        example1_output = example1_output.to(device)
        example2_input = example2_input.to(device)
        example2_output = example2_output.to(device)
        target_input = target_input.to(device)

        B = example1_input.size(0)

        # Tokenize support pairs (already in grayscale format)
        support1_tokens = self.tokenize_support_pair(
            example1_input[0], example1_output[0]
        )
        support2_tokens = self.tokenize_support_pair(
            example2_input[0], example2_output[0]
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
            # Get test input - should be [30, 30]
            test_input = target_input[i]  # [30, 30]

            # Tokenize test input (already in grayscale format)
            # Add batch dimension for tokenizer: [30, 30] -> [1, 30, 30]
            test_tokens = self.patch_tokenizer(test_input.unsqueeze(0))  # [1, 100, d]
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
