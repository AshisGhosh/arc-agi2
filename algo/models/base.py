from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseARCModel(nn.Module, ABC):
    """
    Abstract base class for all ARC models.

    Defines the common interface that all ARC models must implement,
    ensuring consistency across different architectures.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        example1_input: torch.Tensor,
        example1_output: torch.Tensor,
        example2_input: torch.Tensor,
        example2_output: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through complete model.

        Args:
            example1_input: First input image [B, 3, 64, 64]
            example1_output: First output image [B, 3, 64, 64]
            example2_input: Second input image [B, 3, 64, 64]
            example2_output: Second output image [B, 3, 64, 64]
            target_input: Target input image [B, 1, 30, 30]

        Returns:
            Classification logits [B, 10, 30, 30]
        """
        pass

    @abstractmethod
    def forward_rule_latent_training(
        self,
        rule_latent_inputs: torch.Tensor,
        all_train_inputs: torch.Tensor,
        num_train: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for batched rule latent training.

        Args:
            rule_latent_inputs: [B, 2, 2, 3, 64, 64] - 2 examples per task
            all_train_inputs: [B, max_train, 1, 30, 30] - all training inputs
            num_train: [B] - number of training examples per task

        Returns:
            Dictionary containing:
                - training_logits: [B, max_train, 10, 30, 30]
                - rule_latents: [B, rule_dim] or None
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging/debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
        }

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
