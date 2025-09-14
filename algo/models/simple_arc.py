import torch
import torch.nn as nn
from .encoder import ResNetEncoder
from .decoder import MLPDecoder


class SimpleARCModel(nn.Module):
    """
    Simple ARC model combining ResNet encoder and MLP decoder.

    Processes example pairs through ResNet to extract rule latent,
    then uses MLP decoder to generate solution from rule + target input.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize components
        self.encoder = ResNetEncoder(config)
        self.decoder = MLPDecoder(config)

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
            Solution image [B, 1, 30, 30]
        """
        # Extract rule latent from example pairs
        rule_latent = self.encoder(
            example1_input, example1_output, example2_input, example2_output
        )  # [B, 128]

        # Generate solution from rule latent and target input
        solution = self.decoder(rule_latent, target_input)  # [B, 1, 30, 30]

        return solution
