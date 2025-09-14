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
            Classification logits [B, 10, 30, 30] for each pixel and color class
        """
        # Extract rule latent from example pairs
        rule_latent = self.encoder(
            example1_input, example1_output, example2_input, example2_output
        )  # [B, 128]

        # Generate solution logits from rule latent and target input
        logits = self.decoder(rule_latent, target_input)  # [B, 10, 30, 30]

        return logits

    def forward_rule_latent_training(
        self,
        rule_latent_inputs: torch.Tensor,
        all_train_inputs: torch.Tensor,
        num_train: torch.Tensor,
    ) -> dict:
        """
        Forward pass for batched rule latent training.

        Args:
            rule_latent_inputs: [B, 2, 2, 3, 64, 64] - 2 examples per task
            all_train_inputs: [B, max_train, 1, 30, 30] - all training inputs
            num_train: [B] - number of training examples per task

        Returns:
            Dictionary containing:
                - training_logits: [B, max_train, 10, 30, 30] - logits for all training targets
                - rule_latents: [B, 128] - rule latents for each task
        """
        batch_size = rule_latent_inputs.size(0)

        # Create rule latents for all tasks - fully vectorized
        # Reshape to [B, 4, 3, 64, 64] where 4 = 2 examples * 2 images each
        rule_inputs_batch = rule_latent_inputs.view(batch_size, 4, 3, 64, 64)

        # Split into the 4 components the encoder expects
        example1_inputs = rule_inputs_batch[:, 0]  # [B, 3, 64, 64]
        example1_outputs = rule_inputs_batch[:, 1]  # [B, 3, 64, 64]
        example2_inputs = rule_inputs_batch[:, 2]  # [B, 3, 64, 64]
        example2_outputs = rule_inputs_batch[:, 3]  # [B, 3, 64, 64]

        # Run encoder on entire batch at once
        rule_latents = self.encoder(
            example1_inputs, example1_outputs, example2_inputs, example2_outputs
        )  # [B, 128]

        # Generate logits for all training targets - vectorized
        max_train = all_train_inputs.size(1)
        all_logits = torch.zeros(
            batch_size,
            max_train,
            10,
            30,
            30,
            device=rule_latents.device,
            dtype=rule_latents.dtype,
        )

        for i in range(batch_size):
            num_train_i = num_train[i].item()
            if num_train_i > 0:
                # Get training inputs for this task
                train_inputs = all_train_inputs[
                    i, :num_train_i
                ]  # [num_train, 1, 30, 30]

                # Expand rule latent for all training examples
                rule_latent_expanded = rule_latents[i : i + 1].expand(
                    num_train_i, -1
                )  # [num_train, 128]

                # Generate logits
                logits = self.decoder(
                    rule_latent_expanded, train_inputs
                )  # [num_train, 10, 30, 30]
                all_logits[i, :num_train_i] = logits

        return {
            "training_logits": all_logits,
            "rule_latents": rule_latents,
        }
