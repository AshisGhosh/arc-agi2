import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from ..config import Config
from .base_trainer import BaseTrainer
from .losses import (
    calculate_classification_loss,
    calculate_rule_latent_regularization_loss,
)


class ResNetTrainer(BaseTrainer):
    """
    Trainer for SimpleARC model with progress monitoring and checkpointing.
    """

    def __init__(self, model, config: Config, dataset=None):
        """
        Initialize trainer.

        Args:
            model: SimpleARC model instance
            config: Configuration object
            dataset: Dataset instance (needed for getting all training examples)
        """
        super().__init__(model, config, dataset)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using rule latent training (the preferred method for ResNet).

        This method delegates to train_epoch_rule_latent which is the sophisticated
        training approach for the ResNet model.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        return self.train_epoch_rule_latent(train_loader)

    def train_epoch_rule_latent(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using rule latent training approach with multiple test examples.

        Args:
            train_loader: Training data loader with new data structure

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_holdout_loss = 0.0
        total_decoder_passes = 0  # Track total decoder passes for proper normalization
        total_holdout_passes = 0  # Track total holdout passes for proper normalization
        total_batches = 0  # Track total batches for proper normalization
        loss_components = {"cross_entropy_loss": 0.0, "accuracy": 0.0}
        holdout_components = {"cross_entropy_loss": 0.0, "accuracy": 0.0}

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} (Rule Latent)")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            test_inputs = batch["test_inputs"].to(
                self.device
            )  # [B, max_test_examples, 30, 30]
            test_outputs = batch["test_outputs"].to(
                self.device
            )  # [B, max_test_examples, 30, 30]
            test_masks = batch["test_masks"].to(self.device)  # [B, max_test_examples]
            holdout_inputs = batch["holdout_inputs"].to(self.device)  # [B, 1, 30, 30]
            holdout_outputs = batch["holdout_outputs"].to(self.device)  # [B, 1, 30, 30]
            has_holdout = batch["has_holdout"].to(self.device)  # [B]
            num_test_examples = batch[
                "num_test_examples"
            ]  # [B] list of number of test examples per task

            batch_size = len(batch["support_example_inputs_rgb"])

            # Get RGB support examples for ResNet
            support_inputs_rgb = batch[
                "support_example_inputs_rgb"
            ]  # [B] list of [2] RGB tensors
            support_outputs_rgb = batch[
                "support_example_outputs_rgb"
            ]  # [B] list of [2] RGB tensors

            # Create rule latent inputs tensor [B, 2, 2, 3, 64, 64]
            rule_latent_inputs = torch.zeros(
                [batch_size, 2, 2, 3, 64, 64], device=self.device
            )
            for i in range(batch_size):
                rule_latent_inputs[i, 0, 0] = support_inputs_rgb[i][0]  # [3, 64, 64]
                rule_latent_inputs[i, 0, 1] = support_outputs_rgb[i][0]  # [3, 64, 64]
                rule_latent_inputs[i, 1, 0] = support_inputs_rgb[i][1]  # [3, 64, 64]
                rule_latent_inputs[i, 1, 1] = support_outputs_rgb[i][1]  # [3, 64, 64]

            # Reshape rule latent inputs for encoder: [B, 4, 3, 64, 64] where 4 = 2 examples * 2 images each
            rule_inputs_batch = rule_latent_inputs.view(batch_size, 4, 3, 64, 64)

            # Split into the 4 components the encoder expects
            example1_inputs = rule_inputs_batch[:, 0]  # [B, 3, 64, 64]
            example1_outputs = rule_inputs_batch[:, 1]  # [B, 3, 64, 64]
            example2_inputs = rule_inputs_batch[:, 2]  # [B, 3, 64, 64]
            example2_outputs = rule_inputs_batch[:, 3]  # [B, 3, 64, 64]

            # Run encoder on entire batch at once
            rule_latents = self.model.encoder(
                example1_inputs, example1_outputs, example2_inputs, example2_outputs
            )  # [B, 128]

            # Calculate rule latent regularization loss
            reg_loss, reg_components = calculate_rule_latent_regularization_loss(
                rule_latents,
                batch["task_indices"],
                batch["augmentation_groups"],
                self.config.rule_latent_regularization_weight,
            )

            # Process each task in the batch
            batch_losses = []
            batch_holdout_losses = []
            batch_decoder_passes = 0
            batch_holdout_passes = 0

            for i in range(batch_size):
                # Get grayscale support examples for decoder training
                support_example_inputs = batch[
                    "support_example_inputs"
                ]  # [B] list of [2] grayscale tensors
                support_example_outputs = batch[
                    "support_example_outputs"
                ]  # [B] list of [2] grayscale tensors

                # Train decoder on support input→output pairs
                for support_idx in range(2):  # 2 support examples
                    support_input = support_example_inputs[i][support_idx].unsqueeze(
                        0
                    )  # [1, 30, 30]
                    support_output = support_example_outputs[i][support_idx].unsqueeze(
                        0
                    )  # [1, 30, 30]

                    # Evaluate decoder on support pair
                    support_logits = self.model.decoder(
                        rule_latents[i : i + 1], support_input
                    )
                    support_target = support_output

                    # Calculate loss for this support pair
                    loss, components = calculate_classification_loss(
                        support_logits, support_target, self.config
                    )

                    batch_losses.append(loss)
                    batch_decoder_passes += 1

                    # Update loss components (normalize by total decoder passes later)
                    for key, value in components.items():
                        if key != "total_loss":
                            loss_components[key] += value

                # Get number of test examples for this task
                num_test = num_test_examples[i]

                # Evaluate on all test examples for this task
                for test_idx in range(num_test):
                    if test_masks[i, test_idx]:  # Only process valid test examples
                        # Get specific test example
                        test_input = test_inputs[
                            i, test_idx : test_idx + 1
                        ]  # [1, 30, 30]
                        test_output = test_outputs[
                            i, test_idx : test_idx + 1
                        ]  # [1, 30, 30]

                        # Evaluate on this test example
                        test_logits = self.model.decoder(
                            rule_latents[i : i + 1], test_input
                        )
                        test_target = test_output

                        # Calculate loss for this test example
                        loss, components = calculate_classification_loss(
                            test_logits, test_target, self.config
                        )

                        batch_losses.append(loss)
                        batch_decoder_passes += 1

                        # Update loss components (normalize by total decoder passes later)
                        for key, value in components.items():
                            if key != "total_loss":
                                loss_components[key] += value

                # Evaluate on holdout target (if available)
                if has_holdout[i]:
                    holdout_logits = self.model.decoder(
                        rule_latents[i : i + 1], holdout_inputs[i : i + 1]
                    )
                    holdout_target = holdout_outputs[i : i + 1]

                    # Calculate holdout loss
                    holdout_loss, holdout_comp = calculate_classification_loss(
                        holdout_logits, holdout_target, self.config
                    )

                    batch_holdout_losses.append(holdout_loss)
                    batch_holdout_passes += 1

                    # Update holdout loss components (normalize by total holdout passes later)
                    for key, value in holdout_comp.items():
                        if key != "total_loss":
                            holdout_components[key] += value

            # Combine losses for backward pass
            if batch_losses:
                avg_batch_loss = torch.stack(batch_losses).mean()
            else:
                avg_batch_loss = torch.tensor(
                    0.0, device=self.device, requires_grad=True
                )

            if batch_holdout_losses:
                avg_batch_holdout_loss = torch.stack(batch_holdout_losses).mean()
            else:
                avg_batch_holdout_loss = torch.tensor(0.0, device=self.device)

            # Track losses for metrics
            total_loss += avg_batch_loss.item()
            total_holdout_loss += avg_batch_holdout_loss.item()
            total_decoder_passes += batch_decoder_passes
            total_holdout_passes += batch_holdout_passes
            total_batches += 1

            # Note: regularization loss is already included in avg_batch_loss via backward pass
            reg_loss_value = reg_loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            (avg_batch_loss + reg_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{avg_batch_loss:.4f}",
                    "reg": f"{reg_loss_value:.4f}",
                    "holdout": f"{avg_batch_holdout_loss:.4f}",
                }
            )

            # Log interval
            if batch_idx % self.config.log_interval == 0:
                print(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {avg_batch_loss:.4f}, Reg: {reg_loss_value:.4f}"
                )

        # Average metrics
        if total_batches > 0:
            avg_loss = total_loss / total_batches
            avg_holdout_loss = total_holdout_loss / total_batches
        else:
            avg_loss = 0.0
            avg_holdout_loss = 0.0

        # Normalize loss components by total decoder passes processed
        if total_decoder_passes > 0:
            avg_components = {
                k: v / total_decoder_passes for k, v in loss_components.items()
            }
        else:
            avg_components = {k: 0.0 for k in loss_components.keys()}

        if total_holdout_passes > 0:
            avg_holdout_components = {
                k: v / total_holdout_passes for k, v in holdout_components.items()
            }
        else:
            avg_holdout_components = {k: 0.0 for k in holdout_components.keys()}

        return {
            "total_loss": avg_loss,
            "holdout_loss": avg_holdout_loss,
            **avg_components,
            **{f"holdout_{k}": v for k, v in avg_holdout_components.items()},
            "rule_latent_regularization": reg_components.get(
                "rule_latent_regularization", 0.0
            ),
            "active_groups": reg_components.get("active_groups", 0),
        }

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {"cross_entropy_loss": 0.0, "accuracy": 0.0}

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                logits = self.model(
                    batch["example1_input"],
                    batch["example1_output"],
                    batch["example2_input"],
                    batch["example2_output"],
                    batch["target_input"],
                )

                # Calculate loss
                loss, components = calculate_classification_loss(
                    logits, batch["target_output"], self.config
                )

                # Update metrics
                total_loss += loss.item()
                for key, value in components.items():
                    if key != "total_loss":
                        loss_components[key] += value

        # Average metrics
        avg_loss = total_loss / len(val_loader)
        avg_components = {k: v / len(val_loader) for k, v in loss_components.items()}

        return {"total_loss": avg_loss, **avg_components}
