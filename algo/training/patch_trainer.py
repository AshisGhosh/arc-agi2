import torch
from torch.utils.data import DataLoader
from typing import Dict

from tqdm import tqdm


from ..config import Config
from .base_trainer import BaseTrainer


class PatchTrainer(BaseTrainer):
    """
    Trainer for patch-based cross-attention model with episode-based training.

    Uses S→Q split: build KV from 2 supports, decode remaining pairs as queries.
    """

    def __init__(self, model, config: Config, dataset=None):
        super().__init__(model, config, dataset)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using episode-based training.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        return self.train_epoch_episode_based(train_loader)

    def train_epoch_episode_based(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using combination-based training (similar to ResNet decoder logic).

        Args:
            train_loader: Training data loader with patch dataset

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        loss_components = {"cross_entropy_loss": 0.0, "accuracy": 0.0}

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} (Patch)")

        for batch_idx, batch in enumerate(pbar):
            # Move tensor data to device
            test_inputs = batch["test_inputs"].to(
                self.device
            )  # [B, max_test_examples, 30, 30]
            test_targets = batch["test_targets"].to(
                self.device
            )  # [B, max_test_examples, 30, 30]
            test_masks = batch["test_masks"].to(self.device)  # [B, max_test_examples]
            num_test_examples = batch["num_test_examples"]  # [B] list

            # Get raw support examples
            support_example_inputs = batch[
                "support_example_inputs"
            ]  # [B] list of [2] support input tensors
            support_example_outputs = batch[
                "support_example_outputs"
            ]  # [B] list of [2] support output tensors

            batch_size = len(support_example_inputs)

            # Process each task in the batch
            batch_losses = []
            batch_examples = 0

            for i in range(batch_size):
                # Get number of test examples for this task
                num_test = num_test_examples[i]

                # Evaluate on all test examples for this task
                for test_idx in range(num_test):
                    if test_masks[i, test_idx]:  # Only process valid test examples
                        # Get specific test example
                        test_input = test_inputs[
                            i, test_idx : test_idx + 1
                        ]  # [1, 30, 30]
                        test_target = test_targets[
                            i, test_idx : test_idx + 1
                        ]  # [1, 30, 30]

                        # Forward pass through patch model
                        # Get support examples for this task (should be grayscale [1, 30, 30])
                        support_input = support_example_inputs[i][0].unsqueeze(
                            0
                        )  # [1, 30, 30]
                        support_output = support_example_outputs[i][0].unsqueeze(
                            0
                        )  # [1, 30, 30]

                        # Remove channel dimension to get [1, 30, 30]
                        support_input = support_input.squeeze(1)  # [1, 30, 30]
                        support_output = support_output.squeeze(1)  # [1, 30, 30]

                        logits = self.model(
                            support_input,  # [1, 30, 30] - support input
                            support_output,  # [1, 30, 30] - support output
                            test_input,  # [1, 30, 30] - test input
                        )

                        # Calculate loss for this test example
                        loss, components = self._calculate_patch_loss(
                            logits, test_target
                        )

                        batch_losses.append(loss)
                        batch_examples += 1

                        # Update loss components
                        for key, value in components.items():
                            if key != "total_loss":
                                loss_components[key] += value

            # Combine losses for backward pass
            if batch_losses:
                avg_batch_loss = torch.stack(batch_losses).mean()
            else:
                avg_batch_loss = torch.tensor(
                    0.0, device=self.device, requires_grad=True
                )

            # Backward pass
            self.optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            # Update metrics
            total_loss += avg_batch_loss.item()
            total_examples += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{avg_batch_loss.item():.4f}"})

            # Log interval
            if batch_idx % self.config.log_interval == 0:
                print(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {avg_batch_loss.item():.4f}"
                )

        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        avg_components["total_loss"] = avg_loss

        return avg_components

    def _calculate_patch_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple:
        """
        Calculate loss for patch-based model.

        Args:
            logits: [B, 10, 30, 30] - Per-pixel classification logits
            targets: [B, 1, 30, 30] - Target color indices

        Returns:
            Tuple of (loss, loss_components)
        """
        # Flatten spatial dimensions for cross-entropy
        logits_flat = logits.reshape(logits.size(0), 10, -1)  # [B, 10, 900]
        targets_flat = targets.reshape(targets.size(0), -1)  # [B, 900]

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat.long())

        # Calculate accuracy for monitoring
        with torch.no_grad():
            predictions = torch.argmax(logits_flat, dim=1)  # [B, 900]
            accuracy = (predictions == targets_flat).float().mean()

        loss_components = {
            "cross_entropy_loss": loss.item(),
            "accuracy": accuracy.item(),
            "total_loss": loss.item(),
        }

        return loss, loss_components
