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
        Train for one epoch using efficient batched training.

        Args:
            train_loader: Training data loader with flattened patch dataset

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        loss_components = {
            "cross_entropy_loss": 0.0,
            "patch_loss": 0.0,
            "accuracy": 0.0,
        }

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} (Patch)")

        for batch_idx, batch in enumerate(pbar):
            # Move all tensors to device
            support1_inputs = batch["support1_inputs"].to(self.device)  # [B, 30, 30]
            support1_outputs = batch["support1_outputs"].to(self.device)  # [B, 30, 30]
            support2_inputs = batch["support2_inputs"].to(self.device)  # [B, 30, 30]
            support2_outputs = batch["support2_outputs"].to(self.device)  # [B, 30, 30]
            test_inputs = batch["test_inputs"].to(self.device)  # [B, 30, 30]
            test_targets = batch["test_targets"].to(self.device)  # [B, 30, 30]

            batch_size = support1_inputs.size(0)

            # Single forward pass for entire batch
            logits = self.model(
                support1_inputs,
                support1_outputs,
                support2_inputs,
                support2_outputs,
                test_inputs,
            )  # [B, 10, 30, 30]

            # Calculate loss for entire batch
            loss, components = self._calculate_patch_loss(logits, test_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_examples += batch_size

            # Update loss components
            for key, value in components.items():
                if key != "total_loss":
                    loss_components[key] += value

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log interval
            if batch_idx % self.config.log_interval == 0:
                print(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        # Average metrics
        avg_loss = total_loss / total_examples
        avg_components = {k: v / total_examples for k, v in loss_components.items()}
        avg_components["total_loss"] = avg_loss

        return avg_components

    def _calculate_patch_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple:
        """
        Calculate enhanced loss for patch-based model with both pixel and patch supervision.

        Args:
            logits: [B, 10, 30, 30] - Per-pixel classification logits
            targets: [B, 30, 30] or [B, 1, 30, 30] - Target color indices

        Returns:
            Tuple of (loss, loss_components)
        """
        B = logits.size(0)

        # Handle both [B, 30, 30] and [B, 1, 30, 30] target shapes
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # [B, 1, 30, 30] -> [B, 30, 30]

        # Primary pixel-level cross-entropy loss
        logits_flat = logits.reshape(B, 10, -1)  # [B, 10, 900]
        targets_flat = targets.reshape(B, -1)  # [B, 900]
        pixel_loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat.long())

        # Additional patch-level supervision for better shape learning
        patch_loss = self._calculate_patch_level_loss(logits, targets)

        # Combine losses with configurable weights
        pixel_weight = getattr(self.config, "pixel_loss_weight", 1.0)
        patch_weight = getattr(self.config, "patch_loss_weight", 0.1)

        total_loss = pixel_weight * pixel_loss + patch_weight * patch_loss

        # Calculate accuracy for monitoring
        with torch.no_grad():
            predictions = torch.argmax(logits_flat, dim=1)  # [B, 900]
            accuracy = (predictions == targets_flat).float().mean()

        loss_components = {
            "cross_entropy_loss": pixel_loss.item(),
            "patch_loss": patch_loss.item(),
            "total_loss": total_loss.item(),
            "accuracy": accuracy.item(),
        }

        return total_loss, loss_components

    def _calculate_patch_level_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate patch-level loss to encourage consistent patch predictions.

        This helps the model learn to generate coherent shapes at the patch level.
        """
        B, C, H, W = logits.shape
        patch_size = 3  # Should match model's patch_size

        # Reshape to patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        # Reshape logits to patches: [B, C, H, W] -> [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
        logits_patches = logits.view(
            B, C, num_patches_h, patch_size, num_patches_w, patch_size
        )
        targets_patches = targets.view(
            B, num_patches_h, patch_size, num_patches_w, patch_size
        )

        # Get patch-level predictions by averaging logits within each patch
        logits_patch_avg = logits_patches.mean(
            dim=(3, 5)
        )  # [B, C, num_patches_h, num_patches_w]

        # Get patch-level targets by taking the most common color in each patch
        targets_patch_mode = torch.mode(
            targets_patches.view(B, num_patches_h, num_patches_w, -1), dim=-1
        )[0]

        # Flatten for cross-entropy
        logits_patch_flat = logits_patch_avg.view(B, C, -1)  # [B, C, num_patches]
        targets_patch_flat = targets_patch_mode.view(B, -1)  # [B, num_patches]

        # Patch-level cross-entropy loss
        patch_loss = torch.nn.functional.cross_entropy(
            logits_patch_flat, targets_patch_flat.long()
        )

        return patch_loss
