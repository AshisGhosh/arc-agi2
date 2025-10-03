import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm

from ..config import Config
from .base_trainer import BaseTrainer
from .losses import calculate_classification_loss


class TransformerTrainer(BaseTrainer):
    """
    Trainer for transformer-based ARC model with auxiliary losses.

    Features:
    - Main loss: pixel-level cross-entropy on test output
    - Support reconstruction: decode support inputs with rule tokens
    - CLS regularization: contrastive loss on pair summaries
    - Rule token consistency: encourage similar rule tokens across augmentation groups
    """

    def __init__(self, model, config: Config, dataset=None):
        super().__init__(model, config, dataset)

        # Loss weights
        self.support_reconstruction_weight = getattr(
            config, "support_reconstruction_weight", 0.1
        )
        self.cls_regularization_weight = getattr(
            config, "cls_regularization_weight", 0.01
        )
        self.rule_token_consistency_weight = getattr(
            config, "rule_token_consistency_weight", 0.01
        )
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.07)
        self.cls_l2_weight = getattr(config, "cls_l2_weight", 0.01)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with batched auxiliary losses.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        loss_components = {
            "main_loss": 0.0,
            "support_reconstruction_loss": 0.0,
            "cls_regularization_loss": 0.0,
            "rule_token_consistency_loss": 0.0,
            "accuracy": 0.0,
        }

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} (Transformer)")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get support examples and convert to batched format
            support_example_inputs = batch[
                "support_example_inputs"
            ]  # [B] list of [2] tensors
            support_example_outputs = batch[
                "support_example_outputs"
            ]  # [B] list of [2] tensors

            batch_size = len(support_example_inputs)

            # Convert support examples to batched tensors [B, 2, 30, 30]
            support_inputs = torch.stack(
                [
                    torch.stack(
                        [
                            support_example_inputs[i][0].squeeze(0),
                            support_example_inputs[i][1].squeeze(0),
                        ],
                        dim=0,
                    )
                    for i in range(batch_size)
                ]
            )  # [B, 2, 30, 30]

            support_outputs = torch.stack(
                [
                    torch.stack(
                        [
                            support_example_outputs[i][0].squeeze(0),
                            support_example_outputs[i][1].squeeze(0),
                        ],
                        dim=0,
                    )
                    for i in range(batch_size)
                ]
            )  # [B, 2, 30, 30]

            # Get target examples for cycling (instead of test examples)
            target_example_inputs = batch[
                "target_example_inputs"
            ]  # [B] list of target input tensors
            target_example_outputs = batch[
                "target_example_outputs"
            ]  # [B] list of target output tensors

            # Convert target examples to batched tensors
            batched_target_inputs = torch.stack(target_example_inputs).to(
                self.device
            )  # [B, 30, 30]
            batched_target_outputs = torch.stack(target_example_outputs).to(
                self.device
            )  # [B, 30, 30]

            # Main forward pass - cycling: use support examples to predict target
            main_logits = self.model.forward_with_support_batch(
                support_inputs, support_outputs, batched_target_inputs
            )  # [B, 10, 30, 30]

            # Calculate main loss
            main_loss, main_components = calculate_classification_loss(
                main_logits, batched_target_outputs, self.config
            )

            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(main_logits, dim=1)
                accuracy = (predictions == batched_target_outputs).float().mean()

            # Support reconstruction loss - batched
            support_loss = self._calculate_support_reconstruction_loss_batched(
                support_inputs, support_outputs
            )

            # CLS regularization loss - batched
            cls_loss = self._calculate_cls_regularization_loss_batched(
                support_inputs, support_outputs
            )

            # Rule token consistency loss - batched
            consistency_loss = self._calculate_rule_token_consistency_loss_batched(
                support_inputs,
                support_outputs,
                batch["task_indices"],
                batch["augmentation_groups"],
            )

            # Total loss
            total_batch_loss = (
                main_loss
                + self.support_reconstruction_weight * support_loss
                + self.cls_regularization_weight * cls_loss
                + self.rule_token_consistency_weight * consistency_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            # Update metrics
            total_loss += total_batch_loss.item()
            total_examples += batch_size

            # Update loss components
            loss_components["main_loss"] += main_loss.item()
            loss_components["support_reconstruction_loss"] += support_loss.item()
            loss_components["cls_regularization_loss"] += cls_loss.item()
            loss_components["rule_token_consistency_loss"] += consistency_loss.item()
            loss_components["accuracy"] += accuracy.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_batch_loss.item():.4f}",
                    "main": f"{main_loss.item():.4f}",
                    "support": f"{support_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "consistency": f"{consistency_loss.item():.4f}",
                    "acc": f"{accuracy.item():.4f}",
                }
            )

            # Log interval
            if batch_idx % self.config.log_interval == 0:
                print(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {total_batch_loss.item():.4f}, "
                    f"Main: {main_loss.item():.4f}, "
                    f"Support: {support_loss.item():.4f}, "
                    f"CLS: {cls_loss.item():.4f}, "
                    f"Consistency: {consistency_loss.item():.4f}, "
                    f"Acc: {accuracy.item():.4f}"
                )

        # Average metrics
        if total_examples > 0:
            avg_loss = total_loss / total_examples
            avg_components = {k: v / total_examples for k, v in loss_components.items()}
        else:
            avg_loss = 0.0
            avg_components = {k: 0.0 for k in loss_components.keys()}

        avg_components["total_loss"] = avg_loss

        return avg_components

    def _calculate_support_reconstruction_loss_batched(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate batched support reconstruction loss.

        Args:
            support_inputs: [B, 2, 30, 30] - batch of support input pairs
            support_outputs: [B, 2, 30, 30] - batch of support output pairs

        Returns:
            support_loss: scalar tensor
        """
        B = support_inputs.shape[0]

        # Get rule tokens for all pairs
        rule_tokens = self.model.get_rule_tokens(support_inputs, support_outputs)

        # Reshape for batched processing
        all_support_inputs = support_inputs.view(B * 2, 30, 30)  # [2B, 30, 30]
        all_support_outputs = support_outputs.view(B * 2, 30, 30)  # [2B, 30, 30]

        # Expand rule tokens for each support example
        expanded_rule_tokens = rule_tokens.repeat_interleave(
            2, dim=0
        )  # [2B, num_rule_tokens, d_model]

        # Reconstruct all support examples
        support_processed = self.model.cross_attention_decoder(
            all_support_inputs, expanded_rule_tokens
        )
        support_pred = self.model.output_head(support_processed)

        # Calculate loss
        support_loss, _ = calculate_classification_loss(
            support_pred, all_support_outputs, self.config
        )

        return support_loss

    def _calculate_cls_regularization_loss_batched(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate batched CLS regularization loss.

        Args:
            support_inputs: [B, 2, 30, 30] - batch of support input pairs
            support_outputs: [B, 2, 30, 30] - batch of support output pairs

        Returns:
            cls_loss: scalar tensor
        """
        # Get pair summaries
        pair_summaries = self.model.get_pair_summaries(support_inputs, support_outputs)

        # Split into R_1 and R_2
        R_1 = pair_summaries[:, 0, :]  # [B, d_model]
        R_2 = pair_summaries[:, 1, :]  # [B, d_model]

        # Normalize embeddings
        R_1_norm = F.normalize(R_1, p=2, dim=1)
        R_2_norm = F.normalize(R_2, p=2, dim=1)

        # Calculate similarity with temperature scaling
        similarity = (
            torch.sum(R_1_norm * R_2_norm, dim=1) / self.contrastive_temperature
        )  # [B]

        # L2 regularization
        l2_loss = torch.mean(torch.norm(R_1, p=2, dim=1) + torch.norm(R_2, p=2, dim=1))

        # Contrastive loss: encourage R_1 and R_2 to be similar
        contrastive_loss = -torch.mean(similarity)

        return contrastive_loss + self.cls_l2_weight * l2_loss

    def _calculate_rule_token_consistency_loss_batched(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        task_indices: List[int],
        augmentation_groups: List[int],
    ) -> torch.Tensor:
        """
        Calculate batched rule token consistency loss across augmentation groups.

        Args:
            support_inputs: [B, 2, 30, 30] - batch of support input pairs
            support_outputs: [B, 2, 30, 30] - batch of support output pairs
            task_indices: [B] - task index for each sample
            augmentation_groups: [B] - augmentation group for each sample

        Returns:
            consistency_loss: scalar tensor
        """
        B = support_inputs.shape[0]

        # Get rule tokens for all pairs
        rule_tokens = self.model.get_rule_tokens(support_inputs, support_outputs)

        # If bottleneck is enabled, get compressed tokens
        if (
            hasattr(self.model, "rule_bottleneck")
            and self.model.rule_bottleneck is not None
        ):
            # Get the rule tokens before bottleneck expansion
            pair_summaries = self.model.get_pair_summaries(
                support_inputs, support_outputs
            )
            rule_tokens_before_bottleneck = self.model.pma(pair_summaries)
            # Apply only the down-projection to get compressed tokens
            rule_tokens_compressed = self.model.rule_bottleneck.down_proj(
                rule_tokens_before_bottleneck
            )
            rule_tokens = rule_tokens_compressed

        # Group rule tokens by (task_idx, augmentation_group)
        groups = {}
        for i in range(B):
            task_idx = task_indices[i]
            aug_group = augmentation_groups[i]
            key = (task_idx, aug_group)
            if key not in groups:
                groups[key] = []
            groups[key].append(rule_tokens[i])

        # Calculate within-group consistency loss
        total_loss = 0.0
        group_count = 0

        for group_tokens in groups.values():
            if len(group_tokens) > 1:  # Need at least 2 samples for regularization
                group_tensor = torch.stack(
                    group_tokens
                )  # [N, num_rule_tokens, d_model]

                # Flatten rule tokens for comparison
                group_flat = group_tensor.view(
                    group_tensor.size(0), -1
                )  # [N, num_rule_tokens * d_model]

                # Calculate pairwise L2 distances (want low distances for consistency)
                distances = torch.cdist(group_flat, group_flat, p=2)

                # Only consider upper triangular part (avoid double counting and self-comparison)
                mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
                group_loss = distances[mask].mean()

                total_loss += group_loss
                group_count += 1

        # Normalize by number of groups with multiple samples
        if group_count > 0:
            consistency_loss = total_loss / group_count
        else:
            consistency_loss = torch.tensor(0.0, device=rule_tokens.device)

        return consistency_loss

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model with batched processing.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        loss_components = {
            "main_loss": 0.0,
            "support_reconstruction_loss": 0.0,
            "cls_regularization_loss": 0.0,
            "rule_token_consistency_loss": 0.0,
            "accuracy": 0.0,
        }

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get support examples and convert to batched format
                support_example_inputs = batch["support_example_inputs"]
                support_example_outputs = batch["support_example_outputs"]

                batch_size = len(support_example_inputs)

                # Convert support examples to batched tensors [B, 2, 30, 30]
                support_inputs = torch.stack(
                    [
                        torch.stack(
                            [
                                support_example_inputs[i][0].squeeze(0),
                                support_example_inputs[i][1].squeeze(0),
                            ],
                            dim=0,
                        )
                        for i in range(batch_size)
                    ]
                )

                support_outputs = torch.stack(
                    [
                        torch.stack(
                            [
                                support_example_outputs[i][0].squeeze(0),
                                support_example_outputs[i][1].squeeze(0),
                            ],
                            dim=0,
                        )
                        for i in range(batch_size)
                    ]
                )

                # Get target examples for cycling (instead of test examples)
                target_example_inputs = batch["target_example_inputs"]
                target_example_outputs = batch["target_example_outputs"]

                # Convert target examples to batched tensors
                batched_target_inputs = torch.stack(target_example_inputs)
                batched_target_outputs = torch.stack(target_example_outputs)

                # Main forward pass - cycling: use support examples to predict target
                main_logits = self.model.forward_with_support_batch(
                    support_inputs, support_outputs, batched_target_inputs
                )

                # Calculate main loss
                main_loss, _ = calculate_classification_loss(
                    main_logits, batched_target_outputs, self.config
                )

                # Calculate accuracy
                predictions = torch.argmax(main_logits, dim=1)
                accuracy = (predictions == batched_target_outputs).float().mean()

                # Support reconstruction loss - batched
                support_loss = self._calculate_support_reconstruction_loss_batched(
                    support_inputs, support_outputs
                )

                # CLS regularization loss - batched
                cls_loss = self._calculate_cls_regularization_loss_batched(
                    support_inputs, support_outputs
                )

                # Rule token consistency loss - batched
                consistency_loss = self._calculate_rule_token_consistency_loss_batched(
                    support_inputs,
                    support_outputs,
                    batch["task_indices"],
                    batch["augmentation_groups"],
                )

                # Total loss
                total_batch_loss = (
                    main_loss
                    + self.support_reconstruction_weight * support_loss
                    + self.cls_regularization_weight * cls_loss
                    + self.rule_token_consistency_weight * consistency_loss
                )

                # Update metrics
                total_loss += total_batch_loss.item()
                total_examples += batch_size

                # Update loss components
                loss_components["main_loss"] += main_loss.item()
                loss_components["support_reconstruction_loss"] += support_loss.item()
                loss_components["cls_regularization_loss"] += cls_loss.item()
                loss_components["rule_token_consistency_loss"] += (
                    consistency_loss.item()
                )
                loss_components["accuracy"] += accuracy.item()

        # Average metrics
        if total_examples > 0:
            avg_loss = total_loss / total_examples
            avg_components = {k: v / total_examples for k, v in loss_components.items()}
        else:
            avg_loss = 0.0
            avg_components = {k: 0.0 for k in loss_components.keys()}

        avg_components["total_loss"] = avg_loss

        return avg_components
