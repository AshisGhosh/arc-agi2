import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
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
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.07)

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

            # Get test examples
            test_inputs = batch["test_inputs"]  # [B, max_test_examples, 30, 30]
            test_outputs = batch["test_outputs"]  # [B, max_test_examples, 30, 30]
            test_masks = batch["test_masks"]  # [B, max_test_examples]
            num_test_examples = batch["num_test_examples"]  # [B] list

            # Collect all valid test examples for batched processing
            all_test_inputs = []
            all_test_outputs = []
            all_support_inputs_expanded = []
            all_support_outputs_expanded = []

            for i in range(batch_size):
                num_test = num_test_examples[i]
                for test_idx in range(num_test):
                    if test_masks[i, test_idx]:
                        all_test_inputs.append(test_inputs[i, test_idx])  # [30, 30]
                        all_test_outputs.append(test_outputs[i, test_idx])  # [30, 30]
                        all_support_inputs_expanded.append(
                            support_inputs[i]
                        )  # [2, 30, 30]
                        all_support_outputs_expanded.append(
                            support_outputs[i]
                        )  # [2, 30, 30]

            if not all_test_inputs:
                continue  # Skip batch if no valid test examples

            # Convert to batched tensors
            batched_test_inputs = torch.stack(all_test_inputs)  # [N, 30, 30]
            batched_test_outputs = torch.stack(all_test_outputs)  # [N, 30, 30]
            batched_support_inputs = torch.stack(
                all_support_inputs_expanded
            )  # [N, 2, 30, 30]
            batched_support_outputs = torch.stack(
                all_support_outputs_expanded
            )  # [N, 2, 30, 30]

            # Main forward pass - batched
            main_logits = self.model.forward_batched(
                batched_support_inputs, batched_support_outputs, batched_test_inputs
            )  # [N, 10, 30, 30]

            # Calculate main loss
            main_loss, main_components = calculate_classification_loss(
                main_logits, batched_test_outputs, self.config
            )

            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(main_logits, dim=1)
                accuracy = (predictions == batched_test_outputs).float().mean()

            # Support reconstruction loss - batched
            support_loss = self._calculate_support_reconstruction_loss_batched(
                batched_support_inputs, batched_support_outputs
            )

            # CLS regularization loss - batched
            cls_loss = self._calculate_cls_regularization_loss_batched(
                batched_support_inputs, batched_support_outputs
            )

            # Total loss
            total_batch_loss = (
                main_loss
                + self.support_reconstruction_weight * support_loss
                + self.cls_regularization_weight * cls_loss
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
            total_examples += len(all_test_inputs)

            # Update loss components
            loss_components["main_loss"] += main_loss.item()
            loss_components["support_reconstruction_loss"] += support_loss.item()
            loss_components["cls_regularization_loss"] += cls_loss.item()
            loss_components["accuracy"] += accuracy.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_batch_loss.item():.4f}",
                    "main": f"{main_loss.item():.4f}",
                    "support": f"{support_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
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
        rule_tokens = self.model.get_rule_tokens_batched(
            support_inputs, support_outputs
        )

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
        pair_summaries = self.model.get_pair_summaries_batched(
            support_inputs, support_outputs
        )

        # Split into R_1 and R_2
        R_1 = pair_summaries[:, 0, :]  # [B, d_model]
        R_2 = pair_summaries[:, 1, :]  # [B, d_model]

        # Normalize embeddings
        R_1_norm = F.normalize(R_1, p=2, dim=1)
        R_2_norm = F.normalize(R_2, p=2, dim=1)

        # Calculate similarity
        similarity = torch.sum(R_1_norm * R_2_norm, dim=1)  # [B]

        # L2 regularization
        l2_loss = torch.mean(torch.norm(R_1, p=2, dim=1) + torch.norm(R_2, p=2, dim=1))

        # Contrastive loss: encourage R_1 and R_2 to be similar
        contrastive_loss = -torch.mean(similarity)

        return contrastive_loss + 0.01 * l2_loss

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

                # Get test examples
                test_inputs = batch["test_inputs"]
                test_outputs = batch["test_outputs"]
                test_masks = batch["test_masks"]
                num_test_examples = batch["num_test_examples"]

                # Collect all valid test examples for batched processing
                all_test_inputs = []
                all_test_outputs = []
                all_support_inputs_expanded = []
                all_support_outputs_expanded = []

                for i in range(batch_size):
                    num_test = num_test_examples[i]
                    for test_idx in range(num_test):
                        if test_masks[i, test_idx]:
                            all_test_inputs.append(test_inputs[i, test_idx])
                            all_test_outputs.append(test_outputs[i, test_idx])
                            all_support_inputs_expanded.append(support_inputs[i])
                            all_support_outputs_expanded.append(support_outputs[i])

                if not all_test_inputs:
                    continue

                # Convert to batched tensors
                batched_test_inputs = torch.stack(all_test_inputs)
                batched_test_outputs = torch.stack(all_test_outputs)
                batched_support_inputs = torch.stack(all_support_inputs_expanded)
                batched_support_outputs = torch.stack(all_support_outputs_expanded)

                # Main forward pass - batched
                main_logits = self.model.forward_batched(
                    batched_support_inputs, batched_support_outputs, batched_test_inputs
                )

                # Calculate main loss
                main_loss, _ = calculate_classification_loss(
                    main_logits, batched_test_outputs, self.config
                )

                # Calculate accuracy
                predictions = torch.argmax(main_logits, dim=1)
                accuracy = (predictions == batched_test_outputs).float().mean()

                # Support reconstruction loss - batched
                support_loss = self._calculate_support_reconstruction_loss_batched(
                    batched_support_inputs, batched_support_outputs
                )

                # CLS regularization loss - batched
                cls_loss = self._calculate_cls_regularization_loss_batched(
                    batched_support_inputs, batched_support_outputs
                )

                # Total loss
                total_batch_loss = (
                    main_loss
                    + self.support_reconstruction_weight * support_loss
                    + self.cls_regularization_weight * cls_loss
                )

                # Update metrics
                total_loss += total_batch_loss.item()
                total_examples += len(all_test_inputs)

                # Update loss components
                loss_components["main_loss"] += main_loss.item()
                loss_components["support_reconstruction_loss"] += support_loss.item()
                loss_components["cls_regularization_loss"] += cls_loss.item()
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
