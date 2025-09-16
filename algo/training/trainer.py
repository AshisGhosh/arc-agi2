import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import json

from ..config import Config
from ..models import SimpleARCModel
from .losses import calculate_classification_loss


class ARCTrainer:
    """
    Trainer for SimpleARC model with progress monitoring and checkpointing.
    """

    def __init__(self, model: SimpleARCModel, config: Config, dataset=None):
        """
        Initialize trainer.

        Args:
            model: SimpleARC model instance
            config: Configuration object
            dataset: Dataset instance (needed for getting all training examples)
        """
        self.model = model
        self.config = config
        self.dataset = dataset
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        # Create directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {"cross_entropy_loss": 0.0, "accuracy": 0.0}

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
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

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for key, value in components.items():
                if key != "total_loss":  # Skip total_loss as it's handled separately
                    loss_components[key] += value

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log interval
            if batch_idx % self.config.log_interval == 0:
                print(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

        return {"total_loss": avg_loss, **avg_components}

    def train_epoch_rule_latent(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using rule latent training approach.

        Args:
            train_loader: Training data loader with new data structure

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_holdout_loss = 0.0
        loss_components = {"cross_entropy_loss": 0.0, "accuracy": 0.0}
        holdout_components = {"cross_entropy_loss": 0.0, "accuracy": 0.0}

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} (Rule Latent)")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            rule_latent_inputs = batch["rule_latent_inputs"].to(
                self.device
            )  # [B, 2, 2, 3, 64, 64]
            test_inputs = batch["test_inputs"].to(self.device)  # [B, 1, 30, 30]
            test_outputs = batch["test_outputs"].to(self.device)  # [B, 1, 30, 30]
            holdout_inputs = batch["holdout_inputs"].to(self.device)  # [B, 1, 30, 30]
            holdout_outputs = batch["holdout_outputs"].to(self.device)  # [B, 1, 30, 30]
            has_holdout = batch["has_holdout"].to(self.device)  # [B]
            raw_rule_latent_targets = batch[
                "raw_rule_latent_targets"
            ]  # [B] list of raw targets (ARC format)

            batch_size = rule_latent_inputs.size(0)

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

            # Calculate training loss for all tasks
            batch_training_loss = 0.0
            batch_holdout_loss = 0.0

            # Process all tasks at once
            for i in range(batch_size):
                # Get rule latent for this task
                rule_latent = rule_latents[i : i + 1]  # [1, 128]

                # 1. Test example prediction (main target)
                test_logits = self.model.decoder(
                    rule_latent, test_inputs[i : i + 1]
                )  # [1, 10, 30, 30]
                test_loss, test_comp = calculate_classification_loss(
                    test_logits, test_outputs[i : i + 1], self.config
                )
                batch_training_loss += test_loss

                # Update test metrics
                for key, value in test_comp.items():
                    if key != "total_loss":
                        loss_components[key] += value

                # 2. Rule latent example 1 prediction (to prevent memorization)
                # Use raw targets from batch (ARC format)
                ex1_input = raw_rule_latent_targets[i][0]["input"]  # [1, 1, 30, 30]
                ex1_output = raw_rule_latent_targets[i][0]["output"]  # [1, 1, 30, 30]

                ex1_logits = self.model.decoder(
                    rule_latent, ex1_input
                )  # [1, 10, 30, 30]
                ex1_loss, ex1_comp = calculate_classification_loss(
                    ex1_logits, ex1_output, self.config
                )
                batch_training_loss += ex1_loss
                for key, value in ex1_comp.items():
                    if key != "total_loss":
                        loss_components[key] += value

                # 3. Rule latent example 2 prediction (to prevent memorization)
                ex2_input = raw_rule_latent_targets[i][1]["input"]  # [1, 1, 30, 30]
                ex2_output = raw_rule_latent_targets[i][1]["output"]  # [1, 1, 30, 30]

                ex2_logits = self.model.decoder(
                    rule_latent, ex2_input
                )  # [1, 10, 30, 30]
                ex2_loss, ex2_comp = calculate_classification_loss(
                    ex2_logits, ex2_output, self.config
                )
                batch_training_loss += ex2_loss
                for key, value in ex2_comp.items():
                    if key != "total_loss":
                        loss_components[key] += value

                # Calculate holdout loss (if available) - for validation only
                if has_holdout[i]:
                    holdout_logits = self.model.decoder(
                        rule_latent, holdout_inputs[i : i + 1]
                    )
                    holdout_loss, holdout_comp = calculate_classification_loss(
                        holdout_logits, holdout_outputs[i : i + 1], self.config
                    )
                    batch_holdout_loss += holdout_loss.item()
                    total_holdout_loss += holdout_loss.item()

                    # Update holdout metrics
                    for key, value in holdout_comp.items():
                        if key != "total_loss":
                            holdout_components[key] += value

            # Backward pass
            self.optimizer.zero_grad()
            batch_training_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            # Update metrics
            total_loss += batch_training_loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{batch_training_loss.item():.4f}",
                    "holdout": f"{batch_holdout_loss:.4f}"
                    if batch_holdout_loss > 0
                    else "N/A",
                }
            )

        # Calculate averages
        num_samples = len(train_loader)
        metrics = {
            "total_loss": total_loss / num_samples,
            **{k: v / num_samples for k, v in loss_components.items()},
        }

        # Add holdout metrics if available
        if total_holdout_loss > 0:
            metrics.update(
                {
                    "holdout_loss": total_holdout_loss / num_samples,
                    **{
                        f"holdout_{k}": v / num_samples
                        for k, v in holdout_components.items()
                    },
                }
            )

        return metrics

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
                    if (
                        key != "total_loss"
                    ):  # Skip total_loss as it's handled separately
                        loss_components[key] += value

        # Average metrics
        avg_loss = total_loss / len(val_loader)
        avg_components = {k: v / len(val_loader) for k, v in loss_components.items()}

        return {"total_loss": avg_loss, **avg_components}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

        # Load early stopping variables if they exist
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.patience_counter = checkpoint.get("patience_counter", 0)

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Log epoch results
            self.log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint and update best model
            is_best = val_metrics["total_loss"] < self.best_loss
            if is_best:
                self.best_loss = val_metrics["total_loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (patience: {self.config.early_stopping_patience})"
                )
                print(
                    f"Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch}"
                )
                break

        print("Training completed!")

    def log_epoch(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ):
        """
        Log epoch results.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        print(
            f"  Cross-Entropy: {train_metrics['cross_entropy_loss']:.4f} / {val_metrics['cross_entropy_loss']:.4f}"
        )
        print(
            f"  Accuracy: {train_metrics['accuracy']:.4f} / {val_metrics['accuracy']:.4f}"
        )
        print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

        # Save to log file
        log_entry = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        log_file = self.log_dir / "training_log.json"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
