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
            rule_latent_inputs = batch["rule_latent_inputs"].to(self.device)
            holdout_inputs = batch["holdout_inputs"].to(self.device)
            holdout_outputs = batch["holdout_outputs"].to(self.device)
            has_holdout = batch["has_holdout"].to(self.device)

            # Get task indices for this batch
            task_indices = batch["task_indices"]

            # Get all training examples for each task in the batch
            batch_size = rule_latent_inputs.size(0)
            max_train = 0
            all_train_inputs_list = []
            all_train_outputs_list = []
            num_train_list = []

            for i in range(batch_size):
                task_idx = task_indices[i]  # Already an integer from the list
                training_examples = self.dataset.get_all_training_examples_for_task(
                    task_idx
                )

                # Extract inputs and outputs
                train_inputs = [
                    ex["input"].squeeze(0) for ex in training_examples
                ]  # Remove batch dim
                train_outputs = [ex["output"].squeeze(0) for ex in training_examples]

                max_train = max(max_train, len(train_inputs))
                all_train_inputs_list.append(train_inputs)
                all_train_outputs_list.append(train_outputs)
                num_train_list.append(len(train_inputs))

            # Pad all training inputs to consistent shape
            all_train_inputs = torch.zeros([batch_size, max_train, 1, 30, 30]).to(
                self.device
            )
            all_train_outputs = torch.zeros([batch_size, max_train, 1, 30, 30]).to(
                self.device
            )

            for i, (train_inputs, train_outputs) in enumerate(
                zip(all_train_inputs_list, all_train_outputs_list)
            ):
                for j, (train_input, train_output) in enumerate(
                    zip(train_inputs, train_outputs)
                ):
                    all_train_inputs[i, j] = train_input
                    all_train_outputs[i, j] = train_output

            num_train = torch.tensor(num_train_list, dtype=torch.long).to(self.device)

            # Forward pass with batched rule latent training
            outputs = self.model.forward_rule_latent_training(
                rule_latent_inputs, all_train_inputs, num_train
            )

            # Calculate training loss for all tasks - vectorized
            batch_training_loss = 0.0
            batch_holdout_loss = 0.0

            # Process all tasks at once
            for i in range(rule_latent_inputs.size(0)):
                num_train_i = num_train[i].item()
                if num_train_i > 0:
                    # Get training targets for this task
                    train_logits = outputs["training_logits"][
                        i, :num_train_i
                    ]  # [num_train, 10, 30, 30]
                    train_outputs = all_train_outputs[
                        i, :num_train_i
                    ]  # [num_train, 1, 30, 30]

                    # Calculate loss for this task
                    task_loss, components = calculate_classification_loss(
                        train_logits, train_outputs, self.config
                    )
                    batch_training_loss += task_loss

                    # Update metrics
                    for key, value in components.items():
                        if key != "total_loss":
                            loss_components[key] += value

                # Calculate holdout loss (if available)
                if has_holdout[i]:
                    holdout_logits = self.model.decoder(
                        outputs["rule_latents"][i : i + 1], holdout_inputs[i : i + 1]
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
