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

    def __init__(self, model: SimpleARCModel, config: Config):
        """
        Initialize trainer.

        Args:
            model: SimpleARC model instance
            config: Configuration object
        """
        self.model = model
        self.config = config
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

            # Save checkpoint
            is_best = val_metrics["total_loss"] < self.best_loss
            if is_best:
                self.best_loss = val_metrics["total_loss"]

            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

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
        print(f"  Cross-Entropy: {train_metrics['cross_entropy_loss']:.4f} / {val_metrics['cross_entropy_loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f} / {val_metrics['accuracy']:.4f}")
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
