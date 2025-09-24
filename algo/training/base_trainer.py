import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from ..config import Config


class BaseTrainer:
    """
    Base trainer class for ARC models.

    Provides common functionality for model training, checkpointing, and logging.
    Subclasses must implement train_epoch method.
    """

    def __init__(self, model, config: Config, dataset=None):
        """
        Initialize trainer.

        Args:
            model: Model instance
            config: Configuration object
            dataset: Dataset instance (optional)
        """
        self.model = model
        self.config = config
        self.dataset = dataset
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
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
        self.patience_counter = 0

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
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        return checkpoint.get("epoch", 0)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch (to be implemented by subclasses).

        This is the standard interface that all trainers must implement.
        Subclasses can delegate to their specific training methods.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError("Subclasses must implement train_epoch method")

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model (to be implemented by subclasses).

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        raise NotImplementedError("Subclasses must implement validate method")
