#!/usr/bin/env python3
"""
Training script for SimpleARC model.

Trains the model on preprocessed ARC data with progress monitoring.
"""

import torch
from torch.utils.data import DataLoader, random_split
import argparse
from pathlib import Path
import json
from datetime import datetime

from algo.config import Config
from algo.models import SimpleARCModel
from algo.data import ARCDataset
from algo.training import ARCTrainer


def create_data_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataset
    dataset = ARCDataset(config.processed_dir, config)

    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Batch size: {config.batch_size}")

    return train_loader, val_loader


def create_experiment_directory(config: Config, dataset_name: str) -> Path:
    """create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"train_{dataset_name}_{timestamp}"
    experiment_dir = Path("logs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_training_info(
    experiment_dir: Path,
    config: Config,
    dataset_name: str,
    train_size: int,
    val_size: int,
    total_params: int,
) -> None:
    """save training information to json file."""
    training_info = {
        "dataset": dataset_name,
        "config": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "rule_dim": config.rule_dim,
            "device": str(config.device),
        },
        "data_split": {
            "train_size": train_size,
            "val_size": val_size,
            "total_size": train_size + val_size,
        },
        "model_info": {
            "total_parameters": total_params,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(experiment_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train SimpleARC model")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="arc1",
        choices=["arc1", "arc2"],
        help="Dataset to train on (arc1 or arc2, default: arc1)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config file path"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr

    # Set up deterministic training
    config.set_deterministic_training()

    # Map dataset argument to config value
    dataset_mapping = {"arc1": "arc_agi1", "arc2": "arc_agi2"}
    config.training_dataset = dataset_mapping[args.dataset]

    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Dataset: {config.training_dataset}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Rule dimension: {config.rule_dim}")

    # Create experiment directory
    experiment_dir = create_experiment_directory(config, args.dataset)
    print(f"\nExperiment directory: {experiment_dir}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)

    # Create model
    model = SimpleARCModel(config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # Save training info
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    save_training_info(
        experiment_dir, config, args.dataset, train_size, val_size, total_params
    )

    # Create trainer with custom checkpoint path
    trainer = ARCTrainer(model, config)

    # Override the checkpoint directory to save in experiment directory
    trainer.checkpoint_dir = experiment_dir

    # Resume from checkpoint if specified
    if args.resume:
        if Path(args.resume).exists():
            trainer.load_checkpoint(args.resume)
        else:
            print(f"Checkpoint not found: {args.resume}")
            return

    # Start training
    trainer.train(train_loader, val_loader)

    print("Training completed!")
    print(f"Experiment saved to: {experiment_dir}")
    print("You can now view this model in the Streamlit app!")


if __name__ == "__main__":
    main()
