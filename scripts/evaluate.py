#!/usr/bin/env python3
"""
Evaluation script for SimpleARC model.

Evaluates the trained model on ARC tasks and generates metrics.
"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json

from algo.config import Config
from algo.models import SimpleARCModel
from algo.data import ARCDataset


def calculate_perfect_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate perfect accuracy (exact match).

    Args:
        logits: Classification logits [B, 10, 30, 30]
        target: Target values [B, 1, 30, 30]

    Returns:
        Perfect accuracy (0.0 to 1.0)
    """
    # Get predictions using argmax
    predictions = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, 30, 30]

    # Check for exact matches
    exact_matches = torch.all(predictions == target, dim=(1, 2, 3))

    return exact_matches.float().mean().item()


def calculate_pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate pixel-level accuracy.

    Args:
        logits: Classification logits [B, 10, 30, 30]
        target: Target values [B, 1, 30, 30]

    Returns:
        Pixel accuracy (0.0 to 1.0)
    """
    # Get predictions using argmax
    predictions = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, 30, 30]

    # Calculate pixel-level matches
    pixel_matches = (predictions == target).float()

    return pixel_matches.mean().item()


def calculate_near_miss_accuracy(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 2.0
) -> float:
    """
    Calculate near-miss accuracy (within threshold).

    Args:
        logits: Classification logits [B, 10, 30, 30]
        target: Target values [B, 1, 30, 30]
        threshold: Maximum distance for near-miss

    Returns:
        Near-miss accuracy (0.0 to 1.0)
    """
    # Get predictions using argmax
    predictions = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, 30, 30]

    # Calculate absolute difference
    diff = torch.abs(predictions - target)

    # Check if within threshold
    near_misses = (diff <= threshold).float()

    return near_misses.mean().item()


def calculate_perfect_accuracy_foreground(
    logits: torch.Tensor, target: torch.Tensor, background_value: float = 0.0
) -> float:
    """
    Calculate perfect accuracy for foreground (non-background) pixels only.

    Args:
        logits: Classification logits [B, 10, 30, 30]
        target: Target values [B, 1, 30, 30]
        background_value: Value considered as background (default: 0.0)

    Returns:
        Perfect accuracy for foreground pixels (0.0 to 1.0)
    """
    # Get predictions using argmax
    predictions = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, 30, 30]

    # Create mask for foreground pixels in target
    target_foreground_mask = target != background_value

    # If no foreground pixels in target, check if prediction also has no foreground
    if not target_foreground_mask.any():
        pred_foreground_mask = predictions != background_value
        return (not pred_foreground_mask.any()).float().item()

    # Check for exact matches only on foreground pixels
    exact_matches = predictions == target
    foreground_matches = exact_matches & target_foreground_mask

    # Calculate accuracy as: correct foreground pixels / total foreground pixels
    foreground_accuracy = (
        foreground_matches.sum().float() / target_foreground_mask.sum().float()
    )

    return foreground_accuracy.item()


def calculate_pixel_accuracy_foreground(
    logits: torch.Tensor, target: torch.Tensor, background_value: float = 0.0
) -> float:
    """
    Calculate pixel-level accuracy for foreground (non-background) pixels only.

    Args:
        logits: Classification logits [B, 10, 30, 30]
        target: Target values [B, 1, 30, 30]
        background_value: Value considered as background (default: 0.0)

    Returns:
        Pixel accuracy for foreground pixels (0.0 to 1.0)
    """
    # Get predictions using argmax
    predictions = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, 30, 30]

    # Create mask for foreground pixels in target
    target_foreground_mask = target != background_value

    # If no foreground pixels in target, check if prediction also has no foreground
    if not target_foreground_mask.any():
        pred_foreground_mask = predictions != background_value
        return (not pred_foreground_mask.any()).float().item()

    # Calculate pixel-level matches only on foreground pixels
    pixel_matches = predictions == target
    foreground_pixel_matches = pixel_matches & target_foreground_mask

    # Calculate accuracy as: correct foreground pixels / total foreground pixels
    foreground_accuracy = (
        foreground_pixel_matches.sum().float() / target_foreground_mask.sum().float()
    )

    return foreground_accuracy.item()


def calculate_near_miss_accuracy_foreground(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 2.0,
    background_value: float = 0.0,
) -> float:
    """
    Calculate near-miss accuracy for foreground (non-background) pixels only.

    Args:
        logits: Classification logits [B, 10, 30, 30]
        target: Target values [B, 1, 30, 30]
        threshold: Maximum distance for near-miss
        background_value: Value considered as background (default: 0.0)

    Returns:
        Near-miss accuracy for foreground pixels (0.0 to 1.0)
    """
    # Get predictions using argmax
    predictions = torch.argmax(logits, dim=1, keepdim=True)  # [B, 1, 30, 30]

    # Calculate absolute difference
    diff = torch.abs(predictions - target)

    # Create mask for foreground pixels in target
    target_foreground_mask = target != background_value

    # If no foreground pixels in target, check if prediction also has no foreground
    if not target_foreground_mask.any():
        pred_foreground_mask = predictions != background_value
        return (not pred_foreground_mask.any()).float().item()

    # Check if within threshold only on foreground pixels
    near_misses = diff <= threshold
    foreground_near_misses = near_misses & target_foreground_mask

    # Calculate accuracy as: near-miss foreground pixels / total foreground pixels
    foreground_accuracy = (
        foreground_near_misses.sum().float() / target_foreground_mask.sum().float()
    )

    return foreground_accuracy.item()


def evaluate_model(
    model: SimpleARCModel, data_loader: DataLoader, config: Config
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        config: Configuration object

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    total_samples = 0
    perfect_matches = 0
    pixel_correct = 0
    near_miss_correct = 0

    # foreground metrics
    perfect_matches_foreground = 0
    pixel_correct_foreground = 0
    near_miss_correct_foreground = 0

    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # Forward pass
            logits = model(
                batch["example1_input"],
                batch["example1_output"],
                batch["example2_input"],
                batch["example2_output"],
                batch["target_input"],
            )

            # Calculate metrics
            batch_size = logits.size(0)
            total_samples += batch_size

            # Perfect accuracy
            perfect_matches += (
                calculate_perfect_accuracy(logits, batch["target_output"]) * batch_size
            )

            # Pixel accuracy
            pixel_correct += (
                calculate_pixel_accuracy(logits, batch["target_output"]) * batch_size
            )

            # Near-miss accuracy
            near_miss_correct += (
                calculate_near_miss_accuracy(logits, batch["target_output"])
                * batch_size
            )

            # Foreground metrics
            perfect_matches_foreground += (
                calculate_perfect_accuracy_foreground(logits, batch["target_output"])
                * batch_size
            )

            pixel_correct_foreground += (
                calculate_pixel_accuracy_foreground(logits, batch["target_output"])
                * batch_size
            )

            near_miss_correct_foreground += (
                calculate_near_miss_accuracy_foreground(logits, batch["target_output"])
                * batch_size
            )

    # Calculate final metrics
    metrics = {
        "perfect_accuracy": perfect_matches / total_samples,
        "pixel_accuracy": pixel_correct / total_samples,
        "near_miss_accuracy": near_miss_correct / total_samples,
        "total_samples": total_samples,
        # Foreground metrics
        "perfect_accuracy_foreground": perfect_matches_foreground / total_samples,
        "pixel_accuracy_foreground": pixel_correct_foreground / total_samples,
        "near_miss_accuracy_foreground": near_miss_correct_foreground / total_samples,
    }

    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate SimpleARC model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override config with command line arguments
    if args.data_dir:
        config.processed_dir = args.data_dir
    if args.batch_size:
        config.batch_size = args.batch_size

    print("Evaluation Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Data directory: {config.processed_dir}")
    print(f"  Checkpoint: {args.checkpoint}")

    # Load dataset
    dataset = ARCDataset(config.processed_dir, config)
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Loaded dataset with {len(dataset)} samples")

    # Create model
    model = SimpleARCModel(config)

    # Load checkpoint
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    # Load checkpoint with weights_only=False to allow custom objects
    checkpoint = torch.load(
        args.checkpoint, map_location=config.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, data_loader, config)

    # Print results
    print("\nEvaluation Results:")
    print(
        f"  Perfect Accuracy: {metrics['perfect_accuracy']:.4f} ({metrics['perfect_accuracy']*100:.2f}%)"
    )
    print(
        f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f} ({metrics['pixel_accuracy']*100:.2f}%)"
    )
    print(
        f"  Near-Miss Accuracy: {metrics['near_miss_accuracy']:.4f} ({metrics['near_miss_accuracy']*100:.2f}%)"
    )
    print(f"  Total Samples: {metrics['total_samples']}")

    print("\nForeground Results (non-background pixels only):")
    print(
        f"  Perfect Accuracy (foreground): {metrics['perfect_accuracy_foreground']:.4f} ({metrics['perfect_accuracy_foreground']*100:.2f}%)"
    )
    print(
        f"  Pixel Accuracy (foreground): {metrics['pixel_accuracy_foreground']:.4f} ({metrics['pixel_accuracy_foreground']*100:.2f}%)"
    )
    print(
        f"  Near-Miss Accuracy (foreground): {metrics['near_miss_accuracy_foreground']:.4f} ({metrics['near_miss_accuracy_foreground']*100:.2f}%)"
    )

    # Save results
    results_file = Path("logs/evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
