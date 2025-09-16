import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def calculate_classification_loss(
    logits: torch.Tensor, target: torch.Tensor, config
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculate cross-entropy loss for pixel-wise classification.

    Args:
        logits: Classification logits [B, 10, 30, 30]
        target: Target color indices [B, 1, 30, 30]
        config: Configuration object

    Returns:
        Tuple of (total_loss, loss_components)
    """
    # Flatten spatial dimensions for cross-entropy
    logits_flat = logits.reshape(logits.size(0), 10, -1)  # [B, 10, 900]
    target_flat = target.reshape(target.size(0), -1)  # [B, 900]

    # Cross-entropy loss
    cross_entropy_loss = F.cross_entropy(logits_flat, target_flat.long())

    # Calculate accuracy for monitoring
    with torch.no_grad():
        predictions = torch.argmax(logits_flat, dim=1)  # [B, 900]
        accuracy = (predictions == target_flat).float().mean()

    loss_components = {
        "cross_entropy_loss": cross_entropy_loss.item(),
        "accuracy": accuracy.item(),
        "total_loss": cross_entropy_loss.item(),
    }

    return cross_entropy_loss, loss_components
