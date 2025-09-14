import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def partial_credit_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_distance: float = 2.0,
    credit_decay: float = 0.5,
) -> torch.Tensor:
    """
    Calculate partial credit loss for near-miss predictions.

    Args:
        pred: Predicted values [B, ...]
        target: Target values [B, ...]
        max_distance: Maximum distance for partial credit
        credit_decay: Decay rate for credit calculation

    Returns:
        Partial credit loss [B, ...]
    """
    diff = torch.abs(pred - target)

    # Perfect match gets 0 loss
    perfect_mask = (diff == 0).float()

    # Near misses get partial credit
    distance_mask = (diff > 0).float()
    credit = torch.exp(-credit_decay * diff) * distance_mask

    # Far misses get full loss
    far_miss_mask = (diff > max_distance).float()

    # Combine all cases
    loss = 0.0 * perfect_mask + (1.0 - credit) * distance_mask + 1.0 * far_miss_mask

    return loss.mean()


def calculate_combined_loss(
    pred: torch.Tensor, target: torch.Tensor, config
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculate combined loss with L1, L2, and partial credit components.

    Args:
        pred: Predicted values [B, 1, 30, 30]
        target: Target values [B, 1, 30, 30]
        config: Configuration object

    Returns:
        Tuple of (total_loss, loss_components)
    """
    # L1 loss
    l1_loss = F.l1_loss(pred, target)

    # L2 loss
    l2_loss = F.mse_loss(pred, target)

    # Partial credit loss
    partial_credit = partial_credit_loss(
        pred,
        target,
        max_distance=config.partial_credit_max_distance,
        credit_decay=config.partial_credit_decay,
    )

    # Combined loss
    total_loss = (
        config.l1_weight * l1_loss
        + config.l2_weight * l2_loss
        + config.partial_credit_weight * partial_credit
    )

    loss_components = {
        "l1_loss": l1_loss.item(),
        "l2_loss": l2_loss.item(),
        "partial_credit": partial_credit.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_components
