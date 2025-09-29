import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List


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
    # Ensure target is on the same device as logits
    target = target.to(logits.device)

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


def calculate_rule_latent_regularization_loss(
    rule_latents: torch.Tensor,  # [B, latent_dim]
    task_indices: List[int],  # [B]
    augmentation_groups: List[int],  # [B]
    regularization_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculate regularization loss to encourage similar rule latents
    for same task + augmentation group combinations.

    Args:
        rule_latents: Rule latent vectors [B, latent_dim]
        task_indices: Task index for each sample [B]
        augmentation_groups: Augmentation group for each sample [B]
        regularization_weight: Weight for the regularization loss

    Returns:
        Tuple of (regularization_loss, loss_components)
    """
    # Group rule latents by (task_idx, augmentation_group)
    groups = {}
    for i, (task_idx, aug_group) in enumerate(zip(task_indices, augmentation_groups)):
        key = (task_idx, aug_group)
        if key not in groups:
            groups[key] = []
        groups[key].append(rule_latents[i])

    # Calculate within-group similarity loss
    total_loss = 0.0
    group_count = 0
    total_pairs = 0

    for group_latents in groups.values():
        if len(group_latents) > 1:  # Need at least 2 samples for regularization
            group_tensor = torch.stack(group_latents)  # [N, latent_dim]

            # Calculate pairwise cosine similarity (want high similarity)
            # Convert to distance: 1 - cosine_similarity
            similarities = F.cosine_similarity(
                group_tensor.unsqueeze(1), group_tensor.unsqueeze(0), dim=2
            )
            # We want similarities close to 1, so loss is 1 - similarity
            distances = 1 - similarities
            # Only consider upper triangle (avoid duplicates and diagonal)
            mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
            group_loss = distances[mask].mean()

            total_loss += group_loss
            group_count += 1
            total_pairs += mask.sum().item()

    # Normalize by number of groups with multiple samples
    if group_count > 0:
        avg_group_loss = total_loss / group_count
        regularization_loss = regularization_weight * avg_group_loss
        avg_group_loss_value = avg_group_loss.item()
    else:
        regularization_loss = torch.tensor(0.0, device=rule_latents.device)
        avg_group_loss_value = 0.0

    loss_components = {
        "rule_latent_regularization": regularization_loss.item(),
        "avg_group_loss": avg_group_loss_value,
        "active_groups": group_count,
        "total_pairs": total_pairs,
    }

    return regularization_loss, loss_components
