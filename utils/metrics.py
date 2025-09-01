import torch
from typing import Dict


def compute_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """compute cell accuracy and perfect grid accuracy"""
    # predictions: (B, L) or (B, L, num_colors)
    # targets: (B, L)

    if predictions.dim() == 3:
        pred_labels = predictions.argmax(dim=-1)  # (B, L)
    else:
        pred_labels = predictions  # (B, L)

    # cell accuracy (per-position)
    cell_correct = (pred_labels == targets).float()  # (B, L)
    cell_acc = cell_correct.mean().item()

    # perfect grid accuracy (all cells correct)
    perfect_grids = (pred_labels == targets).all(dim=-1).float()  # (B,)
    perfect_grid_acc = perfect_grids.mean().item()

    return {"cell_acc": cell_acc, "perfect_grid_acc": perfect_grid_acc}


def compute_cycle_metrics(
    logits_per_cycle: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """compute metrics for each cycle"""
    # logits_per_cycle: (cycles, B, L, num_colors)
    # targets: (B, L)

    C = logits_per_cycle.shape[0]
    metrics = {}

    for cycle in range(C):
        cycle_logits = logits_per_cycle[cycle]  # (B, L, num_colors)
        cycle_metrics = compute_metrics(cycle_logits, targets)

        metrics[f"cell_acc_cycle_{cycle}"] = cycle_metrics["cell_acc"]
        metrics[f"perfect_grid_acc_cycle_{cycle}"] = cycle_metrics["perfect_grid_acc"]

    # last cycle metrics
    metrics["cell_acc_last"] = metrics[f"cell_acc_cycle_{C - 1}"]
    metrics["perfect_grid_acc_last"] = metrics[f"perfect_grid_acc_cycle_{C - 1}"]

    return metrics
