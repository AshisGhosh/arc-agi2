"""
Color augmentation setup utilities for ARC datasets.

This module provides functions for setting up color augmentation
that work with both ResNet and Patch model datasets.
"""

import copy
from typing import List, Dict, Any
from .augmentation import generate_augmented_examples


def setup_color_augmentation(tasks: List[Dict[str, Any]], config) -> None:
    """
    Set up color augmentation for all tasks.

    Args:
        tasks: List of task dictionaries
        config: Configuration object with augmentation settings
    """
    for task_idx, task in enumerate(tasks):
        # Get original examples and preserve them
        original_examples = task["train"]
        task["original_train"] = copy.deepcopy(original_examples)

        # Generate augmented examples if enabled
        if config.use_color_relabeling:
            augmented_examples = generate_augmented_examples(
                original_examples,
                num_variants=config.augmentation_variants,
                preserve_background=config.preserve_background,
                seed=config.random_seed + task_idx,  # Different seed per task
            )
            # Store augmented examples in task
            task["augmented_train"] = augmented_examples


def get_augmentation_group(
    task: Dict[str, Any], is_counterfactual: bool, i: int, j: int
) -> int:
    """
    Get augmentation group ID for regularization.

    Groups:
    0: original examples (no augmentation)
    1: color-relabeled examples only
    2: counterfactual examples only
    3: counterfactual + color-relabeled examples

    Args:
        task: Task dictionary
        is_counterfactual: Whether this is a counterfactual combination
        i, j: Pair indices for the combination

    Returns:
        Augmentation group ID (0-3)
    """
    # determine if this combination uses augmented examples
    is_augmented = False

    if is_counterfactual:
        # for counterfactual, check if we're using counterfactual augmented examples
        if (
            task.get("counterfactual_augmented_train")
            and i >= len(task["counterfactual_train"])
            and j >= len(task["counterfactual_train"])
        ):
            is_augmented = True
    else:
        # for regular, check if we're using augmented examples
        if (
            task.get("augmented_train")
            and i >= len(task["train"])
            and j >= len(task["train"])
        ):
            is_augmented = True

    # determine group
    if is_counterfactual and is_augmented:
        return 3  # counterfactual + color
    elif is_counterfactual:
        return 2  # counterfactual only
    elif is_augmented:
        return 1  # color only
    else:
        return 0  # original
