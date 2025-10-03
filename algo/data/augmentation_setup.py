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
        original_train_examples = task["train"]
        original_test_examples = task["test"]
        task["original_train"] = copy.deepcopy(original_train_examples)
        task["original_test"] = copy.deepcopy(original_test_examples)

        # Generate augmented examples if enabled
        if config.use_color_relabeling:
            # Augment training examples
            augmented_train_examples = generate_augmented_examples(
                original_train_examples,
                num_variants=config.augmentation_variants,
                preserve_background=config.preserve_background,
                seed=config.random_seed + task_idx,  # Different seed per task
            )
            task["augmented_train"] = augmented_train_examples

            # Augment test examples with the SAME color permutation as training
            # This ensures consistency between training and test augmentation
            augmented_test_examples = generate_augmented_examples(
                original_test_examples,
                num_variants=config.augmentation_variants,
                preserve_background=config.preserve_background,
                seed=config.random_seed + task_idx,  # Same seed as training
            )
            task["augmented_test"] = augmented_test_examples


def get_augmentation_group(
    task: Dict[str, Any],
    is_counterfactual: bool,
    i: int,
    j: int,
    counterfactual_type: str = "original",
) -> int:
    """
    Get augmentation group ID for regularization.

    Groups:
    0: original examples (no augmentation)
    1: color-relabeled examples only
    2: X counterfactual examples only
    3: Y counterfactual examples only
    4: X counterfactual + color-relabeled examples
    5: Y counterfactual + color-relabeled examples

    Args:
        task: Task dictionary
        is_counterfactual: Whether this is a counterfactual combination
        i, j: Pair indices for the combination
        counterfactual_type: Type of counterfactual ("original", "X", "Y")

    Returns:
        Augmentation group ID (0-5)
    """
    # determine if this combination uses augmented examples
    is_augmented = False

    if is_counterfactual:
        # for counterfactual, check if we're using counterfactual augmented examples
        if counterfactual_type == "X":
            if (
                task.get("counterfactual_X_augmented_train")
                and i >= len(task["counterfactual_X_train"])
                and j >= len(task["counterfactual_X_train"])
            ):
                is_augmented = True
        elif counterfactual_type == "Y":
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
        if counterfactual_type == "X":
            return 4  # X counterfactual + color
        elif counterfactual_type == "Y":
            return 5  # Y counterfactual + color
        else:
            return 3  # fallback for unknown counterfactual type
    elif is_counterfactual:
        if counterfactual_type == "X":
            return 2  # X counterfactual only
        elif counterfactual_type == "Y":
            return 3  # Y counterfactual only
        else:
            return 2  # fallback for unknown counterfactual type
    elif is_augmented:
        return 1  # color only
    else:
        return 0  # original
