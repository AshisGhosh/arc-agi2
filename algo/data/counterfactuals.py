"""
Counterfactual transformation utilities for ARC datasets.

This module provides functions for setting up and applying
counterfactual transformations to ARC task examples.
"""

import copy
import torch
import numpy as np
from typing import List, Dict, Any


def setup_counterfactuals(tasks: List[Dict[str, Any]], config) -> None:
    """
    Set up counterfactual examples for all tasks.

    Args:
        tasks: List of task dictionaries
        config: Configuration object with counterfactual settings
    """
    for task in tasks:
        if config.enable_counterfactuals:
            _create_counterfactual_examples(task)


def _create_counterfactual_examples(task: Dict[str, Any]) -> None:
    """
    Pre-generate counterfactual versions of all examples in a task.
    They are transformed after preprocessing to preserve the post padding shape.
    """
    # Create counterfactual training examples
    counterfactual_train = []
    for example in task["train"]:
        cf_example = copy.deepcopy(example)
        cf_example["input"] = example["input"]  # Keep input the same
        # Don't transform output here - we'll do it after preprocessing
        counterfactual_train.append(cf_example)

    # Create counterfactual test examples
    counterfactual_test = []
    for example in task["test"]:
        cf_example = copy.deepcopy(example)
        cf_example["input"] = example["input"]  # Keep input the same
        # Don't transform output here - we'll do it after preprocessing
        counterfactual_test.append(cf_example)

    # Store counterfactual examples in task
    task["counterfactual_train"] = counterfactual_train
    task["counterfactual_test"] = counterfactual_test

    # If color relabeling is enabled, also create counterfactual augmented examples
    if "augmented_train" in task:
        counterfactual_augmented = []
        for example in task["augmented_train"]:
            cf_example = copy.deepcopy(example)
            cf_example["input"] = example["input"]  # Keep input the same
            # Don't transform output here - we'll do it after preprocessing
            counterfactual_augmented.append(cf_example)
        task["counterfactual_augmented_train"] = counterfactual_augmented


def apply_counterfactual_transform(image, transform_type: str):
    """Apply counterfactual transformation to an image."""
    if transform_type == "rotate_90":
        return rotate_90(image)
    elif transform_type == "rotate_180":
        return rotate_180(image)
    elif transform_type == "rotate_270":
        return rotate_270(image)
    elif transform_type == "reflect_h":
        return reflect_horizontal(image)
    elif transform_type == "reflect_v":
        return reflect_vertical(image)
    else:
        raise ValueError(f"Unknown counterfactual transform: {transform_type}")


def rotate_90(image):
    """Rotate image by 90 degrees clockwise."""
    if isinstance(image, torch.Tensor):
        return torch.rot90(image, k=1, dims=[-2, -1])
    else:  # numpy array
        return np.rot90(image, k=1)


def rotate_180(image):
    """Rotate image by 180 degrees."""
    if isinstance(image, torch.Tensor):
        return torch.rot90(image, k=2, dims=[-2, -1])
    else:  # numpy array
        return np.rot90(image, k=2)


def rotate_270(image):
    """Rotate image by 270 degrees clockwise (90 degrees counterclockwise)."""
    if isinstance(image, torch.Tensor):
        return torch.rot90(image, k=3, dims=[-2, -1])
    else:  # numpy array
        return np.rot90(image, k=3)


def reflect_horizontal(image):
    """Reflect image horizontally."""
    if isinstance(image, torch.Tensor):
        return torch.flip(image, dims=[-1])
    else:  # numpy array
        return np.fliplr(image)


def reflect_vertical(image):
    """Reflect image vertically."""
    if isinstance(image, torch.Tensor):
        return torch.flip(image, dims=[-2])
    else:  # numpy array
        return np.flipud(image)
