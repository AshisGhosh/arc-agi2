"""
Color augmentation functions for ARC dataset.

Provides color relabeling functionality to increase training data diversity
while preserving the discrete color structure of ARC tasks.
"""

import numpy as np
from typing import Dict, List, Optional


def generate_color_permutation(
    preserve_background: bool = True, seed: Optional[int] = None
) -> Dict[int, int]:
    """
    Generate deterministic color permutation for ARC colors.

    Args:
        preserve_background: If True, color 0 (background) maps to itself
        seed: Random seed for reproducibility (used for deterministic shuffling)

    Returns:
        Dict mapping old_color -> new_color
    """
    if preserve_background:
        # Use deterministic shuffling based on seed
        colors = list(range(1, 10))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if seed is not None:
            # Use seed to create deterministic shuffle
            rng = np.random.RandomState(seed)
            shuffled = rng.permutation(colors).tolist()
        else:
            # Default: no shuffling
            shuffled = colors

        permutation = {0: 0}  # background unchanged
        for old_color, new_color in zip(colors, shuffled):
            permutation[old_color] = new_color
    else:
        # Use deterministic shuffling for all colors
        colors = list(range(10))
        if seed is not None:
            # Use seed to create deterministic shuffle
            rng = np.random.RandomState(seed)
            shuffled = rng.permutation(colors).tolist()
        else:
            # Default: no shuffling
            shuffled = colors
        permutation = {old: new for old, new in zip(colors, shuffled)}

    return permutation


def apply_color_relabeling(
    image_array: np.ndarray, permutation_map: Dict[int, int]
) -> np.ndarray:
    """
    Apply color relabeling to 2D array with discrete color values.

    Args:
        image_array: 2D numpy array with values 0-9
        permutation_map: Dict mapping old_color -> new_color

    Returns:
        Relabeled 2D array
    """
    result = image_array.copy()
    for old_color, new_color in permutation_map.items():
        result[image_array == old_color] = new_color
    return result


def generate_augmented_examples(
    original_examples: List[Dict],
    num_variants: int = 1,
    preserve_background: bool = True,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generate color-relabeled versions of training examples.

    Args:
        original_examples: List of original training examples
        num_variants: Number of augmented versions per example
        preserve_background: If True, keep background color unchanged
        seed: Random seed for reproducibility

    Returns:
        List of augmented examples
    """
    augmented = []

    for example in original_examples:
        for variant_idx in range(num_variants):
            # Generate unique permutation for each variant using deterministic shuffling
            variant_seed = seed + variant_idx if seed is not None else None
            permutation = generate_color_permutation(preserve_background, variant_seed)

            # Apply relabeling to both input and output
            # Convert to numpy arrays for relabeling, then back to lists for consistency
            input_array = np.array(example["input"])
            output_array = np.array(example["output"])

            aug_input = apply_color_relabeling(input_array, permutation)
            aug_output = apply_color_relabeling(output_array, permutation)

            augmented.append(
                {
                    "input": aug_input.tolist(),  # Convert back to list format
                    "output": aug_output.tolist(),  # Convert back to list format
                }
            )

    return augmented
