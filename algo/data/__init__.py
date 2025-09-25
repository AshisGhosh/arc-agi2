from .preprocessing import (
    preprocess_rgb_image,
    preprocess_grid_image,
    grayscale_to_rgb,
)
from .collate_functions import unified_collate_fn, flatten_patch_collate_fn
from .resnet_dataset import ResNetARCDataset
from .patch_dataset import PatchARCDataset
from typing import Union


def create_dataset(
    raw_data_dir: str,
    config,
    holdout: bool = False,
    use_first_combination_only: bool = False,
    require_multiple_test_pairs: bool = False,
) -> Union[ResNetARCDataset, PatchARCDataset]:
    """
    Create the appropriate dataset based on model type.

    Args:
        raw_data_dir: Directory containing raw JSON task files
        config: Configuration object with model_type specified
        holdout: If True, hold out last train example for validation
        use_first_combination_only: If True, always use first combination (for evaluation)
        require_multiple_test_pairs: If True, only include tasks with multiple test pairs

    Returns:
        The appropriate dataset instance
    """
    if config.model_type == "patch_attention":
        return PatchARCDataset(
            raw_data_dir=raw_data_dir,
            config=config,
            holdout=holdout,
            use_first_combination_only=use_first_combination_only,
            require_multiple_test_pairs=require_multiple_test_pairs,
        )
    else:
        return ResNetARCDataset(
            raw_data_dir=raw_data_dir,
            config=config,
            holdout=holdout,
            use_first_combination_only=use_first_combination_only,
            require_multiple_test_pairs=require_multiple_test_pairs,
        )


def get_collate_fn(model_type: str, use_flattening: bool = False):
    """
    Get the appropriate collate function for the model type.

    Args:
        model_type: Type of model ("simple_arc" or "patch_attention")
        use_flattening: Whether to use flattening collate for patch models

    Returns:
        The appropriate collate function
    """
    if model_type == "patch_attention" and use_flattening:
        return flatten_patch_collate_fn
    else:
        return unified_collate_fn


__all__ = [
    "preprocess_rgb_image",
    "preprocess_grid_image",
    "grayscale_to_rgb",
    "create_dataset",
    "get_collate_fn",
    "unified_collate_fn",
    "ResNetARCDataset",
    "PatchARCDataset",
]
