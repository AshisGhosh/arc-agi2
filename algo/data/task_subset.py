#!/usr/bin/env python3
"""
shared task subset functionality for filtering arc datasets by task indices.

provides a unified way to create task subsets that work with the new batching strategy.
"""

from typing import List
from . import create_dataset


class TaskSubset:
    """
    Task subset that filters by selected task indices.

    Works with both ResNet and Patch model datasets by creating
    the appropriate dataset type and filtering the mapping.
    """

    def __init__(
        self,
        task_indices: List[int],
        config,
        arc_agi1_dir: str,
        holdout: bool = True,
        use_first_combination_only: bool = True,
        require_multiple_test_pairs: bool = False,
        model_type: str = "simple_arc",
    ):
        """
        Create a task subset with only the specified task indices.

        Args:
            task_indices: list of task indices to include in the subset
            config: configuration object
            arc_agi1_dir: path to arc-agi1 dataset directory
            holdout: whether to enable holdout mode
            use_first_combination_only: whether to use only the first combination of each task
            require_multiple_test_pairs: whether to only include tasks with multiple test pairs
            model_type: "simple_arc" for ResNet model, "patch_attention" for patch model
        """
        # Create the appropriate dataset
        self.dataset = create_dataset(
            raw_data_dir=arc_agi1_dir,
            config=config,
            holdout=holdout,
            use_first_combination_only=use_first_combination_only,
            require_multiple_test_pairs=require_multiple_test_pairs,
        )

        # Validate that all selected tasks are valid
        valid_task_indices = set(self.dataset.valid_tasks)
        invalid_tasks = [idx for idx in task_indices if idx not in valid_task_indices]
        if invalid_tasks:
            raise ValueError(
                f"invalid task indices: {invalid_tasks}. valid tasks: {len(valid_task_indices)}"
            )

        # Filter the mapping to only include selected tasks
        self.selected_task_indices = set(task_indices)

        # Both models now use combination_mapping
        self.filtered_combination_mapping = []
        for idx, (task_idx, combo_idx) in enumerate(self.dataset.combination_mapping):
            if task_idx in self.selected_task_indices:
                self.filtered_combination_mapping.append((task_idx, combo_idx))
        self.dataset.combination_mapping = self.filtered_combination_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get sample by index using filtered mapping."""
        return self.dataset[idx]

    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset."""
        return getattr(self.dataset, name)
