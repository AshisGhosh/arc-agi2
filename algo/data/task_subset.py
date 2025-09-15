#!/usr/bin/env python3
"""
shared task subset functionality for filtering arc datasets by task indices.

provides a unified way to create task subsets that work with the new batching strategy.
"""

from typing import List
from algo.data import ARCDataset


class TaskSubset(ARCDataset):
    """
    task subset that filters combinations by selected task indices.

    works with the new batching strategy by filtering the combination_mapping
    to only include combinations from the selected tasks.
    """

    def __init__(
        self,
        task_indices: List[int],
        config,
        arc_agi1_dir: str,
        holdout: bool = True,
        use_first_combination_only: bool = True,
    ):
        """
        create a task subset with only the specified task indices.

        args:
            task_indices: list of task indices to include in the subset
            config: configuration object
            arc_agi1_dir: path to arc-agi1 dataset directory
            holdout: whether to enable holdout mode
            use_first_combination_only: whether to use only the first combination of each task
        """
        # initialize the base dataset
        super().__init__(
            arc_agi1_dir,
            config,
            holdout=holdout,
            use_first_combination_only=use_first_combination_only,
        )

        # validate that all selected tasks are valid
        valid_task_indices = set(self.valid_tasks)
        invalid_tasks = [idx for idx in task_indices if idx not in valid_task_indices]
        if invalid_tasks:
            raise ValueError(
                f"invalid task indices: {invalid_tasks}. valid tasks: {len(valid_task_indices)}"
            )

        # filter the combination mapping to only include selected tasks
        self.selected_task_indices = set(task_indices)
        self.filtered_combination_mapping = []

        for idx, (task_idx, combo_idx) in enumerate(self.combination_mapping):
            if task_idx in self.selected_task_indices:
                self.filtered_combination_mapping.append((task_idx, combo_idx))

        # update the mapping to use filtered indices
        self.combination_mapping = self.filtered_combination_mapping

    def __len__(self):
        return len(self.combination_mapping)

    def __getitem__(self, idx):
        """Get sample by index using filtered combination mapping."""
        # Get combination information using the base class helper
        task_idx, combination_idx, (i, j), is_counterfactual = (
            self._get_combination_info(idx)
        )
        task = self.tasks[task_idx]

        # Get all available examples using the base class helper
        all_examples = self._get_all_examples(task, is_counterfactual)

        # Create rule latent inputs using the base class helper
        rule_latent_inputs = self._create_rule_latent_inputs(
            all_examples, i, j, is_counterfactual
        )

        # Create rule latent targets using the base class helper
        rule_latent_targets = self._create_rule_latent_targets(
            all_examples, i, j, is_counterfactual
        )

        # Create test example using the base class helper
        test_example = self._get_test_example(task, is_counterfactual)

        # Create holdout target using the base class helper
        holdout_example = self._get_holdout_example(task, is_counterfactual)

        return {
            # Core data - only what's needed for this combination
            "rule_latent_examples": rule_latent_inputs,  # 2 examples for encoder (ResNet format)
            "rule_latent_targets": rule_latent_targets,  # 2 examples for decoder (ARC format)
            "test_example": test_example,  # Single test example
            "holdout_example": holdout_example,  # Optional holdout
            # Top-level metadata
            "task_idx": task_idx,  # Top-level task index
            "task_id": task["task_id"],  # Top-level task ID
            "is_counterfactual": is_counterfactual,  # Top-level flag
            "combination_idx": combination_idx,  # Top-level combination index
            "pair_indices": (i, j),  # Top-level pair indices
            "total_combinations": len(self.combinations[task_idx]),  # Top-level count
        }
