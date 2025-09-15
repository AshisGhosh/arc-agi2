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
        # use the filtered mapping
        task_idx, combination_idx = self.combination_mapping[idx]
        task = self.tasks[task_idx]
        task_combinations = self.combinations[task_idx]
        pair_indices = task_combinations[combination_idx]

        # get all available examples (original + augmented)
        all_examples = task["train"]
        if self.config.use_color_relabeling and "augmented_train" in task:
            all_examples = task["train"] + task["augmented_train"]

        # rule latent inputs (2 examples) - preprocess for resnet
        rule_latent_inputs = [
            self._preprocess_example(all_examples[pair_indices[0]]),
            self._preprocess_example(all_examples[pair_indices[1]]),
        ]

        # holdout target (if enabled and available)
        holdout_target = None
        train_examples_to_use = all_examples  # use all examples (original + augmented)
        if self.holdout and len(task["train"]) > 2:
            # for holdout, use the last original train example (not augmented)
            holdout_target = self._preprocess_target(
                task["train"][-1]
            )  # last original train example
            # remove holdout from original train examples to use
            train_examples_to_use = task["train"][:-1]
            # add augmented examples if available
            if self.config.use_color_relabeling and "augmented_train" in task:
                train_examples_to_use = train_examples_to_use + task["augmented_train"]

        # training targets (all available examples) - preprocess appropriately
        training_targets = []
        for train_example in train_examples_to_use:
            training_targets.append(self._preprocess_target(train_example))
        for test_example in task["test"]:
            training_targets.append(self._preprocess_target(test_example))

        return {
            "rule_latent_inputs": rule_latent_inputs,
            "training_targets": training_targets,
            "holdout_target": holdout_target,
            "combination_info": {
                "task_idx": task_idx,
                "task_id": task["task_id"],
                "combination_idx": combination_idx,
                "pair_indices": pair_indices,
                "total_combinations": len(task_combinations),
            },
        }


def create_task_subset(dataset: ARCDataset, task_indices: List[int]) -> TaskSubset:
    """
    create a task subset from an existing dataset.

    args:
        dataset: the original arc dataset
        task_indices: list of task indices to include in the subset

    returns:
        task subset containing only the specified tasks
    """
    return TaskSubset(
        task_indices=task_indices,
        config=dataset.config,
        arc_agi1_dir=str(dataset.raw_data_dir),
        holdout=dataset.holdout,
        use_first_combination_only=dataset.use_first_combination_only,
    )


def create_task_subset_for_evaluation(
    dataset: ARCDataset, task_indices: List[int]
) -> TaskSubset:
    """
    create a task subset specifically for evaluation (always uses first combination only).

    args:
        dataset: the original arc dataset
        task_indices: list of task indices to include in the subset

    returns:
        task subset for evaluation containing only the specified tasks
    """
    return TaskSubset(
        task_indices=task_indices,
        config=dataset.config,
        arc_agi1_dir=str(dataset.raw_data_dir),
        holdout=True,  # always use holdout for evaluation
        use_first_combination_only=True,  # always use first combination for evaluation
    )
