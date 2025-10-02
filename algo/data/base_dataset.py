"""
Base dataset class with common functionality for ARC tasks.

This module provides the abstract base class and common utilities
that are shared between ResNet and Patch model datasets.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import copy
from abc import ABC, abstractmethod

from ..config import Config
from .preprocessing import preprocess_rgb_image, preprocess_grid_image
from .counterfactuals import apply_counterfactual_transform


class BaseARCDataset(Dataset, ABC):
    """
    Abstract base class for ARC datasets with common functionality.

    Provides shared methods for data loading, preprocessing, and validation
    that are used by both ResNet and Patch model datasets.
    """

    def __init__(
        self,
        raw_data_dir: str,
        config: Config,
        holdout: bool = False,
        use_first_combination_only: bool = False,
        require_multiple_test_pairs: bool = False,
    ):
        """
        Initialize base dataset.

        Args:
            raw_data_dir: Directory containing raw JSON task files
            config: Configuration object
            holdout: If True, hold out last train example for validation
            use_first_combination_only: If True, always use first combination (for evaluation)
            require_multiple_test_pairs: If True, only include tasks with multiple test pairs
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.config = config
        self.holdout = holdout
        self.use_first_combination_only = use_first_combination_only
        self.require_multiple_test_pairs = require_multiple_test_pairs

        # Load raw tasks
        self.tasks = self._load_raw_tasks()

        # Filter tasks with sufficient examples
        self.valid_tasks = self._filter_valid_tasks()

    def _load_raw_tasks(self) -> List[Dict[str, Any]]:
        """Load raw JSON task files."""
        tasks = []
        task_files = list(self.raw_data_dir.glob("*.json"))

        for task_file in task_files:
            try:
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    task_data["task_id"] = task_file.stem
                    tasks.append(task_data)
            except Exception as e:
                print(f"Error loading {task_file}: {e}")
                continue

        return tasks

    def _filter_valid_tasks(self) -> List[int]:
        """Filter tasks with sufficient examples for training."""
        valid_indices = []
        for i, task in enumerate(self.tasks):
            # Basic requirements - need at least 2 training examples
            if self.holdout:
                # For holdout mode, need at least 3 training examples
                if len(task["train"]) >= 3:
                    valid_indices.append(i)
            else:
                # For regular mode, need at least 2 training examples
                if len(task["train"]) >= 2:
                    valid_indices.append(i)

        # Apply multiple test pairs filter if requested
        if self.require_multiple_test_pairs:
            multiple_test_indices = self._filter_tasks_with_multiple_test_pairs()
            valid_indices = [
                idx for idx in valid_indices if idx in multiple_test_indices
            ]

        return valid_indices

    def _filter_tasks_with_multiple_test_pairs(self) -> List[int]:
        """Filter tasks that have multiple test examples."""
        multiple_test_indices = []
        for i, task in enumerate(self.tasks):
            # Get test examples (both regular and counterfactual)
            test_examples = self._get_test_examples(task, is_counterfactual=False)

            # Check if task has multiple test examples
            if len(test_examples) > 1:
                multiple_test_indices.append(i)

        return multiple_test_indices

    def _preprocess_rgb(
        self,
        example: Dict[str, Any],
        apply_counterfactual: bool = False,
        counterfactual_type: str = "Y",
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single example to RGB format (input/output pair)."""
        input_tensor = preprocess_rgb_image(example["input"], self.config)
        output_tensor = preprocess_rgb_image(example["output"], self.config)

        # Apply counterfactual transformation after preprocessing if requested
        if apply_counterfactual:
            if counterfactual_type == "Y":
                output_tensor = apply_counterfactual_transform(
                    output_tensor, self.config.counterfactual_transform
                )
            elif counterfactual_type == "X":
                input_tensor = apply_counterfactual_transform(
                    input_tensor, self.config.counterfactual_transform
                )

        return {
            "input": input_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
            "output": output_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
        }

    def _preprocess_grid(
        self,
        example: Dict[str, Any],
        apply_counterfactual: bool = False,
        counterfactual_type: str = "Y",
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single example to grid format (input/output pair)."""
        input_tensor = preprocess_grid_image(example["input"], self.config)
        output_tensor = preprocess_grid_image(example["output"], self.config)

        # Apply counterfactual transformation after preprocessing if requested
        if apply_counterfactual:
            if counterfactual_type == "Y":
                output_tensor = apply_counterfactual_transform(
                    output_tensor, self.config.counterfactual_transform
                )
            elif counterfactual_type == "X":
                input_tensor = apply_counterfactual_transform(
                    input_tensor, self.config.counterfactual_transform
                )

        return {
            "input": input_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
            "output": output_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
        }

    @abstractmethod
    def _get_test_examples(
        self, task: Dict[str, Any], is_counterfactual: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """Get all test examples."""
        pass

    def get_all_training_examples_for_task(
        self, task_idx: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Get all training examples for a specific task (excluding holdout)."""
        task = self.tasks[task_idx]
        training_examples = []

        # Get original training examples (excluding holdout if holdout mode is enabled)
        if self.holdout and len(task["train"]) > 2:
            # Exclude the last training example (holdout) from training
            train_examples = task["train"][:-1]
        else:
            train_examples = task["train"]

        for example in train_examples:
            training_examples.append(self._preprocess_grid(example))

        # Add augmented examples if available
        if self.config.use_color_relabeling and "augmented_train" in task:
            for example in task["augmented_train"]:
                training_examples.append(self._preprocess_grid(example))

        # Add counterfactual examples if available
        if self.config.enable_counterfactuals:
            # Add Y counterfactuals (output transformation)
            if self.config.counterfactual_Y and "counterfactual_train" in task:
                for example in task["counterfactual_train"]:
                    training_examples.append(
                        self._preprocess_grid(
                            example, apply_counterfactual=True, counterfactual_type="Y"
                        )
                    )

            # Add X counterfactuals (input transformation)
            if self.config.counterfactual_X and "counterfactual_X_train" in task:
                for example in task["counterfactual_X_train"]:
                    training_examples.append(
                        self._preprocess_grid(
                            example, apply_counterfactual=True, counterfactual_type="X"
                        )
                    )

        return training_examples

    def _get_task_from_examples(self, all_examples: List[Dict[str, Any]]) -> int:
        """Helper method to get task index from all_examples list."""
        # This is a bit of a hack, but we can find the task by looking at the examples
        # We'll search through all tasks to find one that matches
        for task_idx, task in enumerate(self.tasks):
            if len(task.get("train", [])) > 0:
                # Check if the first example matches
                if all_examples[0] == task["train"][0]:
                    return task_idx
        return 0  # Fallback to first task

    def _get_example_by_index(
        self, all_examples: List[Dict[str, Any]], idx: int, counterfactual_type: str
    ) -> Dict[str, torch.Tensor]:
        """Get example by index, handling both training and test examples."""
        # Check if idx is a test example (negative index)
        if idx < 0:
            # This is a test example, get it from the task's test examples
            task_idx = self._get_task_from_examples(all_examples)
            task = self.tasks[task_idx]
            test_examples = task.get("test", [])
            test_idx = -(idx + 1)  # Convert -1 to 0, -2 to 1, etc.

            if test_idx < len(test_examples):
                test_example = test_examples[test_idx]
                if counterfactual_type != "original":
                    # For counterfactual combinations, create counterfactual version
                    ex = copy.deepcopy(test_example)
                    ex_processed = self._preprocess_grid(
                        ex,
                        apply_counterfactual=True,
                        counterfactual_type=counterfactual_type,
                    )
                    return ex_processed
                else:
                    # For non-counterfactual combinations, use original test example
                    return self._preprocess_grid(test_example)
            else:
                # Fallback to first test example if index is out of range
                test_example = test_examples[0]
                return self._preprocess_grid(test_example)
        else:
            # This is a training example (positive index)
            if counterfactual_type != "original":
                # For counterfactual combinations, create counterfactual version
                ex = copy.deepcopy(all_examples[idx])
                ex_processed = self._preprocess_grid(
                    ex,
                    apply_counterfactual=True,
                    counterfactual_type=counterfactual_type,
                )
                return ex_processed
            else:
                # For non-counterfactual combinations, use original example
                return self._preprocess_grid(all_examples[idx])

    def _create_target_example(
        self,
        all_examples: List[Dict[str, Any]],
        i: int,
        j: int,
        k: int,
        counterfactual_type: str,
    ) -> Dict[str, torch.Tensor]:
        """Create target example from example k (grayscale grid format)."""
        # Check if k is a test example (negative index)
        if k < 0:
            # This is a test example, get it from the task's test examples
            task_idx = self._get_task_from_examples(all_examples)
            task = self.tasks[task_idx]
            test_examples = task.get("test", [])
            test_idx = -(k + 1)  # Convert -1 to 0, -2 to 1, etc.

            if test_idx < len(test_examples):
                test_example = test_examples[test_idx]
                if counterfactual_type != "original":
                    # For counterfactual combinations, create counterfactual version
                    ex = copy.deepcopy(test_example)
                    ex_processed = self._preprocess_grid(
                        ex,
                        apply_counterfactual=True,
                        counterfactual_type=counterfactual_type,
                    )
                    return ex_processed
                else:
                    # For non-counterfactual combinations, use original test example
                    return self._preprocess_grid(test_example)
            else:
                # Fallback to first test example if index is out of range
                test_example = test_examples[0]
                return self._preprocess_grid(test_example)
        else:
            # This is a training example (positive index)
            if counterfactual_type != "original":
                # For counterfactual combinations, create counterfactual version
                ex = copy.deepcopy(all_examples[k])
                ex_processed = self._preprocess_grid(
                    ex,
                    apply_counterfactual=True,
                    counterfactual_type=counterfactual_type,
                )
                return ex_processed
            else:
                # For non-counterfactual combinations, use original example
                return self._preprocess_grid(all_examples[k])

    def _generate_combinations(self) -> List[List[Tuple[int, int, int, str]]]:
        """Generate cycling combinations: (S_A, S_B, T_1, counterfactual_type)."""
        combinations = []

        for task_idx, task in enumerate(self.tasks):
            # Get original examples (already preserved in setup_color_augmentation)
            original_examples = task["train"]

            # Calculate total examples (including augmented if available)
            if self.config.use_color_relabeling and "augmented_train" in task:
                total_examples = len(original_examples) + len(task["augmented_train"])
            else:
                total_examples = len(original_examples)

            # Generate cycling combinations from all examples
            if total_examples >= 2:  # Need at least 2 training examples for cycling
                # For holdout mode, exclude the last original example from combinations
                if self.holdout and len(original_examples) > 2:
                    # Only use the first (len(original_examples) - 1) examples for combinations
                    # This excludes the holdout example
                    max_original_idx = len(original_examples) - 1
                    if self.config.use_color_relabeling and "augmented_train" in task:
                        # Augmented examples come after original examples
                        max_idx = max_original_idx + len(task["augmented_train"])
                    else:
                        max_idx = max_original_idx
                else:
                    # Use all available examples for combinations
                    max_idx = total_examples

                # Generate cycling combinations that always use test examples as targets
                cycling_combinations = self._generate_cycling_with_test_combinations(
                    range(max_idx), task
                )

                # No separate test combinations needed since cycling already includes test examples
                task_combinations = cycling_combinations

                # Add counterfactual combinations if enabled
                if self.config.enable_counterfactuals:
                    # Start with original combinations
                    all_combinations = [
                        (combo[0], combo[1], combo[2], "original")
                        for combo in task_combinations
                    ]

                    # Add Y counterfactual combinations if enabled
                    if self.config.counterfactual_Y:
                        y_counterfactual_combinations = [
                            (combo[0], combo[1], combo[2], "Y")
                            for combo in task_combinations
                        ]
                        all_combinations.extend(y_counterfactual_combinations)

                    # Add X counterfactual combinations if enabled
                    if self.config.counterfactual_X:
                        x_counterfactual_combinations = [
                            (combo[0], combo[1], combo[2], "X")
                            for combo in task_combinations
                        ]
                        all_combinations.extend(x_counterfactual_combinations)

                    task_combinations = all_combinations
                else:
                    # Original behavior - mark all as original
                    task_combinations = [
                        (combo[0], combo[1], combo[2], "original")
                        for combo in task_combinations
                    ]

                combinations.append(task_combinations)
            else:
                combinations.append([])

        return combinations

    def _generate_cycling_combinations(
        self, indices: range
    ) -> List[Tuple[int, int, int]]:
        """Generate cycling combinations: (S_A, S_B, T_1) where each can be any of the indices."""
        combinations = []
        indices_list = list(indices)

        # Generate all possible 3-combinations
        for i in range(len(indices_list)):
            for j in range(i + 1, len(indices_list)):
                for k in range(j + 1, len(indices_list)):
                    # Add all 3 cycling patterns:
                    # 1. (i, j) -> k
                    # 2. (i, k) -> j
                    # 3. (j, k) -> i
                    combinations.append(
                        (indices_list[i], indices_list[j], indices_list[k])
                    )
                    combinations.append(
                        (indices_list[i], indices_list[k], indices_list[j])
                    )
                    combinations.append(
                        (indices_list[j], indices_list[k], indices_list[i])
                    )

        return combinations

    def _generate_test_combinations(
        self, train_indices: range, task: Dict
    ) -> List[Tuple[int, int, int]]:
        """Generate test combinations: (train_i, train_j) -> test_k."""
        combinations = []
        train_indices_list = list(train_indices)
        test_examples = task.get("test", [])

        if len(test_examples) == 0:
            return combinations

        # Generate all possible pairs of training examples
        for i in range(len(train_indices_list)):
            for j in range(i + 1, len(train_indices_list)):
                # For each test example, create a combination
                for test_idx in range(len(test_examples)):
                    # Use a special index to indicate this is a test example
                    # We'll use negative indices to distinguish from training examples
                    test_target_idx = -(test_idx + 1)  # -1, -2, -3, etc.
                    combinations.append(
                        (train_indices_list[i], train_indices_list[j], test_target_idx)
                    )

        return combinations

    def _generate_cycling_with_test_combinations(
        self, train_indices: range, task: Dict
    ) -> List[Tuple[int, int, int]]:
        """Generate cycling combinations that include test examples: (train_i, train_j) -> test_k, (train_i, test_k) -> train_j, (test_k, train_j) -> train_i."""
        combinations = []
        train_indices_list = list(train_indices)
        test_examples = task.get("test", [])

        if len(test_examples) == 0 or len(train_indices_list) < 2:
            return combinations

        # Generate cycling combinations that include test examples
        for i in range(len(train_indices_list)):
            for j in range(i + 1, len(train_indices_list)):
                for test_idx in range(len(test_examples)):
                    # Use negative indices to indicate test examples
                    test_target_idx = -(test_idx + 1)  # -1, -2, -3, etc.

                    # Add all 3 cycling patterns with test examples:
                    # 1. (train_i, train_j) -> test_k
                    # 2. (train_i, test_k) -> train_j
                    # 3. (test_k, train_j) -> train_i
                    combinations.append(
                        (train_indices_list[i], train_indices_list[j], test_target_idx)
                    )
                    combinations.append(
                        (train_indices_list[i], test_target_idx, train_indices_list[j])
                    )
                    combinations.append(
                        (test_target_idx, train_indices_list[j], train_indices_list[i])
                    )

        return combinations

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        pass
