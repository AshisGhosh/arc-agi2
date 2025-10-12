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


class ExampleRetriever:
    """Unified interface for retrieving examples with all complexity handled internally."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.config = dataset.config

    def get_example(
        self, task_idx: int, example_idx: int, group_name: str, counterfactual_type: str
    ) -> Dict[str, torch.Tensor]:
        """
        Get an example by index, handling all complexity internally.

        Args:
            task_idx: Index of the task
            example_idx: Index of the example (can be negative for test examples)
            group_name: Augmentation group name ('original' or 'augmented')
            counterfactual_type: Type of counterfactual ('original', 'X', 'Y')

        Returns:
            Preprocessed example as tensor
        """
        task = self.dataset.tasks[task_idx]

        # Handle test examples (negative indices)
        if example_idx < 0:
            return self._get_test_example(
                task, example_idx, group_name, counterfactual_type
            )

        # Handle training examples (positive indices)
        return self._get_training_example(
            task, example_idx, group_name, counterfactual_type
        )

    def _get_test_example(
        self, task: Dict, example_idx: int, group_name: str, counterfactual_type: str
    ) -> Dict[str, torch.Tensor]:
        """Get test example with proper group selection."""
        test_idx = -(example_idx + 1)  # Convert -1 to 0, -2 to 1, etc.

        # Select correct test examples based on group
        if (
            group_name == "augmented"
            and self.config.use_color_relabeling
            and "augmented_test" in task
        ):
            test_examples = task["augmented_test"]
        else:
            test_examples = task.get("test", [])

        # Get the test example
        if test_idx < len(test_examples):
            test_example = test_examples[test_idx]
        else:
            # Fallback to first test example
            test_example = test_examples[0]

        # Apply counterfactual transformation if needed
        if counterfactual_type != "original":
            ex = copy.deepcopy(test_example)
            return self.dataset._preprocess_grid(
                ex, apply_counterfactual=True, counterfactual_type=counterfactual_type
            )
        else:
            return self.dataset._preprocess_grid(test_example)

    def _get_training_example(
        self, task: Dict, example_idx: int, group_name: str, counterfactual_type: str
    ) -> Dict[str, torch.Tensor]:
        """Get training example from the correct group."""
        # Get the correct group's examples
        groups = self.dataset._get_examples_by_augmentation_group(
            task, counterfactual_type
        )
        group_examples = groups.get(group_name, [])

        # Adjust index if needed (for augmented groups)
        if group_name == "augmented":
            original_size = len(groups.get("original", []))
            adjusted_idx = example_idx - original_size
        else:
            adjusted_idx = example_idx

        # Get the training example
        if 0 <= adjusted_idx < len(group_examples):
            training_example = group_examples[adjusted_idx]
        else:
            # Fallback to first example in group
            training_example = group_examples[0]

        # Apply counterfactual transformation if needed
        if counterfactual_type != "original":
            ex = copy.deepcopy(training_example)
            return self.dataset._preprocess_grid(
                ex, apply_counterfactual=True, counterfactual_type=counterfactual_type
            )
        else:
            return self.dataset._preprocess_grid(training_example)

    def get_raw_example(
        self, task_idx: int, example_idx: int, group_name: str, counterfactual_type: str
    ) -> Dict[str, Any]:
        """
        Get raw example (unprocessed) by index.

        Args:
            task_idx: Index of the task
            example_idx: Index of the example (can be negative for test examples)
            group_name: Augmentation group name ('original' or 'augmented')
            counterfactual_type: Type of counterfactual ('original', 'X', 'Y')

        Returns:
            Raw example as dictionary
        """
        task = self.dataset.tasks[task_idx]

        # Handle test examples (negative indices)
        if example_idx < 0:
            test_idx = -(example_idx + 1)

            # Select correct test examples based on group
            if (
                group_name == "augmented"
                and self.config.use_color_relabeling
                and "augmented_test" in task
            ):
                test_examples = task["augmented_test"]
            else:
                test_examples = task.get("test", [])

            if test_idx < len(test_examples):
                return test_examples[test_idx]
            else:
                return test_examples[0]

        # Handle training examples (positive indices)
        groups = self.dataset._get_examples_by_augmentation_group(
            task, counterfactual_type
        )
        group_examples = groups.get(group_name, [])

        # Adjust index if needed (for augmented groups)
        if group_name == "augmented":
            original_size = len(groups.get("original", []))
            adjusted_idx = example_idx - original_size
        else:
            adjusted_idx = example_idx

        if 0 <= adjusted_idx < len(group_examples):
            return group_examples[adjusted_idx]
        else:
            return group_examples[0]

    def determine_group_name(
        self, task_idx: int, i: int, j: int, counterfactual_type: str
    ) -> str:
        """
        Determine the augmentation group name based on indices.

        Args:
            task_idx: Index of the task
            i, j: Training example indices (only these matter for group determination)
            counterfactual_type: Type of counterfactual

        Returns:
            Group name ('original' or 'augmented')
        """
        task = self.dataset.tasks[task_idx]
        groups = self.dataset._get_examples_by_augmentation_group(
            task, counterfactual_type
        )

        original_size = len(groups.get("original", []))
        augmented_size = len(groups.get("augmented", []))

        # Only check training example indices (i, j) - ignore test examples (negative k)
        has_augmented_training = (
            i >= 0 and i >= original_size and i < (original_size + augmented_size)
        ) or (j >= 0 and j >= original_size and j < (original_size + augmented_size))

        return "augmented" if has_augmented_training else "original"


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

    def _get_examples_by_augmentation_group(
        self, task: Dict[str, Any], counterfactual_type: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get examples separated by augmentation group to avoid mixing groups in combinations."""
        groups = {}

        # Start with training examples, excluding holdout if holdout mode is enabled
        if self.holdout and len(task["train"]) > 2:
            # Exclude the last training example (holdout) from rule latent creation
            original_examples = task["train"][:-1]
        else:
            original_examples = task["train"]

        # Group 0: Original training examples only (no test examples in groups)
        groups["original"] = original_examples.copy()

        # Group 1: Color-augmented training examples only (if available)
        if self.config.use_color_relabeling and "augmented_train" in task:
            groups["augmented"] = task["augmented_train"].copy()
        else:
            groups["augmented"] = []

        # Handle counterfactual groups
        if counterfactual_type == "Y":
            groups["counterfactual"] = task["counterfactual_train"].copy()
            if (
                self.config.use_color_relabeling
                and "counterfactual_augmented_train" in task
            ):
                groups["counterfactual_augmented"] = task[
                    "counterfactual_augmented_train"
                ].copy()
            else:
                groups["counterfactual_augmented"] = []
        elif counterfactual_type == "X":
            groups["counterfactual"] = task["counterfactual_X_train"].copy()
            if (
                self.config.use_color_relabeling
                and "counterfactual_X_augmented_train" in task
            ):
                groups["counterfactual_augmented"] = task[
                    "counterfactual_X_augmented_train"
                ].copy()
            else:
                groups["counterfactual_augmented"] = []
        else:
            groups["counterfactual"] = []
            groups["counterfactual_augmented"] = []

        return groups

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
        self,
        all_examples: List[Dict[str, Any]],
        idx: int,
        counterfactual_type: str,
        group_name: str = None,
        task_idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Get example by index, handling both training and test examples."""
        # Check if idx is a test example (negative index)
        if idx < 0:
            # This is a test example, get it from the task's test examples
            if task_idx is None:
                task_idx = self._get_task_from_examples(all_examples)
            task = self.tasks[task_idx]
            test_idx = -(idx + 1)  # Convert -1 to 0, -2 to 1, etc.

            # Determine which test examples to use based on the group
            if (
                group_name == "augmented"
                and self.config.use_color_relabeling
                and "augmented_test" in task
            ):
                test_examples = task["augmented_test"]
            else:
                test_examples = task.get("test", [])

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
                    # For non-counterfactual combinations, use test example as-is
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
        group_name: str = None,
        task_idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Create target example from example k (grayscale grid format)."""
        # Check if k is a test example (negative index)
        if k < 0:
            # This is a test example, get it from the task's test examples
            if task_idx is None:
                task_idx = self._get_task_from_examples(all_examples)
            task = self.tasks[task_idx]
            test_idx = -(k + 1)  # Convert -1 to 0, -2 to 1, etc.

            # Determine which test examples to use based on the augmentation group
            # Use the provided group_name if available, otherwise try to determine from indices
            if group_name is not None:
                is_augmented_group = group_name == "augmented"
            else:
                # Fallback: try to determine from indices (this may not work with adjusted indices)
                groups = self._get_examples_by_augmentation_group(
                    task, counterfactual_type
                )
                original_size = len(groups.get("original", []))
                augmented_size = len(groups.get("augmented", []))

                is_augmented_group = (
                    i >= original_size
                    and j >= original_size
                    and i < (original_size + augmented_size)
                    and j < (original_size + augmented_size)
                )

            if (
                is_augmented_group
                and self.config.use_color_relabeling
                and "augmented_test" in task
            ):
                # Use augmented test examples for augmented group combinations
                test_examples = task["augmented_test"]
            else:
                # Use original test examples for original group combinations
                test_examples = task.get("test", [])

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
                    # For non-counterfactual combinations, use test example as-is
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
        """Generate cycling combinations: (S_A, S_B, T_1, counterfactual_type) with separate augmentation groups."""
        combinations = []

        for task_idx, task in enumerate(self.tasks):
            task_combinations = []

            # Generate combinations for each counterfactual type
            counterfactual_types = ["original"]
            if self.config.enable_counterfactuals:
                if self.config.counterfactual_Y:
                    counterfactual_types.append("Y")
                if self.config.counterfactual_X:
                    counterfactual_types.append("X")

            for counterfactual_type in counterfactual_types:
                # Get examples separated by augmentation group
                groups = self._get_examples_by_augmentation_group(
                    task, counterfactual_type
                )

                # Generate combinations for each group separately
                for group_name, group_examples in groups.items():
                    if len(group_examples) >= 2:  # Need at least 2 examples for cycling
                        if self.config.use_cycling:
                            # Generate cycling combinations for this group
                            group_combinations = (
                                self._generate_cycling_with_test_combinations_for_group(
                                    group_examples,
                                    task,
                                    group_name,
                                    counterfactual_type,
                                )
                            )
                        else:
                            # Generate simple (A, B) -> T combinations only
                            group_combinations = (
                                self._generate_simple_combinations_for_group(
                                    group_examples,
                                    task,
                                    group_name,
                                    counterfactual_type,
                                )
                            )
                        task_combinations.extend(group_combinations)

            # Remove duplicates while preserving order
            seen = set()
            unique_combinations = []
            for combo in task_combinations:
                if combo not in seen:
                    seen.add(combo)
                    unique_combinations.append(combo)

            combinations.append(unique_combinations)

        return combinations

    def _generate_cycling_with_test_combinations_for_group(
        self,
        group_examples: List[Dict[str, Any]],
        task: Dict,
        group_name: str,
        counterfactual_type: str,
    ) -> List[Tuple[int, int, int, str]]:
        """Generate cycling combinations for a specific augmentation group."""
        combinations = []
        test_examples = task.get("test", [])

        if len(test_examples) == 0 or len(group_examples) < 2:
            return combinations

        # Calculate the starting index for this group to avoid conflicts
        # Original group starts at 0, augmented group starts after original group
        if group_name == "original":
            start_idx = 0
        elif group_name == "augmented":
            # Get the size of the original group to offset augmented indices
            original_groups = self._get_examples_by_augmentation_group(
                task, counterfactual_type
            )
            start_idx = len(original_groups.get("original", []))
        else:
            # For counterfactual groups, calculate appropriate offset
            start_idx = 0  # This will be handled by the group separation logic

        # Generate cycling combinations within the group
        # Use only the first pair of training examples to avoid too many combinations
        if len(group_examples) >= 2:
            i, j = 0, 1  # Use first two training examples
            for test_idx in range(len(test_examples)):
                # Use negative indices to indicate test examples
                test_target_idx = -(test_idx + 1)  # -1, -2, -3, etc.

                # Offset the indices to avoid conflicts between groups
                offset_i = start_idx + i
                offset_j = start_idx + j

                # Add all 3 cycling patterns with test examples:
                # 1. (group_i, group_j) -> test_k
                # 2. (group_i, test_k) -> group_j
                # 3. (test_k, group_j) -> group_i
                combinations.append(
                    (offset_i, offset_j, test_target_idx, counterfactual_type)
                )
                combinations.append(
                    (offset_i, test_target_idx, offset_j, counterfactual_type)
                )
                combinations.append(
                    (test_target_idx, offset_j, offset_i, counterfactual_type)
                )

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

    def _generate_simple_combinations_for_group(
        self,
        group_examples: List[Dict[str, Any]],
        task: Dict,
        group_name: str,
        counterfactual_type: str,
    ) -> List[Tuple[int, int, int, str]]:
        """Generate simple (A, B) -> T combinations for a specific augmentation group."""
        combinations = []
        test_examples = task.get("test", [])

        if len(test_examples) == 0 or len(group_examples) < 2:
            return combinations

        # Calculate the starting index for this group to avoid conflicts
        if group_name == "original":
            start_idx = 0
        elif group_name == "augmented":
            # Get the size of the original group to offset augmented indices
            original_groups = self._get_examples_by_augmentation_group(
                task, counterfactual_type
            )
            start_idx = len(original_groups.get("original", []))
        else:
            start_idx = 0

        # Generate simple (A, B) -> T combinations
        # Use only the first pair of training examples to avoid too many combinations
        if len(group_examples) >= 2:
            i, j = 0, 1  # Use first two training examples
            for test_idx in range(len(test_examples)):
                # Use negative indices to indicate test examples
                test_target_idx = -(test_idx + 1)  # -1, -2, -3, etc.

                # Offset the indices to avoid conflicts between groups
                offset_i = start_idx + i
                offset_j = start_idx + j

                # Add simple (A, B) -> T pattern only
                combinations.append(
                    (offset_i, offset_j, test_target_idx, counterfactual_type)
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
