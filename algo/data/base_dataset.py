"""
Base dataset class with common functionality for ARC tasks.

This module provides the abstract base class and common utilities
that are shared between ResNet and Patch model datasets.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Any
import json
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

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        pass
