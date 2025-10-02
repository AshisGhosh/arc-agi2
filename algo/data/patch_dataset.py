"""
Patch-specific dataset implementation.

This module provides the PatchARCDataset class that handles
combination-based training for patch-based cross-attention models.
"""

import copy
import torch
from typing import List, Dict, Any, Tuple

from .base_dataset import BaseARCDataset
from .augmentation_setup import setup_color_augmentation, get_augmentation_group
from .counterfactuals import setup_counterfactuals


class PatchARCDataset(BaseARCDataset):
    """
    Dataset for patch-based cross-attention models with combination-based training.

    Uses the same combination-based approach as ResNet for consistency.
    Each combination contains support examples from training and test examples.
    """

    def __init__(
        self,
        raw_data_dir: str,
        config,
        holdout: bool = False,
        use_first_combination_only: bool = False,
        require_multiple_test_pairs: bool = False,
    ):
        """
        Initialize Patch dataset.

        Args:
            raw_data_dir: Directory containing raw JSON task files
            config: Configuration object
            holdout: If True, hold out last train example for validation
            use_first_combination_only: If True, always use first combination (for evaluation)
            require_multiple_test_pairs: If True, only include tasks with multiple test pairs
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            config=config,
            holdout=holdout,
            use_first_combination_only=use_first_combination_only,
            require_multiple_test_pairs=require_multiple_test_pairs,
        )

        # Set up augmentation and counterfactuals
        setup_color_augmentation(self.tasks, self.config)
        setup_counterfactuals(self.tasks, self.config)

        # Generate combinations and create mapping (same as ResNet)
        self.combinations = self._generate_combinations()
        self._create_combination_mapping()

    def _create_combination_mapping(self):
        """Create mapping from linear index to (task_idx, combination_idx) pairs (same as ResNet)."""
        self.combination_mapping = []
        self.task_start_indices = {}

        for task_idx in self.valid_tasks:
            task_combinations = self.combinations[task_idx]
            self.task_start_indices[task_idx] = len(self.combination_mapping)

            if self.use_first_combination_only:
                # For evaluation mode, only use the first combination of each task
                if len(task_combinations) > 0:
                    self.combination_mapping.append((task_idx, 0))
            else:
                # For training mode, use all combinations
                for combo_idx in range(len(task_combinations)):
                    self.combination_mapping.append((task_idx, combo_idx))

    def __len__(self) -> int:
        """Return total number of combinations across all valid tasks (same as ResNet)."""
        return len(self.combination_mapping)

    def _get_combination_info(
        self, idx: int
    ) -> Tuple[int, int, Tuple[int, int, int], str]:
        """Get task index, combination index, cycling indices, and counterfactual type (same as ResNet)."""
        task_idx, combination_idx = self.combination_mapping[idx]
        task_combinations = self.combinations[task_idx]
        combination = task_combinations[combination_idx]

        # Check if this is a counterfactual combination
        if len(combination) == 4:  # (i, j, k, counterfactual_type)
            i, j, k, counterfactual_type = combination
        else:
            i, j, k = combination
            counterfactual_type = "original"

        return task_idx, combination_idx, (i, j, k), counterfactual_type

    def _get_all_examples(
        self, task: Dict[str, Any], counterfactual_type: str
    ) -> List[Dict[str, Any]]:
        """Get all available examples (original + augmented + counterfactual if applicable) (same as ResNet)."""
        # Start with training examples, excluding holdout if holdout mode is enabled
        if self.holdout and len(task["train"]) > 2:
            # Exclude the last training example (holdout) from support creation
            all_examples = task["train"][:-1]
        else:
            all_examples = task["train"]

        if self.config.use_color_relabeling and "augmented_train" in task:
            all_examples = all_examples + task["augmented_train"]

        if counterfactual_type == "Y":
            all_examples = all_examples + task["counterfactual_train"]
            if (
                self.config.use_color_relabeling
                and "counterfactual_augmented_train" in task
            ):
                all_examples = all_examples + task["counterfactual_augmented_train"]
        elif counterfactual_type == "X":
            all_examples = all_examples + task["counterfactual_X_train"]
            if (
                self.config.use_color_relabeling
                and "counterfactual_X_augmented_train" in task
            ):
                all_examples = all_examples + task["counterfactual_X_augmented_train"]

        return all_examples

    def _create_support_examples(
        self,
        all_examples: List[Dict[str, Any]],
        i: int,
        j: int,
        k: int,
        counterfactual_type: str,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create support examples from examples i and j (same as ResNet - grayscale format)."""
        # Get the actual examples for i and j, handling test examples
        ex1 = self._get_example_by_index(all_examples, i, counterfactual_type)
        ex2 = self._get_example_by_index(all_examples, j, counterfactual_type)

        return [ex1, ex2]

    def _create_support_targets(
        self,
        all_examples: List[Dict[str, Any]],
        i: int,
        j: int,
        k: int,
        counterfactual_type: str,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create support targets (ARC format) from examples i and j for decoder (same as ResNet)."""
        if counterfactual_type != "original":
            # For counterfactual combinations, create counterfactual versions
            ex1 = copy.deepcopy(all_examples[i])
            ex2 = copy.deepcopy(all_examples[j])

            # Preprocess with counterfactual transformation for decoder
            ex1_processed = self._preprocess_grid(
                ex1, apply_counterfactual=True, counterfactual_type=counterfactual_type
            )
            ex2_processed = self._preprocess_grid(
                ex2, apply_counterfactual=True, counterfactual_type=counterfactual_type
            )

            return [ex1_processed, ex2_processed]
        else:
            # For non-counterfactual combinations, use original examples
            return [
                self._preprocess_grid(all_examples[i]),
                self._preprocess_grid(all_examples[j]),
            ]

    def _get_test_examples(
        self, task: Dict[str, Any], counterfactual_type: str
    ) -> List[Dict[str, torch.Tensor]]:
        """Get all test examples."""
        if (
            counterfactual_type == "Y"
            and task.get("counterfactual_test")
            and len(task["counterfactual_test"]) > 0
        ):
            test_examples = []
            for test_example in task["counterfactual_test"]:
                test_examples.append(
                    self._preprocess_grid(
                        test_example, apply_counterfactual=True, counterfactual_type="Y"
                    )
                )
            return test_examples
        elif (
            counterfactual_type == "X"
            and task.get("counterfactual_X_test")
            and len(task["counterfactual_X_test"]) > 0
        ):
            test_examples = []
            for test_example in task["counterfactual_X_test"]:
                test_examples.append(
                    self._preprocess_grid(
                        test_example, apply_counterfactual=True, counterfactual_type="X"
                    )
                )
            return test_examples
        else:
            test_examples = []
            for test_example in task["test"]:
                test_examples.append(self._preprocess_grid(test_example))
            return test_examples

    def _get_holdout_example(
        self, task: Dict[str, Any], counterfactual_type: str
    ) -> Dict[str, torch.Tensor]:
        """Get holdout example if available (same as ResNet)."""
        if not self.holdout or len(task["train"]) <= 2:
            return None

        if counterfactual_type == "Y" and len(task.get("counterfactual_train", [])) > 2:
            return self._preprocess_grid(
                task["counterfactual_train"][-1],
                apply_counterfactual=True,
                counterfactual_type="Y",
            )
        elif (
            counterfactual_type == "X"
            and len(task.get("counterfactual_X_train", [])) > 2
        ):
            return self._preprocess_grid(
                task["counterfactual_X_train"][-1],
                apply_counterfactual=True,
                counterfactual_type="X",
            )
        else:
            return self._preprocess_grid(task["train"][-1])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index - cycling combination data for patch model (same structure as ResNet)."""
        # Get combination information
        task_idx, combination_idx, (i, j, k), counterfactual_type = (
            self._get_combination_info(idx)
        )
        task = self.tasks[task_idx]

        # Get all available examples
        all_examples = self._get_all_examples(task, counterfactual_type)

        # Create support examples (2 examples for encoder)
        support_examples = self._create_support_examples(
            all_examples, i, j, k, counterfactual_type
        )

        # Create support targets (2 examples for decoder)
        support_targets = self._create_support_targets(
            all_examples, i, j, k, counterfactual_type
        )

        # Create target example (the third example in cycling)
        target_example = self._create_target_example(
            all_examples, i, j, k, counterfactual_type
        )

        # Create test examples (all of them)
        test_examples = self._get_test_examples(task, counterfactual_type)

        # Create holdout example (if available)
        holdout_example = self._get_holdout_example(task, counterfactual_type)

        # Determine augmentation group for regularization (use original function!)
        is_counterfactual = counterfactual_type != "original"
        augmentation_group = get_augmentation_group(
            task, is_counterfactual, i, j, counterfactual_type
        )

        return {
            # Core data - cycling format
            "support_examples": support_examples,  # 2 examples for encoder (grayscale grid format)
            "support_examples_rgb": None,  # No RGB support examples needed for patch model
            "support_targets": support_targets,  # 2 examples for decoder
            "target_example": target_example,  # Target example for cycling (grayscale grid format)
            "test_examples": test_examples,  # All test examples
            "num_test_examples": len(test_examples),  # Number of test examples
            "holdout_example": holdout_example,  # Optional holdout
            # Top-level metadata - same as ResNet
            "task_idx": task_idx,  # Top-level task index
            "task_id": task["task_id"],  # Top-level task ID
            "is_counterfactual": is_counterfactual,  # Top-level flag
            "combination_idx": combination_idx,  # Top-level combination index
            "cycling_indices": (i, j, k),  # Top-level cycling indices
            "total_combinations": len(self.combinations[task_idx]),  # Top-level count
            "augmentation_group": augmentation_group,  # Augmentation group for regularization
        }

    def get_task_combinations(self, task_idx: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get all combinations for a specific task (same as ResNet)."""
        if task_idx not in self.valid_tasks:
            raise ValueError(f"Task {task_idx} not valid")

        task_combinations = self.combinations[task_idx]

        regular_combinations = []
        counterfactual_combinations = []

        for combo_idx, combo in enumerate(task_combinations):
            if len(combo) == 4:
                i, j, k, counterfactual_type = combo
                is_counterfactual = counterfactual_type != "original"
            else:
                i, j, k = combo
                counterfactual_type = "original"
                is_counterfactual = False

            # Get the data for this combination
            linear_idx = None
            for idx, (t_idx, c_idx) in enumerate(self.combination_mapping):
                if t_idx == task_idx and c_idx == combo_idx:
                    linear_idx = idx
                    break

            if linear_idx is not None:
                combination_data = self[linear_idx]

                clean_data = {
                    "combination_idx": combo_idx,
                    "cycling_indices": (i, j, k),
                    "is_counterfactual": is_counterfactual,
                    "counterfactual_type": counterfactual_type,
                    "support_examples": combination_data["support_examples"],
                    "target_example": combination_data["target_example"],
                    "test_examples": combination_data["test_examples"],
                    "num_test_examples": combination_data["num_test_examples"],
                    "holdout_example": combination_data["holdout_example"],
                    "task_id": combination_data["task_id"],
                    "task_idx": combination_data["task_idx"],
                    "total_combinations": combination_data["total_combinations"],
                }

                if is_counterfactual:
                    counterfactual_combinations.append(clean_data)
                else:
                    regular_combinations.append(clean_data)

        return {
            "regular": regular_combinations,
            "counterfactual": counterfactual_combinations,
            "all": regular_combinations + counterfactual_combinations,
        }
