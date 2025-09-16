import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import itertools
import copy
import numpy as np
from ..config import Config
from .preprocessing import preprocess_example_image, preprocess_target_image
from .augmentation import generate_augmented_examples


def custom_collate_fn(batch):
    """
    Custom collate function for ARC dataset with simplified structure.

    Args:
        batch: List of samples from the dataset

    Returns:
        Collated batch with proper tensor structure
    """
    batch_size = len(batch)

    # Pre-allocate tensors
    rule_latent_inputs = torch.zeros(
        [batch_size, 2, 2, 3, 64, 64]
    )  # 2 examples, 2 images each
    test_inputs = torch.zeros([batch_size, 1, 30, 30])
    test_outputs = torch.zeros([batch_size, 1, 30, 30])
    holdout_inputs = torch.zeros([batch_size, 1, 30, 30])
    holdout_outputs = torch.zeros([batch_size, 1, 30, 30])
    has_holdout = torch.zeros([batch_size], dtype=torch.bool)

    # Collect metadata for each sample
    task_indices = []
    task_ids = []
    is_counterfactual = []
    combination_indices = []
    pair_indices = []
    total_combinations = []

    # Collect raw rule latent examples for decoder
    raw_rule_latent_examples = []
    raw_rule_latent_targets = []

    # Fill with real data
    for i, sample in enumerate(batch):
        # Rule latent inputs (2 examples for ResNet)
        rule_latent_inputs[i, 0, 0] = sample["rule_latent_examples"][0][
            "input"
        ].squeeze(0)  # [3, 64, 64]
        rule_latent_inputs[i, 0, 1] = sample["rule_latent_examples"][0][
            "output"
        ].squeeze(0)
        rule_latent_inputs[i, 1, 0] = sample["rule_latent_examples"][1][
            "input"
        ].squeeze(0)
        rule_latent_inputs[i, 1, 1] = sample["rule_latent_examples"][1][
            "output"
        ].squeeze(0)

        # Test example
        test_inputs[i] = sample["test_example"]["input"].squeeze(0)
        test_outputs[i] = sample["test_example"]["output"].squeeze(0)

        # Holdout example (if available)
        if sample["holdout_example"] is not None:
            holdout_inputs[i] = sample["holdout_example"]["input"].squeeze(0)
            holdout_outputs[i] = sample["holdout_example"]["output"].squeeze(0)
            has_holdout[i] = True

        # Collect raw rule latent examples for decoder
        raw_rule_latent_examples.append(sample["rule_latent_examples"])
        raw_rule_latent_targets.append(sample["rule_latent_targets"])

        # Collect metadata
        task_indices.append(sample["task_idx"])
        task_ids.append(sample["task_id"])
        is_counterfactual.append(sample["is_counterfactual"])
        combination_indices.append(sample["combination_idx"])
        pair_indices.append(sample["pair_indices"])
        total_combinations.append(sample["total_combinations"])

    return {
        "rule_latent_inputs": rule_latent_inputs,  # [B, 2, 2, 3, 64, 64]
        "test_inputs": test_inputs,  # [B, 1, 30, 30]
        "test_outputs": test_outputs,  # [B, 1, 30, 30]
        "holdout_inputs": holdout_inputs,  # [B, 1, 30, 30]
        "holdout_outputs": holdout_outputs,  # [B, 1, 30, 30]
        "has_holdout": has_holdout,  # [B]
        "raw_rule_latent_examples": raw_rule_latent_examples,  # [B] list of raw examples (ResNet format)
        "raw_rule_latent_targets": raw_rule_latent_targets,  # [B] list of raw targets (ARC format)
        "task_indices": task_indices,  # [B] list of task indices
        "task_ids": task_ids,  # [B] list of task IDs
        "is_counterfactual": is_counterfactual,  # [B] list of counterfactual flags
        "combination_indices": combination_indices,  # [B] list of combination indices
        "pair_indices": pair_indices,  # [B] list of pair indices
        "total_combinations": total_combinations,  # [B] list of total combinations per task
    }


class ARCDataset(Dataset):
    """
    Dataset for ARC tasks with full combination coverage and holdout support.

    Loads raw JSON data and provides access to all combinations of train examples
    for rule latent creation. Each epoch sees all combinations from all tasks.
    Supports holdout validation mode.
    """

    def __init__(
        self,
        raw_data_dir: str,
        config: Config,
        holdout: bool = False,
        use_first_combination_only: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            raw_data_dir: Directory containing raw JSON task files
            config: Configuration object
            holdout: If True, hold out last train example for validation
            use_first_combination_only: If True, always use first combination (for evaluation)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.config = config
        self.holdout = holdout
        self.use_first_combination_only = use_first_combination_only

        # Load raw tasks
        self.tasks = self._load_raw_tasks()

        # Generate combinations for each task
        self.combinations = self._generate_combinations()

        # Filter tasks with sufficient examples
        self.valid_tasks = self._filter_valid_tasks()

        # create mapping from linear index to (task_idx, combination_idx) for full combination epochs
        self._create_combination_mapping()

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

    def _generate_combinations(self) -> List[List[Tuple[int, int]]]:
        """Generate all possible 2-combinations for rule latent creation."""
        combinations = []

        for task_idx, task in enumerate(self.tasks):
            # Get original examples and preserve them
            original_examples = task["train"]
            task["original_train"] = copy.deepcopy(
                original_examples
            )  # Preserve original

            # Generate augmented examples if enabled
            if self.config.use_color_relabeling:
                augmented_examples = generate_augmented_examples(
                    original_examples,
                    num_variants=self.config.augmentation_variants,
                    preserve_background=self.config.preserve_background,
                    seed=self.config.random_seed + task_idx,  # Different seed per task
                )
                # Store augmented examples in task
                task["augmented_train"] = augmented_examples
                total_examples = len(original_examples) + len(augmented_examples)
            else:
                total_examples = len(original_examples)

            # Generate combinations from all examples
            if total_examples >= 2:
                # For holdout mode, exclude the last original example from combinations
                if self.holdout and len(original_examples) > 2:
                    # Only use the first (len(original_examples) - 1) examples for combinations
                    # This excludes the holdout example
                    max_original_idx = len(original_examples) - 1
                    if self.config.use_color_relabeling:
                        # Augmented examples come after original examples
                        max_idx = max_original_idx + len(augmented_examples)
                    else:
                        max_idx = max_original_idx

                    # Generate combinations only from non-holdout examples
                    task_combinations = list(itertools.combinations(range(max_idx), 2))
                else:
                    # Use all available examples for combinations
                    task_combinations = list(
                        itertools.combinations(range(total_examples), 2)
                    )

                # NEW: Add counterfactual combinations if enabled
                if self.config.enable_counterfactuals:
                    # Create counterfactual examples for this task
                    self._create_counterfactual_examples(task)

                    # Add counterfactual combinations
                    # Mark them with a flag to indicate they're counterfactual
                    counterfactual_combinations = [
                        (combo[0], combo[1], True) for combo in task_combinations
                    ]
                    # Mark original combinations as not counterfactual
                    original_combinations = [
                        (combo[0], combo[1], False) for combo in task_combinations
                    ]
                    # Combine both
                    task_combinations = (
                        original_combinations + counterfactual_combinations
                    )
                else:
                    # Original behavior - mark all as not counterfactual
                    task_combinations = [
                        (combo[0], combo[1], False) for combo in task_combinations
                    ]

                combinations.append(task_combinations)
            else:
                combinations.append([])

        return combinations

    def _filter_valid_tasks(self) -> List[int]:
        """Filter tasks with sufficient examples for training."""
        valid_indices = []
        for i, task in enumerate(self.tasks):
            if self.holdout:
                # For holdout mode, need at least 3 training examples
                # (2 for rule latent creation + 1 for holdout + at least 1 remaining for training)
                if len(task["train"]) >= 3:
                    valid_indices.append(i)
            else:
                # For regular mode, need at least 2 training examples
                if len(task["train"]) >= 2:
                    valid_indices.append(i)
        return valid_indices

    def _create_combination_mapping(self):
        """Create mapping from linear index to (task_idx, combination_idx) pairs."""
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
        """Return total number of combinations across all valid tasks."""
        return len(self.combination_mapping)

    def _preprocess_example(
        self, example: Dict[str, Any], apply_counterfactual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single example (input/output pair)."""
        input_tensor = preprocess_example_image(example["input"], self.config)
        output_tensor = preprocess_example_image(example["output"], self.config)

        # Apply counterfactual transformation after preprocessing if requested
        if apply_counterfactual:
            output_tensor = self._apply_counterfactual_transform(output_tensor)

        return {
            "input": input_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
            "output": output_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
        }

    def _preprocess_target(
        self, example: Dict[str, Any], apply_counterfactual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single target example (input/output pair)."""
        input_tensor = preprocess_target_image(example["input"], self.config)
        output_tensor = preprocess_target_image(example["output"], self.config)

        # Apply counterfactual transformation after preprocessing if requested
        if apply_counterfactual:
            output_tensor = self._apply_counterfactual_transform(output_tensor)

        return {
            "input": input_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
            "output": output_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
        }

    def _apply_counterfactual_transform(self, image):
        """Apply counterfactual transformation to an image."""
        if self.config.counterfactual_transform == "rotate_90":
            return self._rotate_90(image)
        elif self.config.counterfactual_transform == "rotate_180":
            return self._rotate_180(image)
        elif self.config.counterfactual_transform == "rotate_270":
            return self._rotate_270(image)
        elif self.config.counterfactual_transform == "reflect_h":
            return self._reflect_horizontal(image)
        elif self.config.counterfactual_transform == "reflect_v":
            return self._reflect_vertical(image)
        else:
            raise ValueError(
                f"Unknown counterfactual transform: {self.config.counterfactual_transform}"
            )

    def _rotate_90(self, image):
        """Rotate image by 90 degrees clockwise."""
        if isinstance(image, torch.Tensor):
            return torch.rot90(image, k=1, dims=[-2, -1])
        else:  # numpy array
            return np.rot90(image, k=1)

    def _rotate_180(self, image):
        """Rotate image by 180 degrees."""
        if isinstance(image, torch.Tensor):
            return torch.rot90(image, k=2, dims=[-2, -1])
        else:  # numpy array
            return np.rot90(image, k=2)

    def _rotate_270(self, image):
        """Rotate image by 270 degrees clockwise (90 degrees counterclockwise)."""
        if isinstance(image, torch.Tensor):
            return torch.rot90(image, k=3, dims=[-2, -1])
        else:  # numpy array
            return np.rot90(image, k=3)

    def _reflect_horizontal(self, image):
        """Reflect image horizontally."""
        if isinstance(image, torch.Tensor):
            return torch.flip(image, dims=[-1])
        else:  # numpy array
            return np.fliplr(image)

    def _reflect_vertical(self, image):
        """Reflect image vertically."""
        if isinstance(image, torch.Tensor):
            return torch.flip(image, dims=[-2])
        else:  # numpy array
            return np.flipud(image)

    def _create_counterfactual_examples(self, task):
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
        if self.config.use_color_relabeling and "augmented_train" in task:
            counterfactual_augmented = []
            for example in task["augmented_train"]:
                cf_example = copy.deepcopy(example)
                cf_example["input"] = example["input"]  # Keep input the same
                # Don't transform output here - we'll do it after preprocessing
                counterfactual_augmented.append(cf_example)
            task["counterfactual_augmented_train"] = counterfactual_augmented

    def _get_combination_info(self, idx: int) -> Tuple[int, int, Tuple[int, int], bool]:
        """Get task index, combination index, pair indices, and counterfactual flag."""
        task_idx, combination_idx = self.combination_mapping[idx]
        task_combinations = self.combinations[task_idx]
        pair_indices = task_combinations[combination_idx]

        # Check if this is a counterfactual combination
        if len(pair_indices) == 3:  # (i, j, is_counterfactual)
            i, j, is_counterfactual = pair_indices
        else:
            i, j = pair_indices
            is_counterfactual = False

        return task_idx, combination_idx, (i, j), is_counterfactual

    def _get_all_examples(
        self, task: Dict[str, Any], is_counterfactual: bool
    ) -> List[Dict[str, Any]]:
        """Get all available examples (original + augmented + counterfactual if applicable)."""
        # Start with training examples, excluding holdout if holdout mode is enabled
        if self.holdout and len(task["train"]) > 2:
            # Exclude the last training example (holdout) from rule latent creation
            all_examples = task["train"][:-1]
        else:
            all_examples = task["train"]

        if self.config.use_color_relabeling and "augmented_train" in task:
            all_examples = all_examples + task["augmented_train"]

        if is_counterfactual:
            all_examples = all_examples + task["counterfactual_train"]
            if (
                self.config.use_color_relabeling
                and "counterfactual_augmented_train" in task
            ):
                all_examples = all_examples + task["counterfactual_augmented_train"]

        return all_examples

    def _create_rule_latent_inputs(
        self,
        all_examples: List[Dict[str, Any]],
        i: int,
        j: int,
        is_counterfactual: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create rule latent inputs from examples i and j."""
        if is_counterfactual:
            # For counterfactual combinations, create counterfactual versions
            ex1 = copy.deepcopy(all_examples[i])
            ex2 = copy.deepcopy(all_examples[j])

            # Preprocess with counterfactual transformation
            ex1_processed = self._preprocess_example(ex1, apply_counterfactual=True)
            ex2_processed = self._preprocess_example(ex2, apply_counterfactual=True)

            return [ex1_processed, ex2_processed]
        else:
            # For non-counterfactual combinations, use original examples
            return [
                self._preprocess_example(all_examples[i]),
                self._preprocess_example(all_examples[j]),
            ]

    def _create_rule_latent_targets(
        self,
        all_examples: List[Dict[str, Any]],
        i: int,
        j: int,
        is_counterfactual: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create rule latent targets (ARC format) from examples i and j for decoder."""
        if is_counterfactual:
            # For counterfactual combinations, create counterfactual versions
            ex1 = copy.deepcopy(all_examples[i])
            ex2 = copy.deepcopy(all_examples[j])

            # Preprocess with counterfactual transformation for decoder
            ex1_processed = self._preprocess_target(ex1, apply_counterfactual=True)
            ex2_processed = self._preprocess_target(ex2, apply_counterfactual=True)

            return [ex1_processed, ex2_processed]
        else:
            # For non-counterfactual combinations, use original examples
            return [
                self._preprocess_target(all_examples[i]),
                self._preprocess_target(all_examples[j]),
            ]

    def _create_training_targets(
        self,
        task: Dict[str, Any],
        is_counterfactual: bool,
        train_examples_to_use: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create training targets based on whether this is a counterfactual combination."""
        if is_counterfactual:
            # Use counterfactual training examples (with transformations applied)
            training_targets = []

            # Add counterfactual training examples (apply transformation after preprocessing)
            for train_example in task["counterfactual_train"]:
                training_targets.append(
                    self._preprocess_target(train_example, apply_counterfactual=True)
                )

            # Add counterfactual augmented examples if available
            if (
                self.config.use_color_relabeling
                and "counterfactual_augmented_train" in task
            ):
                for train_example in task["counterfactual_augmented_train"]:
                    training_targets.append(
                        self._preprocess_target(
                            train_example, apply_counterfactual=True
                        )
                    )

            # Add test examples (both original and counterfactual)
            for test_example in task["test"]:
                training_targets.append(self._preprocess_target(test_example))
            for test_example in task["counterfactual_test"]:
                training_targets.append(
                    self._preprocess_target(test_example, apply_counterfactual=True)
                )

            return training_targets
        else:
            # Use original training examples (existing logic)
            training_targets = []
            if train_examples_to_use is None:
                train_examples_to_use = task["train"]
            for train_example in train_examples_to_use:
                training_targets.append(self._preprocess_target(train_example))
            for test_example in task["test"]:
                training_targets.append(self._preprocess_target(test_example))
            return training_targets

    def _create_holdout_target(
        self, task: Dict[str, Any], is_counterfactual: bool
    ) -> Dict[str, torch.Tensor]:
        """Create holdout target if enabled and available."""
        if not self.holdout or len(task["train"]) <= 2:
            return None

        if is_counterfactual and len(task["counterfactual_train"]) > 2:
            return self._preprocess_target(
                task["counterfactual_train"][-1], apply_counterfactual=True
            )
        else:
            return self._preprocess_target(task["train"][-1])

    def _get_train_examples_to_use(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get training examples to use (excluding holdout if applicable)."""
        if self.holdout and len(task["train"]) > 2:
            # Remove holdout from original train examples to use
            train_examples_to_use = task["train"][:-1]
            # Add augmented examples if available
            if self.config.use_color_relabeling and "augmented_train" in task:
                train_examples_to_use = train_examples_to_use + task["augmented_train"]
            return train_examples_to_use
        else:
            return task["train"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample by index - simplified structure.

        Returns only what's needed for a single combination:
        - 2 examples for rule latent creation
        - 1 test example for evaluation
        - Optional holdout
        """
        # Get combination information
        task_idx, combination_idx, (i, j), is_counterfactual = (
            self._get_combination_info(idx)
        )
        task = self.tasks[task_idx]

        # Get all available examples
        all_examples = self._get_all_examples(task, is_counterfactual)

        # Create rule latent inputs (2 examples for encoder)
        rule_latent_inputs = self._create_rule_latent_inputs(
            all_examples, i, j, is_counterfactual
        )

        # Create rule latent targets (2 examples for decoder)
        rule_latent_targets = self._create_rule_latent_targets(
            all_examples, i, j, is_counterfactual
        )

        # Create test example
        test_example = self._get_test_example(task, is_counterfactual)

        # Create holdout example (if available)
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

    def _get_test_example(
        self, task: Dict[str, Any], is_counterfactual: bool
    ) -> Dict[str, torch.Tensor]:
        """Get single test example."""
        if is_counterfactual and task.get("counterfactual_test"):
            return self._preprocess_target(
                task["counterfactual_test"][0], apply_counterfactual=True
            )
        else:
            return self._preprocess_target(task["test"][0])

    def _get_holdout_example(
        self, task: Dict[str, Any], is_counterfactual: bool
    ) -> Dict[str, torch.Tensor]:
        """Get holdout example if available."""
        if not self.holdout or len(task["train"]) <= 2:
            return None

        if is_counterfactual and len(task.get("counterfactual_train", [])) > 2:
            return self._preprocess_target(
                task["counterfactual_train"][-1], apply_counterfactual=True
            )
        else:
            return self._preprocess_target(task["train"][-1])

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
            training_examples.append(self._preprocess_target(example))

        # Add augmented examples if available
        if self.config.use_color_relabeling and "augmented_train" in task:
            for example in task["augmented_train"]:
                training_examples.append(self._preprocess_target(example))

        # Add counterfactual examples if available
        if self.config.enable_counterfactuals and "counterfactual_train" in task:
            for example in task["counterfactual_train"]:
                training_examples.append(
                    self._preprocess_target(example, apply_counterfactual=True)
                )

        return training_examples

    def get_task_combinations(self, task_idx: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get all combinations for a specific task."""
        if task_idx not in self.valid_tasks:
            raise ValueError(f"Task {task_idx} not valid")

        task_combinations = self.combinations[task_idx]

        regular_combinations = []
        counterfactual_combinations = []

        for combo_idx, combo in enumerate(task_combinations):
            if len(combo) == 3:
                i, j, is_counterfactual = combo
            else:
                i, j = combo
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
                    "pair_indices": (i, j),
                    "is_counterfactual": is_counterfactual,
                    "rule_latent_examples": combination_data["rule_latent_examples"],
                    "test_example": combination_data["test_example"],
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
