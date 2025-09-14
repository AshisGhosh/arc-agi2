import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import itertools
from ..config import Config
from .preprocessing import preprocess_example_image, preprocess_target_image
from .augmentation import generate_augmented_examples


def custom_collate_fn(batch):
    """
    Custom collate function for ARC dataset with full tensor batching.

    Args:
        batch: List of samples from the dataset

    Returns:
        Collated batch with proper tensor structure
    """
    batch_size = len(batch)
    # Find the maximum number of training examples in this batch
    max_train = max(len(sample["training_targets"]) for sample in batch)

    # Pre-allocate tensors
    rule_latent_inputs = torch.zeros(
        [batch_size, 2, 2, 3, 64, 64]
    )  # 2 examples, 2 images each
    all_train_inputs = torch.zeros([batch_size, max_train, 1, 30, 30])
    all_train_outputs = torch.zeros([batch_size, max_train, 1, 30, 30])
    test_inputs = torch.zeros([batch_size, 1, 30, 30])
    test_outputs = torch.zeros([batch_size, 1, 30, 30])
    holdout_inputs = torch.zeros([batch_size, 1, 30, 30])
    holdout_outputs = torch.zeros([batch_size, 1, 30, 30])
    num_train = torch.zeros([batch_size], dtype=torch.long)
    has_holdout = torch.zeros([batch_size], dtype=torch.bool)

    # Fill with real data
    for i, sample in enumerate(batch):
        # Rule latent inputs (2 examples for ResNet)
        rule_latent_inputs[i, 0, 0] = sample["rule_latent_inputs"][0]["input"].squeeze(
            0
        )  # [3, 64, 64]
        rule_latent_inputs[i, 0, 1] = sample["rule_latent_inputs"][0]["output"].squeeze(
            0
        )
        rule_latent_inputs[i, 1, 0] = sample["rule_latent_inputs"][1]["input"].squeeze(
            0
        )
        rule_latent_inputs[i, 1, 1] = sample["rule_latent_inputs"][1]["output"].squeeze(
            0
        )

        # Training targets
        targets = sample["training_targets"]
        num_train[i] = len(targets)

        for j, target in enumerate(targets):
            all_train_inputs[i, j] = target["input"].squeeze(0)  # [1, 30, 30]
            all_train_outputs[i, j] = target["output"].squeeze(0)

        # Test target (last in training_targets)
        test_inputs[i] = targets[-1]["input"].squeeze(0)
        test_outputs[i] = targets[-1]["output"].squeeze(0)

        # Holdout target (if available)
        if sample["holdout_target"] is not None:
            holdout_inputs[i] = sample["holdout_target"]["input"].squeeze(0)
            holdout_outputs[i] = sample["holdout_target"]["output"].squeeze(0)
            has_holdout[i] = True

    return {
        "rule_latent_inputs": rule_latent_inputs,  # [B, 2, 2, 3, 64, 64]
        "all_train_inputs": all_train_inputs,  # [B, max_train, 1, 30, 30]
        "all_train_outputs": all_train_outputs,  # [B, max_train, 1, 30, 30]
        "test_inputs": test_inputs,  # [B, 1, 30, 30]
        "test_outputs": test_outputs,  # [B, 1, 30, 30]
        "holdout_inputs": holdout_inputs,  # [B, 1, 30, 30]
        "holdout_outputs": holdout_outputs,  # [B, 1, 30, 30]
        "num_train": num_train,  # [B]
        "has_holdout": has_holdout,  # [B]
    }


class ARCDataset(Dataset):
    """
    Dataset for ARC tasks with combination cycling and holdout support.

    Loads raw JSON data and cycles through different combinations of train examples
    for rule latent creation. Supports holdout validation mode.
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

        # per-task cycling counters
        self.task_cycle_counters = {i: 0 for i in self.valid_tasks}

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
            # Get original examples
            original_examples = task["train"]

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
                # Use all available examples for combinations
                task_combinations = list(
                    itertools.combinations(range(total_examples), 2)
                )
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

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.valid_tasks)

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess a single example (input/output pair)."""
        input_tensor = preprocess_example_image(example["input"], self.config)
        output_tensor = preprocess_example_image(example["output"], self.config)
        return {
            "input": input_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
            "output": output_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
        }

    def _preprocess_target(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess a single target example (input/output pair)."""
        input_tensor = preprocess_target_image(example["input"], self.config)
        output_tensor = preprocess_target_image(example["output"], self.config)
        return {
            "input": input_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
            "output": output_tensor.unsqueeze(0),  # Add batch dimension [1, C, H, W]
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample by index with combination cycling.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - rule_latent_inputs: List of 2 examples for rule latent creation
                - training_targets: List of all examples for training
                - holdout_target: Holdout example (if holdout=True)
                - combination_info: Information about current combination
        """
        task_idx = self.valid_tasks[idx]
        task = self.tasks[task_idx]
        task_combinations = self.combinations[task_idx]

        # Cycle through combinations
        combination_idx = self._get_combination_idx(idx)
        pair_indices = task_combinations[combination_idx]

        # Get all available examples (original + augmented)
        all_examples = task["train"]
        if self.config.use_color_relabeling and "augmented_train" in task:
            all_examples = task["train"] + task["augmented_train"]

        # Rule latent inputs (2 examples) - preprocess for ResNet
        rule_latent_inputs = [
            self._preprocess_example(all_examples[pair_indices[0]]),
            self._preprocess_example(all_examples[pair_indices[1]]),
        ]

        # Holdout target (if enabled and available)
        holdout_target = None
        train_examples_to_use = all_examples  # Use all examples (original + augmented)
        if self.holdout and len(task["train"]) > 2:
            # For holdout, use the last original train example (not augmented)
            holdout_target = self._preprocess_target(
                task["train"][-1]
            )  # Last original train example
            # Remove holdout from original train examples to use
            train_examples_to_use = task["train"][:-1]
            # Add augmented examples if available
            if self.config.use_color_relabeling and "augmented_train" in task:
                train_examples_to_use = train_examples_to_use + task["augmented_train"]

        # Training targets (all available examples) - preprocess appropriately
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

    def _get_combination_idx(self, idx: int) -> int:
        """Get combination index for this task (cycles through combinations)."""
        task_idx = self.valid_tasks[idx]
        task_combinations = self.combinations[task_idx]

        if self.use_first_combination_only:
            # For evaluation, always use first combination
            return 0
        else:
            # For training, cycle through combinations
            combination_idx = self.task_cycle_counters[task_idx] % len(
                task_combinations
            )
            self.task_cycle_counters[task_idx] += 1
            return combination_idx
