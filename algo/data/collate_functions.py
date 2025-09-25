"""
Collate functions for different model types.

This module provides collate functions that handle batching
for both ResNet and Patch model datasets.
"""

import torch


def unified_collate_fn(batch):
    """
    Unified collate function for both ResNet and Patch models.

    Provides raw support examples and lets trainers handle reshaping as needed.

    Args:
        batch: List of combination samples from the dataset

    Returns:
        Collated batch with raw support examples and test data
    """
    batch_size = len(batch)

    # Find maximum number of test examples in this batch
    max_test_examples = max(sample["num_test_examples"] for sample in batch)

    # Test examples are now always grayscale grid format [1, 30, 30]
    test_inputs = torch.zeros([batch_size, max_test_examples, 30, 30])
    test_outputs = torch.zeros([batch_size, max_test_examples, 30, 30])
    holdout_inputs = torch.zeros([batch_size, 1, 30, 30])
    holdout_outputs = torch.zeros([batch_size, 1, 30, 30])

    test_masks = torch.zeros([batch_size, max_test_examples], dtype=torch.bool)
    has_holdout = torch.zeros([batch_size], dtype=torch.bool)

    # Collect metadata for each sample (only used fields)
    task_indices = []
    num_test_examples = []
    augmentation_groups = []

    # Collect support examples - let trainers handle reshaping
    support_example_inputs = []  # [B] list of [2] support input tensors
    support_example_outputs = []  # [B] list of [2] support output tensors
    support_example_inputs_rgb = []  # [B] list of [2] RGB support input tensors (for ResNet)
    support_example_outputs_rgb = []  # [B] list of [2] RGB support output tensors (for ResNet)

    # Fill with real data
    for i, sample in enumerate(batch):
        # Get support examples (always grayscale grid format [1, 30, 30])
        support_inputs = [
            sample["support_examples"][0]["input"].squeeze(0),  # [1, 30, 30]
            sample["support_examples"][1]["input"].squeeze(0),
        ]
        support_outputs = [
            sample["support_examples"][0]["output"].squeeze(0),  # [1, 30, 30]
            sample["support_examples"][1]["output"].squeeze(0),
        ]
        support_example_inputs.append(support_inputs)
        support_example_outputs.append(support_outputs)

        # Get RGB support examples (optional, for ResNet only)
        if sample["support_examples_rgb"] is not None:
            # We have RGB support examples from ResNet dataset
            rgb_support_inputs = [
                sample["support_examples_rgb"][0]["input"].squeeze(0),  # [3, 64, 64]
                sample["support_examples_rgb"][1]["input"].squeeze(0),
            ]
            rgb_support_outputs = [
                sample["support_examples_rgb"][0]["output"].squeeze(0),  # [3, 64, 64]
                sample["support_examples_rgb"][1]["output"].squeeze(0),
            ]
            support_example_inputs_rgb.append(rgb_support_inputs)
            support_example_outputs_rgb.append(rgb_support_outputs)
        else:
            # No RGB support examples (from Patch dataset)
            support_example_inputs_rgb.append(None)
            support_example_outputs_rgb.append(None)

        # Test examples (variable length with masking)
        sample_test_examples = sample["test_examples"]
        sample_num_test = sample["num_test_examples"]
        for j, test_example in enumerate(sample_test_examples):
            test_inputs[i, j] = test_example["input"].squeeze(0)
            test_outputs[i, j] = test_example["output"].squeeze(0)
            test_masks[i, j] = True

        # Set mask for unused slots
        for j in range(sample_num_test, max_test_examples):
            test_masks[i, j] = False

        # Holdout example (if available)
        if sample["holdout_example"] is not None:
            holdout_inputs[i] = sample["holdout_example"]["input"].squeeze(0)
            holdout_outputs[i] = sample["holdout_example"]["output"].squeeze(0)
            has_holdout[i] = True

        # Collect metadata (only used fields)
        task_indices.append(sample["task_idx"])
        num_test_examples.append(sample["num_test_examples"])
        augmentation_groups.append(sample["augmentation_group"])

    return {
        # Support examples - trainers handle reshaping
        "support_example_inputs": support_example_inputs,  # [B] list of [2] support input tensors
        "support_example_outputs": support_example_outputs,  # [B] list of [2] support output tensors
        # RGB support examples (for ResNet) - None if grayscale data
        "support_example_inputs_rgb": support_example_inputs_rgb,  # [B] list of [2] RGB support input tensors or None
        "support_example_outputs_rgb": support_example_outputs_rgb,  # [B] list of [2] RGB support output tensors or None
        # Test examples
        "test_inputs": test_inputs,  # [B, max_test_examples, 30, 30]
        "test_outputs": test_outputs,  # [B, max_test_examples, 30, 30]
        "test_masks": test_masks,  # [B, max_test_examples] - True for valid test examples
        # Holdout examples
        "holdout_inputs": holdout_inputs,  # [B, 1, 30, 30]
        "holdout_outputs": holdout_outputs,  # [B, 1, 30, 30]
        "has_holdout": has_holdout,  # [B]
        # Metadata (only used fields)
        "task_indices": task_indices,  # [B] list of task indices
        "num_test_examples": num_test_examples,  # [B] list of number of test examples per task
        "augmentation_groups": augmentation_groups,  # [B] list of augmentation group IDs
    }


def flatten_patch_collate_fn(batch):
    """
    Flattening collate function for patch dataset.

    Converts combinations to individual (support1, support2, test) samples
    during batching for efficient processing.

    Args:
        batch: List of combination samples from PatchARCDataset

    Returns:
        Dictionary with flattened samples ready for batching
    """
    flattened_samples = []

    # Flatten each combination into individual samples
    for sample in batch:
        # Extract support examples (same for all test examples in this combination)
        support1_input = sample["support_examples"][0]["input"].squeeze()  # [30, 30]
        support1_output = sample["support_examples"][0]["output"].squeeze()  # [30, 30]
        support2_input = sample["support_examples"][1]["input"].squeeze()  # [30, 30]
        support2_output = sample["support_examples"][1]["output"].squeeze()  # [30, 30]

        # Create one sample per test example
        for test_example in sample["test_examples"]:
            flattened_samples.append(
                {
                    "support1_input": support1_input,
                    "support1_output": support1_output,
                    "support2_input": support2_input,
                    "support2_output": support2_output,
                    "test_input": test_example["input"].squeeze(),  # [30, 30]
                    "test_target": test_example["output"].squeeze(),  # [30, 30]
                    "task_idx": sample["task_idx"],
                    "task_id": sample["task_id"],
                    "combination_idx": sample["combination_idx"],
                    "pair_indices": sample["pair_indices"],
                    "is_counterfactual": sample["is_counterfactual"],
                    "augmentation_group": sample["augmentation_group"],
                }
            )

    # Stack tensors
    support1_inputs = torch.stack(
        [s["support1_input"] for s in flattened_samples]
    )  # [B, 30, 30]
    support1_outputs = torch.stack(
        [s["support1_output"] for s in flattened_samples]
    )  # [B, 30, 30]
    support2_inputs = torch.stack(
        [s["support2_input"] for s in flattened_samples]
    )  # [B, 30, 30]
    support2_outputs = torch.stack(
        [s["support2_output"] for s in flattened_samples]
    )  # [B, 30, 30]
    test_inputs = torch.stack(
        [s["test_input"] for s in flattened_samples]
    )  # [B, 30, 30]
    test_targets = torch.stack(
        [s["test_target"] for s in flattened_samples]
    )  # [B, 30, 30]

    # Collect metadata
    task_indices = [s["task_idx"] for s in flattened_samples]
    task_ids = [s["task_id"] for s in flattened_samples]
    combination_indices = [s["combination_idx"] for s in flattened_samples]
    pair_indices = [s["pair_indices"] for s in flattened_samples]
    is_counterfactuals = [s["is_counterfactual"] for s in flattened_samples]
    augmentation_groups = [s["augmentation_group"] for s in flattened_samples]

    return {
        # Batched tensors
        "support1_inputs": support1_inputs,  # [B, 30, 30]
        "support1_outputs": support1_outputs,  # [B, 30, 30]
        "support2_inputs": support2_inputs,  # [B, 30, 30]
        "support2_outputs": support2_outputs,  # [B, 30, 30]
        "test_inputs": test_inputs,  # [B, 30, 30]
        "test_targets": test_targets,  # [B, 30, 30]
        # Compatibility keys for evaluation
        "test_outputs": test_targets,  # Alias for test_targets
        "num_test_examples": [1]
        * len(flattened_samples),  # Each sample has 1 test example
        # Metadata (lists)
        "task_indices": task_indices,  # [B] list
        "task_ids": task_ids,  # [B] list
        "combination_indices": combination_indices,  # [B] list
        "pair_indices": pair_indices,  # [B] list
        "is_counterfactuals": is_counterfactuals,  # [B] list
        "augmentation_groups": augmentation_groups,  # [B] list
    }
