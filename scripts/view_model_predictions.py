#!/usr/bin/env python3
"""
streamlit app for visualizing model predictions from overfitting experiments.

interactive web interface to load checkpoints and visualize model outputs.
"""

import streamlit as st
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from algo.config import Config
from algo.data import ARCDataset
from algo.data.task_subset import TaskSubset
from algo.models.simple_arc import SimpleARCModel
from scripts.visualization_utils import (
    tensor_to_numpy,
    tensor_to_grayscale_numpy,
    denormalize_rgb,
    visualize_prediction_comparison,
)


# set page config
st.set_page_config(page_title="arc model predictions", page_icon="🤖", layout="wide")


@dataclass
class NoiseConfig:
    """Configuration for noise injection parameters."""

    # Rule latent noise
    inject_noise: bool = False
    noise_type: str = "gaussian"
    noise_std: float = 1.0
    noise_range: float = 1.0
    noise_ratio: float = 1.0

    # Support example noise - A input
    noise_a_input: bool = False
    noise_a_input_type: str = "gaussian"
    noise_a_input_std: float = 1.0
    noise_a_input_range: float = 1.0
    noise_a_input_ratio: float = 1.0

    # Support example noise - A output
    noise_a_output: bool = False
    noise_a_output_type: str = "gaussian"
    noise_a_output_std: float = 1.0
    noise_a_output_range: float = 1.0
    noise_a_output_ratio: float = 1.0

    # Support example noise - B input
    noise_b_input: bool = False
    noise_b_input_type: str = "gaussian"
    noise_b_input_std: float = 1.0
    noise_b_input_range: float = 1.0
    noise_b_input_ratio: float = 1.0

    # Support example noise - B output
    noise_b_output: bool = False
    noise_b_output_type: str = "gaussian"
    noise_b_output_std: float = 1.0
    noise_b_output_range: float = 1.0
    noise_b_output_ratio: float = 1.0

    # Test input noise
    noise_test_inputs: bool = False
    noise_test_type: str = "gaussian"
    noise_test_std: float = 1.0
    noise_test_range: float = 1.0
    noise_test_ratio: float = 1.0

    def has_any_noise(self) -> bool:
        """Check if any noise is enabled."""
        return (
            self.inject_noise
            or self.noise_a_input
            or self.noise_a_output
            or self.noise_b_input
            or self.noise_b_output
            or self.noise_test_inputs
        )

    def get_noise_components(self) -> List[str]:
        """Get list of active noise components."""
        components = []
        if self.inject_noise:
            components.append("rule_latent")
        if self.noise_a_input:
            components.append("A_input")
        if self.noise_a_output:
            components.append("A_output")
        if self.noise_b_input:
            components.append("B_input")
        if self.noise_b_output:
            components.append("B_output")
        if self.noise_test_inputs:
            components.append("test_inputs")
        return components

    def get_noise_info_string(self) -> str:
        """Get formatted noise info string for display."""
        if not self.has_any_noise():
            return ""

        components = self.get_noise_components()
        components_str = ", ".join(components)

        # Determine which noise parameters to display based on active components
        if self.inject_noise and not any(
            [
                self.noise_a_input,
                self.noise_a_output,
                self.noise_b_input,
                self.noise_b_output,
                self.noise_test_inputs,
            ]
        ):
            # Only rule latent noise is active
            if self.noise_type == "gaussian":
                return f" (noise: {self.noise_type}, std={self.noise_std:.1f}, ratio={self.noise_ratio:.1f}, components: {components_str})"
            elif self.noise_type == "uniform":
                return f" (noise: {self.noise_type}, range={self.noise_range:.1f}, ratio={self.noise_ratio:.1f}, components: {components_str})"
            else:
                return f" (noise: {self.noise_type}, ratio={self.noise_ratio:.1f}, components: {components_str})"
        else:
            # Support example noise is active - show the first active support noise parameters
            if self.noise_a_input:
                if self.noise_a_input_type == "gaussian":
                    return f" (noise: {self.noise_a_input_type}, std={self.noise_a_input_std:.1f}, ratio={self.noise_a_input_ratio:.1f}, components: {components_str})"
                elif self.noise_a_input_type == "uniform":
                    return f" (noise: {self.noise_a_input_type}, range={self.noise_a_input_range:.1f}, ratio={self.noise_a_input_ratio:.1f}, components: {components_str})"
                else:
                    return f" (noise: {self.noise_a_input_type}, ratio={self.noise_a_input_ratio:.1f}, components: {components_str})"
            elif self.noise_a_output:
                if self.noise_a_output_type == "gaussian":
                    return f" (noise: {self.noise_a_output_type}, std={self.noise_a_output_std:.1f}, ratio={self.noise_a_output_ratio:.1f}, components: {components_str})"
                elif self.noise_a_output_type == "uniform":
                    return f" (noise: {self.noise_a_output_type}, range={self.noise_a_output_range:.1f}, ratio={self.noise_a_output_ratio:.1f}, components: {components_str})"
                else:
                    return f" (noise: {self.noise_a_output_type}, ratio={self.noise_a_output_ratio:.1f}, components: {components_str})"
            elif self.noise_b_input:
                if self.noise_b_input_type == "gaussian":
                    return f" (noise: {self.noise_b_input_type}, std={self.noise_b_input_std:.1f}, ratio={self.noise_b_input_ratio:.1f}, components: {components_str})"
                elif self.noise_b_input_type == "uniform":
                    return f" (noise: {self.noise_b_input_type}, range={self.noise_b_input_range:.1f}, ratio={self.noise_b_input_ratio:.1f}, components: {components_str})"
                else:
                    return f" (noise: {self.noise_b_input_type}, ratio={self.noise_b_input_ratio:.1f}, components: {components_str})"
            elif self.noise_b_output:
                if self.noise_b_output_type == "gaussian":
                    return f" (noise: {self.noise_b_output_type}, std={self.noise_b_output_std:.1f}, ratio={self.noise_b_output_ratio:.1f}, components: {components_str})"
                elif self.noise_b_output_type == "uniform":
                    return f" (noise: {self.noise_b_output_type}, range={self.noise_b_output_range:.1f}, ratio={self.noise_b_output_ratio:.1f}, components: {components_str})"
                else:
                    return f" (noise: {self.noise_b_output_type}, ratio={self.noise_b_output_ratio:.1f}, components: {components_str})"
            elif self.noise_test_inputs:
                if self.noise_test_type == "gaussian":
                    return f" (noise: {self.noise_test_type}, std={self.noise_test_std:.1f}, ratio={self.noise_test_ratio:.1f}, components: {components_str})"
                elif self.noise_test_type == "uniform":
                    return f" (noise: {self.noise_test_type}, range={self.noise_test_range:.1f}, ratio={self.noise_test_ratio:.1f}, components: {components_str})"
                else:
                    return f" (noise: {self.noise_test_type}, ratio={self.noise_test_ratio:.1f}, components: {components_str})"
            else:
                # Fallback to rule latent noise
                if self.noise_type == "gaussian":
                    return f" (noise: {self.noise_type}, std={self.noise_std:.1f}, ratio={self.noise_ratio:.1f}, components: {components_str})"
                elif self.noise_type == "uniform":
                    return f" (noise: {self.noise_type}, range={self.noise_range:.1f}, ratio={self.noise_ratio:.1f}, components: {components_str})"
                else:
                    return f" (noise: {self.noise_type}, ratio={self.noise_ratio:.1f}, components: {components_str})"


def create_noise_config_from_ui(
    inject_noise,
    noise_type,
    noise_std,
    noise_range,
    noise_ratio,
    noise_a_input,
    noise_a_input_type,
    noise_a_input_std,
    noise_a_input_range,
    noise_a_input_ratio,
    noise_a_output,
    noise_a_output_type,
    noise_a_output_std,
    noise_a_output_range,
    noise_a_output_ratio,
    noise_b_input,
    noise_b_input_type,
    noise_b_input_std,
    noise_b_input_range,
    noise_b_input_ratio,
    noise_b_output,
    noise_b_output_type,
    noise_b_output_std,
    noise_b_output_range,
    noise_b_output_ratio,
    noise_test_inputs,
    noise_test_type,
    noise_test_std,
    noise_test_range,
    noise_test_ratio,
) -> NoiseConfig:
    """Create NoiseConfig from UI parameters."""
    return NoiseConfig(
        inject_noise=inject_noise,
        noise_type=noise_type,
        noise_std=noise_std,
        noise_range=noise_range,
        noise_ratio=noise_ratio,
        noise_a_input=noise_a_input,
        noise_a_input_type=noise_a_input_type,
        noise_a_input_std=noise_a_input_std,
        noise_a_input_range=noise_a_input_range,
        noise_a_input_ratio=noise_a_input_ratio,
        noise_a_output=noise_a_output,
        noise_a_output_type=noise_a_output_type,
        noise_a_output_std=noise_a_output_std,
        noise_a_output_range=noise_a_output_range,
        noise_a_output_ratio=noise_a_output_ratio,
        noise_b_input=noise_b_input,
        noise_b_input_type=noise_b_input_type,
        noise_b_input_std=noise_b_input_std,
        noise_b_input_range=noise_b_input_range,
        noise_b_input_ratio=noise_b_input_ratio,
        noise_b_output=noise_b_output,
        noise_b_output_type=noise_b_output_type,
        noise_b_output_std=noise_b_output_std,
        noise_b_output_range=noise_b_output_range,
        noise_b_output_ratio=noise_b_output_ratio,
        noise_test_inputs=noise_test_inputs,
        noise_test_type=noise_test_type,
        noise_test_std=noise_test_std,
        noise_test_range=noise_test_range,
        noise_test_ratio=noise_test_ratio,
    )


def extract_sample_from_batch(batch, sample_idx, evaluation_mode="test"):
    """Extract individual sample data from batched format."""
    # Extract all test examples for this sample
    test_examples = []
    num_test_examples = batch["num_test_examples"][sample_idx]
    test_masks = batch["test_masks"][sample_idx]

    for test_idx in range(num_test_examples):
        if test_masks[test_idx]:  # Only include valid test examples
            test_examples.append(
                {
                    "input": batch["test_inputs"][sample_idx, test_idx],
                    "output": batch["test_outputs"][sample_idx, test_idx],
                }
            )

    sample = {
        "train_examples": [],
        "test_examples": test_examples,
        "num_test_examples": len(test_examples),
    }

    # Add holdout data if available
    if batch["has_holdout"][sample_idx]:
        sample["holdout_example"] = {
            "input": batch["holdout_inputs"][sample_idx],
            "output": batch["holdout_outputs"][sample_idx],
        }

    # Extract training examples from rule_latent_inputs (2 examples for encoder)
    for i in range(2):  # Always 2 examples for rule latent
        sample["train_examples"].append(
            {
                "input": batch["rule_latent_inputs"][
                    sample_idx, i, 0
                ],  # [B, 2, 2, 3, 64, 64] -> [3, 64, 64]
                "output": batch["rule_latent_inputs"][
                    sample_idx, i, 1
                ],  # [B, 2, 2, 3, 64, 64] -> [3, 64, 64]
            }
        )

    return sample


def load_experiment_info(experiment_dir: Path) -> Dict[str, Any]:
    """load experiment information from directory."""
    info = {}

    # load training info
    training_info_path = experiment_dir / "training_info.json"
    if training_info_path.exists():
        with open(training_info_path, "r") as f:
            info["training"] = json.load(f)

    # load task selection
    task_selection_path = experiment_dir / "task_selection.json"
    if task_selection_path.exists():
        with open(task_selection_path, "r") as f:
            info["tasks"] = json.load(f)

    # load evaluation results
    eval_path = experiment_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            info["evaluation"] = json.load(f)

    return info


def load_model_checkpoint(
    checkpoint_path: str, config: Config = None
) -> SimpleARCModel:
    """load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Use config from checkpoint if available, otherwise use provided config
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config is None:
        config = Config()

    model = SimpleARCModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    return model


def get_available_experiments(logs_dir: Path) -> List[Tuple[str, Path]]:
    """get list of available experiments (overfitting and training)."""
    experiments = []

    for item in logs_dir.iterdir():
        if item.is_dir() and (
            item.name.startswith("overfit_") or item.name.startswith("train_")
        ):
            # check if it has a model checkpoint
            if (item / "best_model.pt").exists():
                experiments.append((item.name, item))

    # sort by name (which includes timestamp)
    experiments.sort(key=lambda x: x[0])
    return experiments


def generate_noise_latent(
    original_latent, noise_type, noise_std=None, noise_range=None, noise_ratio=1.0
):
    """generate noise to replace part or all of the rule latent."""
    if noise_ratio == 0.0:
        return original_latent

    # generate noise based on type
    if noise_type == "gaussian":
        noise = torch.randn_like(original_latent) * noise_std
    elif noise_type == "uniform":
        noise = torch.rand_like(original_latent) * 2 * noise_range - noise_range
    elif noise_type == "zeros":
        noise = torch.zeros_like(original_latent)
    elif noise_type == "ones":
        noise = torch.ones_like(original_latent)
    else:
        raise ValueError(f"unknown noise type: {noise_type}")

    # mix original and noise based on ratio
    if noise_ratio == 1.0:
        return noise
    else:
        return (1 - noise_ratio) * original_latent + noise_ratio * noise


def generate_noise_tensor(
    original_tensor, noise_type, noise_std=None, noise_range=None, noise_ratio=1.0
):
    """generate noise to replace part or all of any tensor (examples, test inputs, etc.)."""
    if noise_ratio == 0.0:
        return original_tensor

    # generate noise based on type
    if noise_type == "gaussian":
        noise = torch.randn_like(original_tensor) * noise_std
    elif noise_type == "uniform":
        noise = torch.rand_like(original_tensor) * 2 * noise_range - noise_range
    elif noise_type == "zeros":
        noise = torch.zeros_like(original_tensor)
    elif noise_type == "ones":
        noise = torch.ones_like(original_tensor)
    else:
        raise ValueError(f"unknown noise type: {noise_type}")

    # mix original and noise based on ratio
    if noise_ratio == 1.0:
        return noise
    else:
        return (1 - noise_ratio) * original_tensor + noise_ratio * noise


def apply_noise_to_examples(rule_latent_examples, noise_config: NoiseConfig):
    """apply noise to individual training examples (A input, A output, B input, B output)."""
    # create a copy to avoid modifying the original
    noisy_examples = []
    for i, example in enumerate(rule_latent_examples):
        noisy_example = {}

        # Determine which example this is (A=0, B=1)
        is_a = i == 0

        # Apply noise to input
        if is_a and noise_config.noise_a_input:
            noisy_example["input"] = generate_noise_tensor(
                example["input"],
                noise_config.noise_a_input_type,
                noise_config.noise_a_input_std,
                noise_config.noise_a_input_range,
                noise_config.noise_a_input_ratio,
            )
        elif not is_a and noise_config.noise_b_input:
            noisy_example["input"] = generate_noise_tensor(
                example["input"],
                noise_config.noise_b_input_type,
                noise_config.noise_b_input_std,
                noise_config.noise_b_input_range,
                noise_config.noise_b_input_ratio,
            )
        else:
            noisy_example["input"] = example["input"]

        # Apply noise to output
        if is_a and noise_config.noise_a_output:
            noisy_example["output"] = generate_noise_tensor(
                example["output"],
                noise_config.noise_a_output_type,
                noise_config.noise_a_output_std,
                noise_config.noise_a_output_range,
                noise_config.noise_a_output_ratio,
            )
        elif not is_a and noise_config.noise_b_output:
            noisy_example["output"] = generate_noise_tensor(
                example["output"],
                noise_config.noise_b_output_type,
                noise_config.noise_b_output_std,
                noise_config.noise_b_output_range,
                noise_config.noise_b_output_ratio,
            )
        else:
            noisy_example["output"] = example["output"]

        noisy_examples.append(noisy_example)

    return noisy_examples


def apply_noise_to_test_inputs(
    test_examples, noise_type, noise_std=None, noise_range=None, noise_ratio=1.0
):
    """apply noise to test input examples."""
    if noise_ratio == 0.0:
        return test_examples

    # create a copy to avoid modifying the original
    noisy_test_examples = []
    for example in test_examples:
        noisy_example = {}
        noisy_example["input"] = generate_noise_tensor(
            example["input"], noise_type, noise_std, noise_range, noise_ratio
        )
        noisy_example["output"] = example["output"]  # don't noise the target output
        noisy_test_examples.append(noisy_example)

    return noisy_test_examples


def calculate_accuracy_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """calculate accuracy metrics for predictions."""
    with torch.no_grad():
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)

        perfect_matches = (predictions == targets).all(dim=(1, 2))
        perfect_accuracy = perfect_matches.float().mean().item()

        pixel_matches = predictions == targets
        pixel_accuracy = pixel_matches.float().mean().item()

        diff = torch.abs(predictions.float() - targets.float())
        near_miss = (diff <= 1.0).all(dim=(1, 2))
        near_miss_accuracy = near_miss.float().mean().item()

        return {
            "perfect_accuracy": perfect_accuracy,
            "pixel_accuracy": pixel_accuracy,
            "near_miss_accuracy": near_miss_accuracy,
        }


def evaluate_model_on_tasks(
    model,
    dataset,
    config,
    evaluation_mode="test",
    progress_bar=None,
    noise_config: NoiseConfig = None,
    enable_color_augmentation=False,
    augmentation_variants=1,
    preserve_background=True,
    augmentation_seed=42,
    enable_counterfactuals=False,
    counterfactual_transform="rotate_90",
    selected_task_indices=None,
    test_all_test_pairs=False,
):
    """evaluate model on all tasks in dataset and return results.

    Args:
        model: The model to evaluate
        dataset: Dataset to evaluate on
        config: Configuration object
        evaluation_mode: "test" for test targets, "holdout" for holdout targets
        progress_bar: Streamlit progress bar
        noise_config: NoiseConfig object containing all noise injection parameters
        enable_color_augmentation: Whether to enable color relabeling
        augmentation_variants: Number of augmented versions per example
        preserve_background: Whether to preserve background color
        augmentation_seed: Random seed for augmentation
    """
    if noise_config is None:
        noise_config = NoiseConfig()

    results = []

    # Create augmented dataset if either color augmentation or counterfactuals are enabled
    if enable_color_augmentation or enable_counterfactuals:
        # Create a copy of the config with both augmentations enabled if needed
        augmented_config = Config()
        augmented_config.__dict__.update(config.__dict__)  # Copy all config values

        if enable_color_augmentation:
            augmented_config.use_color_relabeling = True
            augmented_config.augmentation_variants = augmentation_variants
            augmented_config.preserve_background = preserve_background
            augmented_config.random_seed = augmentation_seed

        if enable_counterfactuals:
            augmented_config.enable_counterfactuals = True
            augmented_config.counterfactual_transform = counterfactual_transform

        # Create augmented dataset with both augmentations
        task_indices = (
            selected_task_indices
            if selected_task_indices
            else list(dataset.valid_tasks)
        )
        dataset = TaskSubset(
            task_indices=task_indices,
            config=augmented_config,
            arc_agi1_dir=config.arc_agi1_dir,
            holdout=True,
            use_first_combination_only=False,
        )

    # Set deterministic training for reproducible results
    config.set_deterministic_training()

    # Get task indices
    task_indices = (
        selected_task_indices if selected_task_indices else list(dataset.valid_tasks)
    )

    with torch.no_grad():
        total_tasks = len(task_indices)

        for task_idx in task_indices:
            if progress_bar:
                progress_value = min(
                    (task_idx + 1) / total_tasks, 1.0
                )  # Ensure value is between 0 and 1
                progress_bar.progress(progress_value)

            # Get all combinations for this task
            task_combinations_data = dataset.get_task_combinations(task_idx)
            task_combinations = task_combinations_data["all"]

            # Get all training examples for this task
            training_examples = dataset.get_all_training_examples_for_task(task_idx)
            train_inputs = [
                ex["input"].squeeze(0) for ex in training_examples
            ]  # Remove batch dim
            max_train = len(train_inputs)

            # Pad training inputs to consistent shape
            all_train_inputs = torch.zeros([1, max_train, 1, 30, 30])
            for j, train_input in enumerate(train_inputs):
                all_train_inputs[0, j] = train_input

            num_train = torch.tensor([max_train], dtype=torch.long)

            task_results = []

            for combo in task_combinations:
                # Extract combination data
                combo_idx = combo["combination_idx"]
                i, j = combo["pair_indices"]
                is_counterfactual = combo["is_counterfactual"]

                # Get the preprocessed data from the combination
                rule_latent_examples = combo["rule_latent_examples"]
                test_examples = combo["test_examples"]
                num_test_examples = combo["num_test_examples"]
                holdout_example = combo.get("holdout_example")

                # Apply noise to training examples if requested
                if (
                    noise_config.noise_a_input
                    or noise_config.noise_a_output
                    or noise_config.noise_b_input
                    or noise_config.noise_b_output
                ):
                    rule_latent_examples = apply_noise_to_examples(
                        rule_latent_examples, noise_config
                    )

                # Apply noise to test inputs if requested
                if noise_config.noise_test_inputs:
                    test_examples = apply_noise_to_test_inputs(
                        test_examples,
                        noise_config.noise_test_type,
                        noise_config.noise_test_std,
                        noise_config.noise_test_range,
                        noise_config.noise_test_ratio,
                    )

                # Extract the preprocessed tensors for rule latent creation
                example1_input = rule_latent_examples[0][
                    "input"
                ]  # Already has batch dimension
                example1_output = rule_latent_examples[0]["output"]
                example2_input = rule_latent_examples[1]["input"]
                example2_output = rule_latent_examples[1]["output"]

                # Create rule latent inputs tensor [1, 2, 2, 3, 64, 64]
                rule_latent_inputs = torch.zeros([1, 2, 2, 3, 64, 64])
                rule_latent_inputs[0, 0, 0] = example1_input.squeeze(0)  # [3, 64, 64]
                rule_latent_inputs[0, 0, 1] = example1_output.squeeze(0)
                rule_latent_inputs[0, 1, 0] = example2_input.squeeze(0)
                rule_latent_inputs[0, 1, 1] = example2_output.squeeze(0)

                # Run model inference
                outputs = model.forward_rule_latent_training(
                    rule_latent_inputs,
                    all_train_inputs,
                    num_train,
                )

                # inject noise into rule latent if requested
                if noise_config.inject_noise:
                    original_latent = outputs["rule_latents"][0:1]
                    noisy_latent = generate_noise_latent(
                        original_latent,
                        noise_config.noise_type,
                        noise_config.noise_std,
                        noise_config.noise_range,
                        noise_config.noise_ratio,
                    )
                    outputs["rule_latents"][0:1] = noisy_latent

                if evaluation_mode == "test":
                    if test_all_test_pairs and len(test_examples) > 1:
                        # Evaluate on all test examples and create separate results for each
                        for test_idx in range(num_test_examples):
                            test_example = test_examples[test_idx]
                            target_input = test_example["input"].unsqueeze(
                                0
                            )  # Add batch dimension
                            target_output = test_example["output"].unsqueeze(0)

                            target_logits = model.decoder(
                                outputs["rule_latents"][0:1],
                                target_input,
                            )
                            predictions = torch.argmax(target_logits, dim=1).squeeze(0)
                            metrics = calculate_accuracy_metrics(
                                predictions, target_output
                            )

                            # Create separate result for this test example
                            test_sample_data = {
                                "train_examples": [
                                    {
                                        "input": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[0]["input"]
                                            )
                                        ),
                                        "output": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[0]["output"]
                                            )
                                        ),
                                    },
                                    {
                                        "input": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[1]["input"]
                                            )
                                        ),
                                        "output": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[1]["output"]
                                            )
                                        ),
                                    },
                                ],
                                "test_examples": [
                                    {
                                        "input": tensor_to_grayscale_numpy(
                                            test_ex["input"]
                                        ),
                                        "output": tensor_to_grayscale_numpy(
                                            test_ex["output"]
                                        ),
                                    }
                                    for test_ex in test_examples
                                ],
                                "num_test_examples": num_test_examples,
                            }

                            # Add holdout data if available
                            if holdout_example is not None:
                                test_sample_data["holdout_example"] = {
                                    "input": tensor_to_grayscale_numpy(
                                        holdout_example["input"]
                                    ),
                                    "output": tensor_to_grayscale_numpy(
                                        holdout_example["output"]
                                    ),
                                }

                            task_results.append(
                                {
                                    "combination_idx": combo_idx,
                                    "pair_indices": (i, j),
                                    "is_counterfactual": is_counterfactual,
                                    "test_example_idx": test_idx,
                                    "perfect_accuracy": metrics["perfect_accuracy"],
                                    "pixel_accuracy": metrics["pixel_accuracy"],
                                    "near_miss_accuracy": metrics["near_miss_accuracy"],
                                    "predictions": predictions,
                                    "logits": target_logits,
                                    "sample_data": test_sample_data,
                                }
                            )

                        # Skip the normal result creation since we created separate results above
                        continue
                else:
                    # Use first test example only (original behavior)
                    test_example = test_examples[0] if test_examples else None
                    if test_example:
                        target_input = test_example["input"].unsqueeze(0)
                        target_output = test_example["output"].unsqueeze(0)

                        target_logits = model.decoder(
                            outputs["rule_latents"][0:1],
                            target_input,
                        )
                        predictions = torch.argmax(target_logits, dim=1).squeeze(0)
                        metrics = calculate_accuracy_metrics(predictions, target_output)
                    else:
                        metrics = {"accuracy": 0.0, "exact_match": 0.0}
                        predictions = None

                if evaluation_mode == "holdout" and holdout_example is not None:
                    # Evaluate on holdout example
                    target_input = holdout_example["input"].unsqueeze(0)
                    target_output = holdout_example["output"].unsqueeze(0)

                    target_logits = model.decoder(
                        outputs["rule_latents"][0:1],
                        target_input,
                    )
                    predictions = torch.argmax(target_logits, dim=1).squeeze(0)
                    metrics = calculate_accuracy_metrics(predictions, target_output)
                else:
                    # Fallback to first test example
                    test_example = test_examples[0]
                    target_input = test_example["input"].unsqueeze(0)
                    target_output = test_example["output"].unsqueeze(0)

                    target_logits = model.decoder(
                        outputs["rule_latents"][0:1],
                        target_input,
                    )
                    predictions = torch.argmax(target_logits, dim=1).squeeze(0)
                    metrics = calculate_accuracy_metrics(predictions, target_output)

                # Create sample data for visualization
                sample_data = {
                    "train_examples": [
                        {
                            "input": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[0]["input"])
                            ),
                            "output": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[0]["output"])
                            ),
                        },
                        {
                            "input": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[1]["input"])
                            ),
                            "output": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[1]["output"])
                            ),
                        },
                    ],
                    "test_examples": [
                        {
                            "input": tensor_to_grayscale_numpy(test_example["input"]),
                            "output": tensor_to_grayscale_numpy(test_example["output"]),
                        }
                        for test_example in test_examples
                    ],
                    "num_test_examples": num_test_examples,
                }

                # Add holdout data if available
                if holdout_example is not None:
                    sample_data["holdout_example"] = {
                        "input": tensor_to_grayscale_numpy(holdout_example["input"]),
                        "output": tensor_to_grayscale_numpy(holdout_example["output"]),
                    }

                task_results.append(
                    {
                        "combination_idx": combo_idx,
                        "pair_indices": (i, j),
                        "is_counterfactual": is_counterfactual,
                        "perfect_accuracy": metrics["perfect_accuracy"],
                        "pixel_accuracy": metrics["pixel_accuracy"],
                        "near_miss_accuracy": metrics["near_miss_accuracy"],
                        "predictions": predictions,
                        "logits": target_logits,
                        "sample_data": sample_data,
                    }
                )

            # For regular evaluation, we want to return all combinations as individual results
            # This matches the format expected by the visualization code
            for combo_result in task_results:
                result_entry = {
                    "task_idx": task_idx,
                    "global_task_index": task_idx,
                    "task_id": dataset.tasks[task_idx]["task_id"],
                    "combination_idx": combo_result["combination_idx"],
                    "pair_indices": combo_result["pair_indices"],
                    "is_counterfactual": combo_result["is_counterfactual"],
                    "perfect_accuracy": combo_result["perfect_accuracy"],
                    "pixel_accuracy": combo_result["pixel_accuracy"],
                    "near_miss_accuracy": combo_result["near_miss_accuracy"],
                    "predictions": combo_result["predictions"],
                    "logits": combo_result["logits"],
                    "sample_data": combo_result["sample_data"],
                    "evaluation_mode": evaluation_mode,
                }

                # Add test example index if it exists (for test all pairs mode)
                if "test_example_idx" in combo_result:
                    result_entry["test_example_idx"] = combo_result["test_example_idx"]

                results.append(result_entry)

    return results


def test_all_combinations(
    model,
    dataset,
    config,
    evaluation_mode="test",
    progress_bar=None,
    noise_config: NoiseConfig = None,
    enable_color_augmentation=False,
    augmentation_variants=1,
    preserve_background=True,
    augmentation_seed=42,
    enable_counterfactuals=False,
    counterfactual_transform="rotate_90",
    selected_task_indices=None,
    test_all_test_pairs=False,
):
    """test all possible combinations of train examples for rule latent creation.

    Args:
        model: The model to evaluate
        dataset: Dataset to test on
        config: Configuration object
        evaluation_mode: "test" for test targets, "holdout" for holdout targets
        progress_bar: Streamlit progress bar
        noise_config: NoiseConfig object containing all noise injection parameters
        enable_color_augmentation: Whether to enable color relabeling
        augmentation_variants: Number of augmented versions per example
        preserve_background: Whether to preserve background color
        augmentation_seed: Random seed for augmentation
        test_all_test_pairs: Whether to evaluate on all test examples for each combination
    """
    if noise_config is None:
        noise_config = NoiseConfig()

    # Set deterministic training for reproducible results
    config.set_deterministic_training()

    results = []

    # Create augmented dataset if either color augmentation or counterfactuals are enabled
    if enable_color_augmentation or enable_counterfactuals:
        # Create a copy of the config with both augmentations enabled if needed
        augmented_config = Config()
        augmented_config.__dict__.update(config.__dict__)  # Copy all config values

        if enable_color_augmentation:
            augmented_config.use_color_relabeling = True
            augmented_config.augmentation_variants = augmentation_variants
            augmented_config.preserve_background = preserve_background
            augmented_config.random_seed = augmentation_seed

        if enable_counterfactuals:
            augmented_config.enable_counterfactuals = True
            augmented_config.counterfactual_transform = counterfactual_transform

        # Create augmented dataset with both augmentations
        task_indices = (
            selected_task_indices
            if selected_task_indices
            else list(dataset.valid_tasks)
        )
        dataset = TaskSubset(
            task_indices=task_indices,
            config=augmented_config,
            arc_agi1_dir=config.arc_agi1_dir,
            holdout=True,
            use_first_combination_only=False,
        )

    with torch.no_grad():
        # Get the task indices to iterate over
        task_indices_to_process = (
            selected_task_indices
            if selected_task_indices
            else list(dataset.valid_tasks)
        )

        for task_idx in task_indices_to_process:
            if progress_bar:
                progress_bar.progress(
                    (task_indices_to_process.index(task_idx) + 1)
                    / len(task_indices_to_process)
                )

            # get the task data
            task_data = dataset[task_indices_to_process.index(task_idx)]

            # get the global task index for display
            global_task_index = task_idx

            # get all possible combinations for this task using the new method
            task_combinations_data = dataset.get_task_combinations(task_idx)
            task_combinations = task_combinations_data["all"]

            task_results = []
            for combo in task_combinations:
                # Extract combination data
                combo_idx = combo["combination_idx"]
                i, j = combo["pair_indices"]
                is_counterfactual = combo["is_counterfactual"]

                # Get the preprocessed data from the combination
                rule_latent_examples = combo["rule_latent_examples"]
                test_examples = combo["test_examples"]
                num_test_examples = combo["num_test_examples"]
                holdout_example = combo.get("holdout_example")

                # Apply noise to training examples if requested
                if (
                    noise_config.noise_a_input
                    or noise_config.noise_a_output
                    or noise_config.noise_b_input
                    or noise_config.noise_b_output
                ):
                    rule_latent_examples = apply_noise_to_examples(
                        rule_latent_examples, noise_config
                    )

                # Apply noise to test inputs if requested
                if noise_config.noise_test_inputs:
                    test_examples = apply_noise_to_test_inputs(
                        test_examples,
                        noise_config.noise_test_type,
                        noise_config.noise_test_std,
                        noise_config.noise_test_range,
                        noise_config.noise_test_ratio,
                    )

                # Extract the preprocessed tensors
                example1_input = rule_latent_examples[0][
                    "input"
                ]  # Already has batch dimension
                example1_output = rule_latent_examples[0]["output"]
                example2_input = rule_latent_examples[1]["input"]
                example2_output = rule_latent_examples[1]["output"]

                rule_latent = model.encoder(
                    example1_input, example1_output, example2_input, example2_output
                )

                # inject noise into rule latent if requested
                if noise_config.inject_noise:
                    rule_latent = generate_noise_latent(
                        rule_latent,
                        noise_config.noise_type,
                        noise_config.noise_std,
                        noise_config.noise_range,
                        noise_config.noise_ratio,
                    )

                # evaluate on target(s)
                if evaluation_mode == "test":
                    if test_all_test_pairs and len(test_examples) > 1:
                        # Evaluate on all test examples and create separate results for each
                        for test_idx, test_example in enumerate(test_examples):
                            logits = model.decoder(rule_latent, test_example["input"])
                            predictions = torch.argmax(logits, dim=1).squeeze(0)
                            metrics = calculate_accuracy_metrics(
                                predictions, test_example["output"]
                            )

                            # Create separate result for this test example
                            test_sample_data = {
                                "train_examples": [
                                    {
                                        "input": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[0]["input"]
                                            )
                                        ),
                                        "output": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[0]["output"]
                                            )
                                        ),
                                    },
                                    {
                                        "input": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[1]["input"]
                                            )
                                        ),
                                        "output": tensor_to_numpy(
                                            denormalize_rgb(
                                                rule_latent_examples[1]["output"]
                                            )
                                        ),
                                    },
                                ],
                                "test_examples": [
                                    {
                                        "input": tensor_to_grayscale_numpy(
                                            test_ex["input"]
                                        ),
                                        "output": tensor_to_grayscale_numpy(
                                            test_ex["output"]
                                        ),
                                    }
                                    for test_ex in test_examples
                                ],
                                "num_test_examples": num_test_examples,
                            }

                            # Add holdout data if available
                            if holdout_example is not None:
                                test_sample_data["holdout_example"] = {
                                    "input": tensor_to_grayscale_numpy(
                                        holdout_example["input"]
                                    ),
                                    "output": tensor_to_grayscale_numpy(
                                        holdout_example["output"]
                                    ),
                                }

                            task_results.append(
                                {
                                    "combination_idx": combo_idx,
                                    "pair_indices": (i, j),
                                    "is_counterfactual": is_counterfactual,
                                    "test_example_idx": test_idx,
                                    "perfect_accuracy": metrics["perfect_accuracy"],
                                    "pixel_accuracy": metrics["pixel_accuracy"],
                                    "near_miss_accuracy": metrics["near_miss_accuracy"],
                                    "predictions": predictions,
                                    "logits": logits,
                                    "sample_data": test_sample_data,
                                }
                            )

                        # Skip the normal result creation since we created separate results above
                        continue
                    else:
                        # Use first test example (original behavior)
                        target = test_examples[0]
                        logits = model.decoder(rule_latent, target["input"])
                        predictions = torch.argmax(logits, dim=1).squeeze(0)
                        metrics = calculate_accuracy_metrics(
                            predictions, target["output"]
                        )
                elif evaluation_mode == "holdout" and holdout_example is not None:
                    target = holdout_example
                    logits = model.decoder(rule_latent, target["input"])
                    predictions = torch.argmax(logits, dim=1).squeeze(0)
                    metrics = calculate_accuracy_metrics(predictions, target["output"])
                else:
                    # Fallback to first test example
                    target = test_examples[0]
                logits = model.decoder(rule_latent, target["input"])
                predictions = torch.argmax(logits, dim=1).squeeze(0)
                metrics = calculate_accuracy_metrics(predictions, target["output"])

                # Create sample data for visualization using the same approach as view_dataset.py
                # Create visualization data
                sample_data = {
                    "train_examples": [
                        {
                            "input": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[0]["input"])
                            ),  # [1, 3, 64, 64] -> [64, 64, 3]
                            "output": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[0]["output"])
                            ),
                        },
                        {
                            "input": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[1]["input"])
                            ),
                            "output": tensor_to_numpy(
                                denormalize_rgb(rule_latent_examples[1]["output"])
                            ),
                        },
                    ],
                    "test_examples": [
                        {
                            "input": tensor_to_grayscale_numpy(test_example["input"]),
                            "output": tensor_to_grayscale_numpy(test_example["output"]),
                        }
                        for test_example in test_examples
                    ],
                    "num_test_examples": num_test_examples,
                }

                # Add holdout data if available
                if holdout_example is not None:
                    sample_data["holdout_example"] = {
                        "input": tensor_to_grayscale_numpy(holdout_example["input"]),
                        "output": tensor_to_grayscale_numpy(holdout_example["output"]),
                    }

                task_results.append(
                    {
                        "combination_idx": combo_idx,
                        "pair_indices": (i, j),
                        "is_counterfactual": is_counterfactual,  # Add counterfactual flag
                        "perfect_accuracy": metrics["perfect_accuracy"],
                        "pixel_accuracy": metrics["pixel_accuracy"],
                        "near_miss_accuracy": metrics["near_miss_accuracy"],
                        "predictions": predictions,
                        "logits": logits,
                        "sample_data": sample_data,  # Add visualization data
                    }
                )

            # Return individual combination results (same format as evaluate_model_on_tasks)
            for combo_result in task_results:
                results.append(
                    {
                        "task_idx": task_idx,
                        "global_task_index": global_task_index,
                        "task_id": task_data["task_id"],
                        "combination_idx": combo_result["combination_idx"],
                        "pair_indices": combo_result["pair_indices"],
                        "is_counterfactual": combo_result["is_counterfactual"],
                        "test_example_idx": combo_result.get(
                            "test_example_idx"
                        ),  # Add test example index
                        "perfect_accuracy": combo_result["perfect_accuracy"],
                        "pixel_accuracy": combo_result["pixel_accuracy"],
                        "near_miss_accuracy": combo_result["near_miss_accuracy"],
                        "predictions": combo_result["predictions"],
                        "logits": combo_result["logits"],
                        "sample_data": combo_result["sample_data"],
                        "evaluation_mode": evaluation_mode,
                    }
                )

    return results


def main():
    """main streamlit app."""
    st.title("🤖 arc model predictions")
    st.markdown("visualize model predictions from overfitting experiments")

    st.sidebar.header("experiment controls")
    logs_dir = Path("logs")
    if not logs_dir.exists():
        st.error("❌ logs directory not found")
        st.stop()

    experiments = get_available_experiments(logs_dir)
    if not experiments:
        st.error("❌ no overfitting experiments found")
        st.stop()
    experiment_names = [name for name, _ in experiments]
    selected_exp_name = st.sidebar.selectbox(
        "select experiment", experiment_names, index=0
    )
    selected_exp_path = next(
        path for name, path in experiments if name == selected_exp_name
    )

    exp_info = load_experiment_info(selected_exp_path)
    st.sidebar.subheader("experiment info")
    if "training" in exp_info:
        st.sidebar.write(
            f"**best epoch:** {exp_info['training'].get('best_epoch', 'n/a')}"
        )
        best_loss = exp_info["training"].get("best_loss", "n/a")
        if (
            best_loss is not None
            and best_loss != "n/a"
            and isinstance(best_loss, (int, float))
        ):
            st.sidebar.write(f"**best loss:** {best_loss:.4f}")
        else:
            st.sidebar.write(
                f"**best loss:** {best_loss if best_loss is not None else 'n/a'}"
            )
        st.sidebar.write(
            f"**total epochs:** {exp_info['training'].get('total_epochs', 'n/a')}"
        )

    if "tasks" in exp_info:
        st.sidebar.write(f"**tasks:** {exp_info['tasks'].get('n_tasks', 'n/a')}")
        task_indices = exp_info["tasks"].get("task_indices", [])
        sorted_indices = sorted(task_indices) if task_indices else []
        st.sidebar.write(f"**task indices:** {sorted_indices}")

    st.sidebar.subheader("task set selection")
    task_set = st.sidebar.radio(
        "evaluate on:",
        ["overfit tasks only", "all test tasks"],
        index=0,
    )

    model_path = selected_exp_path / "best_model.pt"
    if not model_path.exists():
        st.error(f"❌ model checkpoint not found: {model_path}")
        st.stop()

    try:
        # Load checkpoint first to get the config
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Load config from checkpoint if available, otherwise use default
        if "config" in checkpoint:
            config = checkpoint["config"]
            st.sidebar.info("✅ loaded config from checkpoint")
        else:
            config = Config()
            st.sidebar.warning("⚠️ no config in checkpoint, using default config")

        # Create model with the loaded config
        model = SimpleARCModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        st.sidebar.success("✅ model loaded successfully")
    except Exception as e:
        st.error(f"❌ failed to load model: {e}")
        st.stop()

    config.use_color_relabeling = False
    config.enable_counterfactuals = False
    dataset = ARCDataset(
        config.arc_agi1_dir, config, holdout=True, use_first_combination_only=False
    )
    if task_set == "overfit tasks only":
        if "tasks" in exp_info and "task_indices" in exp_info["tasks"]:
            task_indices = exp_info["tasks"]["task_indices"]
            # Create a task subset for evaluation (will be updated later based on options)
            dataset = TaskSubset(
                task_indices=task_indices,
                config=dataset.config,
                arc_agi1_dir=str(dataset.raw_data_dir),
                holdout=True,
                use_first_combination_only=True,  # Default, will be updated later
            )
            st.sidebar.write(f"**evaluating on {len(task_indices)} overfit tasks**")
            st.sidebar.write(f"**task indices:** {sorted(task_indices)}")
        else:
            st.error("❌ no task indices found in experiment info")
            st.stop()
    else:
        st.sidebar.write(f"**evaluating on all {len(dataset)} test tasks**")

    st.sidebar.subheader("evaluation options")
    evaluation_mode = st.sidebar.selectbox(
        "evaluation mode",
        ["test", "holdout"],
        index=0,
        help="test: evaluate on test targets, holdout: evaluate on holdout targets",
    )

    test_combinations = st.sidebar.checkbox(
        "test all combinations",
        value=False,
        help="test all possible combinations of train examples for rule latent creation",
    )

    test_all_test_pairs = st.sidebar.checkbox(
        "test all test pairs",
        value=False,
        help="evaluate on all test examples for each task/combination",
    )

    st.sidebar.subheader("rule latent analysis")
    inject_noise = st.sidebar.checkbox(
        "inject noise into rule latent",
        value=False,
        help="replace rule latent with noise to test if it's actually useful",
    )

    noise_type = "gaussian"
    noise_std = 1.0
    noise_range = 1.0
    noise_ratio = 1.0

    if inject_noise:
        noise_type = st.sidebar.selectbox(
            "noise type",
            ["gaussian", "uniform", "zeros", "ones"],
            index=0,
            help="type of noise to inject",
        )

        if noise_type == "gaussian":
            noise_std = st.sidebar.slider(
                "noise std",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="standard deviation for gaussian noise",
            )
        elif noise_type == "uniform":
            noise_range = st.sidebar.slider(
                "noise range",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="range for uniform noise [-range, +range]",
            )

        noise_ratio = st.sidebar.slider(
            "noise ratio",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="fraction of rule latent to replace with noise (1.0 = full replacement)",
        )

    st.sidebar.subheader("support example noise")

    # Support A input noise
    noise_a_input = st.sidebar.checkbox(
        "noise support A input",
        value=False,
        help="inject noise into support example A input to test model robustness",
    )

    noise_a_input_type = "gaussian"
    noise_a_input_std = 1.0
    noise_a_input_range = 1.0
    noise_a_input_ratio = 1.0

    if noise_a_input:
        noise_a_input_type = st.sidebar.selectbox(
            "A input noise type",
            ["gaussian", "uniform", "zeros", "ones"],
            index=0,
            help="type of noise to inject into support A input",
        )

        if noise_a_input_type == "gaussian":
            noise_a_input_std = st.sidebar.slider(
                "A input noise std",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="standard deviation for gaussian noise on support A input",
            )
        elif noise_a_input_type == "uniform":
            noise_a_input_range = st.sidebar.slider(
                "A input noise range",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="range for uniform noise on support A input [-range, +range]",
            )

        noise_a_input_ratio = st.sidebar.slider(
            "A input noise ratio",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="fraction of support A input to replace with noise (1.0 = full replacement)",
        )

    # Support A output noise
    noise_a_output = st.sidebar.checkbox(
        "noise support A output",
        value=False,
        help="inject noise into support example A output to test model robustness",
    )

    noise_a_output_type = "gaussian"
    noise_a_output_std = 1.0
    noise_a_output_range = 1.0
    noise_a_output_ratio = 1.0

    if noise_a_output:
        noise_a_output_type = st.sidebar.selectbox(
            "A output noise type",
            ["gaussian", "uniform", "zeros", "ones"],
            index=0,
            help="type of noise to inject into support A output",
        )

        if noise_a_output_type == "gaussian":
            noise_a_output_std = st.sidebar.slider(
                "A output noise std",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="standard deviation for gaussian noise on support A output",
            )
        elif noise_a_output_type == "uniform":
            noise_a_output_range = st.sidebar.slider(
                "A output noise range",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="range for uniform noise on support A output [-range, +range]",
            )

        noise_a_output_ratio = st.sidebar.slider(
            "A output noise ratio",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="fraction of support A output to replace with noise (1.0 = full replacement)",
        )

    # Support B input noise
    noise_b_input = st.sidebar.checkbox(
        "noise support B input",
        value=False,
        help="inject noise into support example B input to test model robustness",
    )

    noise_b_input_type = "gaussian"
    noise_b_input_std = 1.0
    noise_b_input_range = 1.0
    noise_b_input_ratio = 1.0

    if noise_b_input:
        noise_b_input_type = st.sidebar.selectbox(
            "B input noise type",
            ["gaussian", "uniform", "zeros", "ones"],
            index=0,
            help="type of noise to inject into support B input",
        )

        if noise_b_input_type == "gaussian":
            noise_b_input_std = st.sidebar.slider(
                "B input noise std",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="standard deviation for gaussian noise on support B input",
            )
        elif noise_b_input_type == "uniform":
            noise_b_input_range = st.sidebar.slider(
                "B input noise range",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="range for uniform noise on support B input [-range, +range]",
            )

        noise_b_input_ratio = st.sidebar.slider(
            "B input noise ratio",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="fraction of support B input to replace with noise (1.0 = full replacement)",
        )

    # Support B output noise
    noise_b_output = st.sidebar.checkbox(
        "noise support B output",
        value=False,
        help="inject noise into support example B output to test model robustness",
    )

    noise_b_output_type = "gaussian"
    noise_b_output_std = 1.0
    noise_b_output_range = 1.0
    noise_b_output_ratio = 1.0

    if noise_b_output:
        noise_b_output_type = st.sidebar.selectbox(
            "B output noise type",
            ["gaussian", "uniform", "zeros", "ones"],
            index=0,
            help="type of noise to inject into support B output",
        )

        if noise_b_output_type == "gaussian":
            noise_b_output_std = st.sidebar.slider(
                "B output noise std",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="standard deviation for gaussian noise on support B output",
            )
        elif noise_b_output_type == "uniform":
            noise_b_output_range = st.sidebar.slider(
                "B output noise range",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="range for uniform noise on support B output [-range, +range]",
            )

        noise_b_output_ratio = st.sidebar.slider(
            "B output noise ratio",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="fraction of support B output to replace with noise (1.0 = full replacement)",
        )

    st.sidebar.subheader("test input noise")
    noise_test_inputs = st.sidebar.checkbox(
        "noise test inputs",
        value=False,
        help="inject noise into test inputs to test model robustness",
    )

    noise_test_type = "gaussian"
    noise_test_std = 1.0
    noise_test_range = 1.0
    noise_test_ratio = 1.0

    if noise_test_inputs:
        noise_test_type = st.sidebar.selectbox(
            "test noise type",
            ["gaussian", "uniform", "zeros", "ones"],
            index=0,
            help="type of noise to inject into test inputs",
        )

        if noise_test_type == "gaussian":
            noise_test_std = st.sidebar.slider(
                "test noise std",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="standard deviation for gaussian noise on test inputs",
            )
        elif noise_test_type == "uniform":
            noise_test_range = st.sidebar.slider(
                "test noise range",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="range for uniform noise on test inputs [-range, +range]",
            )

        noise_test_ratio = st.sidebar.slider(
            "test noise ratio",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="fraction of test inputs to replace with noise (1.0 = full replacement)",
        )

    st.sidebar.subheader("color augmentation")
    enable_color_augmentation = st.sidebar.checkbox(
        "enable color relabeling",
        value=False,
        help="apply color relabeling to test model robustness",
    )

    augmentation_variants = 1
    preserve_background = True
    augmentation_seed = 42

    if enable_color_augmentation:
        augmentation_variants = st.sidebar.slider(
            "augmentation variants",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="number of color-relabeled versions per example",
        )

        preserve_background = st.sidebar.checkbox(
            "preserve background",
            value=True,
            help="keep background color (0) unchanged during relabeling",
        )

        augmentation_seed = st.sidebar.number_input(
            "augmentation seed",
            min_value=0,
            max_value=10000,
            value=42,
            step=1,
            help="random seed for reproducible color relabeling",
        )

    st.sidebar.subheader("counterfactual analysis")
    enable_counterfactuals = st.sidebar.checkbox(
        "enable counterfactuals",
        value=False,
        help="include counterfactual (rotated) examples in evaluation",
    )

    counterfactual_transform = "rotate_90"
    if enable_counterfactuals:
        counterfactual_transform = st.sidebar.selectbox(
            "counterfactual transform",
            ["rotate_90", "rotate_180", "rotate_270", "reflect_h", "reflect_v"],
            index=0,
            help="type of transformation to apply to outputs",
        )

    # Update dataset based on options if needed
    if task_set == "overfit tasks only" and test_combinations:
        # Recreate dataset with all combinations when testing all combinations
        if "tasks" in exp_info and "task_indices" in exp_info["tasks"]:
            task_indices = exp_info["tasks"]["task_indices"]
            dataset = TaskSubset(
                task_indices=task_indices,
                config=dataset.config,
                arc_agi1_dir=str(dataset.raw_data_dir),
                holdout=True,
                use_first_combination_only=False,  # Use all combinations
            )
            if test_all_test_pairs:
                st.sidebar.write("**mode:** all combinations, all test pairs")
            else:
                st.sidebar.write("**mode:** all combinations")
    elif (
        task_set == "overfit tasks only"
        and test_all_test_pairs
        and not test_combinations
    ):
        # When testing all test pairs but not all combinations, use first combination only
        st.sidebar.write("**mode:** first combination, all test pairs")

    if st.sidebar.button("🚀 evaluate model", type="primary"):
        if "evaluation_results" in st.session_state:
            del st.session_state.evaluation_results
        if "combination_results" in st.session_state:
            del st.session_state.combination_results

        progress_bar = st.progress(0)
        status_text = st.empty()

        if test_combinations:
            status_text.text("testing all combinations...")
            # For testing all combinations, we need to create a task subset with all combinations
            if task_set == "overfit tasks only":
                if "tasks" in exp_info and "task_indices" in exp_info["tasks"]:
                    task_indices = exp_info["tasks"]["task_indices"]
                    # Create a task subset with all combinations for testing
                    test_dataset = TaskSubset(
                        task_indices=task_indices,
                        config=dataset.config,
                        arc_agi1_dir=str(dataset.raw_data_dir),
                        holdout=True,
                        use_first_combination_only=False,
                    )
                else:
                    st.error("❌ no task indices found in experiment info")
                    st.stop()
            else:
                test_dataset = dataset

            # Get task indices for the test dataset
            if task_set == "overfit tasks only":
                task_indices = (
                    exp_info["tasks"]["task_indices"]
                    if "tasks" in exp_info and "task_indices" in exp_info["tasks"]
                    else None
                )
            else:
                task_indices = None

            noise_config = create_noise_config_from_ui(
                inject_noise,
                noise_type,
                noise_std,
                noise_range,
                noise_ratio,
                noise_a_input,
                noise_a_input_type,
                noise_a_input_std,
                noise_a_input_range,
                noise_a_input_ratio,
                noise_a_output,
                noise_a_output_type,
                noise_a_output_std,
                noise_a_output_range,
                noise_a_output_ratio,
                noise_b_input,
                noise_b_input_type,
                noise_b_input_std,
                noise_b_input_range,
                noise_b_input_ratio,
                noise_b_output,
                noise_b_output_type,
                noise_b_output_std,
                noise_b_output_range,
                noise_b_output_ratio,
                noise_test_inputs,
                noise_test_type,
                noise_test_std,
                noise_test_range,
                noise_test_ratio,
            )

            results = test_all_combinations(
                model,
                test_dataset,
                config,
                evaluation_mode,
                progress_bar,
                noise_config,
                enable_color_augmentation,
                augmentation_variants,
                preserve_background,
                augmentation_seed,
                enable_counterfactuals,
                counterfactual_transform,
                selected_task_indices=task_indices,
                test_all_test_pairs=test_all_test_pairs,
            )
            st.session_state.combination_results = results
        else:
            status_text.text("evaluating model on tasks...")
            # Get task indices for the dataset
            if task_set == "overfit tasks only":
                task_indices = (
                    exp_info["tasks"]["task_indices"]
                    if "tasks" in exp_info and "task_indices" in exp_info["tasks"]
                    else None
                )
            else:
                task_indices = None

            noise_config = create_noise_config_from_ui(
                inject_noise,
                noise_type,
                noise_std,
                noise_range,
                noise_ratio,
                noise_a_input,
                noise_a_input_type,
                noise_a_input_std,
                noise_a_input_range,
                noise_a_input_ratio,
                noise_a_output,
                noise_a_output_type,
                noise_a_output_std,
                noise_a_output_range,
                noise_a_output_ratio,
                noise_b_input,
                noise_b_input_type,
                noise_b_input_std,
                noise_b_input_range,
                noise_b_input_ratio,
                noise_b_output,
                noise_b_output_type,
                noise_b_output_std,
                noise_b_output_range,
                noise_b_output_ratio,
                noise_test_inputs,
                noise_test_type,
                noise_test_std,
                noise_test_range,
                noise_test_ratio,
            )

            results = evaluate_model_on_tasks(
                model,
                dataset,
                config,
                evaluation_mode,
                progress_bar,
                noise_config,
                enable_color_augmentation,
                augmentation_variants,
                preserve_background,
                augmentation_seed,
                enable_counterfactuals,
                counterfactual_transform,
                selected_task_indices=task_indices,
                test_all_test_pairs=test_all_test_pairs,
            )
            st.session_state.evaluation_results = results

        st.session_state.task_set = task_set
        st.session_state.evaluation_mode = evaluation_mode
        st.session_state.test_combinations = test_combinations
        st.session_state.test_all_test_pairs = test_all_test_pairs
        st.session_state.inject_noise = inject_noise
        st.session_state.noise_type = noise_type
        st.session_state.noise_std = noise_std
        st.session_state.noise_range = noise_range
        st.session_state.noise_ratio = noise_ratio
        st.session_state.noise_a_input = noise_a_input
        st.session_state.noise_a_input_type = noise_a_input_type
        st.session_state.noise_a_input_std = noise_a_input_std
        st.session_state.noise_a_input_range = noise_a_input_range
        st.session_state.noise_a_input_ratio = noise_a_input_ratio
        st.session_state.noise_a_output = noise_a_output
        st.session_state.noise_a_output_type = noise_a_output_type
        st.session_state.noise_a_output_std = noise_a_output_std
        st.session_state.noise_a_output_range = noise_a_output_range
        st.session_state.noise_a_output_ratio = noise_a_output_ratio
        st.session_state.noise_b_input = noise_b_input
        st.session_state.noise_b_input_type = noise_b_input_type
        st.session_state.noise_b_input_std = noise_b_input_std
        st.session_state.noise_b_input_range = noise_b_input_range
        st.session_state.noise_b_input_ratio = noise_b_input_ratio
        st.session_state.noise_b_output = noise_b_output
        st.session_state.noise_b_output_type = noise_b_output_type
        st.session_state.noise_b_output_std = noise_b_output_std
        st.session_state.noise_b_output_range = noise_b_output_range
        st.session_state.noise_b_output_ratio = noise_b_output_ratio
        st.session_state.noise_test_inputs = noise_test_inputs
        st.session_state.noise_test_type = noise_test_type
        st.session_state.noise_test_std = noise_test_std
        st.session_state.noise_test_range = noise_test_range
        st.session_state.noise_test_ratio = noise_test_ratio
        st.session_state.enable_color_augmentation = enable_color_augmentation
        st.session_state.augmentation_variants = augmentation_variants
        st.session_state.preserve_background = preserve_background
        st.session_state.augmentation_seed = augmentation_seed
        st.session_state.enable_counterfactuals = enable_counterfactuals
        st.session_state.counterfactual_transform = counterfactual_transform

        progress_bar.empty()
        status_text.text(f"evaluation complete on {len(results)} tasks!")

        # Debug: Show which tasks have multiple test examples
        if st.session_state.get("test_all_test_pairs", False):
            multi_test_tasks = []
            for result in results:
                sample_data = result.get("sample_data", {})
                num_test_examples = sample_data.get("num_test_examples", 1)
                if num_test_examples > 1:
                    multi_test_tasks.append(
                        {
                            "task_id": result["task_id"],
                            "task_idx": result.get(
                                "global_task_index", result.get("task_idx")
                            ),
                            "num_test_examples": num_test_examples,
                        }
                    )

            if multi_test_tasks:
                st.sidebar.success(
                    f"✅ Found {len(multi_test_tasks)} multi-test tasks:"
                )
                for task in multi_test_tasks:
                    st.sidebar.write(
                        f"  - Task {task['task_id']} (idx {task['task_idx']}): {task['num_test_examples']} test examples"
                    )
            else:
                st.sidebar.warning("⚠️ No multi-test tasks found in evaluation results")

        st.rerun()

    if "evaluation_results" in st.session_state:
        results = st.session_state.evaluation_results
        current_task_set = st.session_state.get("task_set", "unknown")

        # show noise info if applicable
        noise_info = ""
        if (
            st.session_state.get("inject_noise", False)
            or st.session_state.get("noise_a_input", False)
            or st.session_state.get("noise_a_output", False)
            or st.session_state.get("noise_b_input", False)
            or st.session_state.get("noise_b_output", False)
            or st.session_state.get("noise_test_inputs", False)
        ):
            # Create NoiseConfig from session state for display
            noise_config = create_noise_config_from_ui(
                st.session_state.get("inject_noise", False),
                st.session_state.get("noise_type", "gaussian"),
                st.session_state.get("noise_std", 1.0),
                st.session_state.get("noise_range", 1.0),
                st.session_state.get("noise_ratio", 1.0),
                st.session_state.get("noise_a_input", False),
                st.session_state.get("noise_a_input_type", "gaussian"),
                st.session_state.get("noise_a_input_std", 1.0),
                st.session_state.get("noise_a_input_range", 1.0),
                st.session_state.get("noise_a_input_ratio", 1.0),
                st.session_state.get("noise_a_output", False),
                st.session_state.get("noise_a_output_type", "gaussian"),
                st.session_state.get("noise_a_output_std", 1.0),
                st.session_state.get("noise_a_output_range", 1.0),
                st.session_state.get("noise_a_output_ratio", 1.0),
                st.session_state.get("noise_b_input", False),
                st.session_state.get("noise_b_input_type", "gaussian"),
                st.session_state.get("noise_b_input_std", 1.0),
                st.session_state.get("noise_b_input_range", 1.0),
                st.session_state.get("noise_b_input_ratio", 1.0),
                st.session_state.get("noise_b_output", False),
                st.session_state.get("noise_b_output_type", "gaussian"),
                st.session_state.get("noise_b_output_std", 1.0),
                st.session_state.get("noise_b_output_range", 1.0),
                st.session_state.get("noise_b_output_ratio", 1.0),
                st.session_state.get("noise_test_inputs", False),
                st.session_state.get("noise_test_type", "gaussian"),
                st.session_state.get("noise_test_std", 1.0),
                st.session_state.get("noise_test_range", 1.0),
                st.session_state.get("noise_test_ratio", 1.0),
            )
            noise_info = noise_config.get_noise_info_string()

        # show counterfactual info if applicable
        counterfactual_info = ""
        if st.session_state.get("enable_counterfactuals", False):
            counterfactual_transform = st.session_state.get(
                "counterfactual_transform", "rotate_90"
            )
            counterfactual_info = f" (counterfactuals: {counterfactual_transform})"

        # Add test pairs info if applicable
        test_pairs_info = ""
        if st.session_state.get("test_all_test_pairs", False):
            test_pairs_info = " (all test pairs)"

        st.subheader(
            f"📊 evaluation results ({current_task_set}){test_pairs_info}{noise_info}{counterfactual_info}"
        )

        # create results dataframe
        df_data = []
        for result in results:
            task_id_display = result["task_id"]
            if result.get("is_counterfactual", False):
                task_id_display += " (counterfactual)"

            # Add combination info if available
            combination_info = ""
            if "combination_idx" in result:
                combination_info = f" - combo {result['combination_idx']}"
                if "pair_indices" in result:
                    combination_info += f" {result['pair_indices']}"

            # Add test example info if available
            test_example_info = ""
            if "test_example_idx" in result:
                test_example_info = f" - test {result['test_example_idx']}"

            df_data.append(
                {
                    "idx": result["global_task_index"],
                    "task_id": task_id_display + combination_info + test_example_info,
                    "perfect": f"{result['perfect_accuracy']:.3f}",
                    "pixel": f"{result['pixel_accuracy']:.3f}",
                    "near_miss": f"{result['near_miss_accuracy']:.3f}",
                    "status": (
                        "✅ perfect"
                        if result["perfect_accuracy"] == 1.0
                        else "⚠️ partial"
                        if result["pixel_accuracy"] > 0.5
                        else "❌ failed"
                    ),
                }
            )

        df = pd.DataFrame(df_data)

        # display summary stats
        if st.session_state.get("test_all_test_pairs", False):
            col1, col2, col3, col4, col5 = st.columns(5)
        else:
            col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("total tasks", len(results))
        with col2:
            perfect_count = sum(1 for r in results if r["perfect_accuracy"] == 1.0)
            st.metric("perfect tasks", f"{perfect_count}/{len(results)}")
        with col3:
            avg_pixel = np.mean([r["pixel_accuracy"] for r in results])
            st.metric("avg pixel accuracy", f"{avg_pixel:.3f}")
        with col4:
            avg_near_miss = np.mean([r["near_miss_accuracy"] for r in results])
            st.metric("avg near-miss", f"{avg_near_miss:.3f}")

        # Add additional test pairs stats if enabled
        if st.session_state.get("test_all_test_pairs", False):
            with col5:
                # Count tasks with multiple test examples
                multi_test_tasks = 0
                multi_test_perfect = 0
                for result in results:
                    sample_data = result.get("sample_data", {})
                    num_test_examples = sample_data.get("num_test_examples", 1)
                    if num_test_examples > 1:
                        multi_test_tasks += 1
                        if result["perfect_accuracy"] == 1.0:
                            multi_test_perfect += 1

                if multi_test_tasks > 0:
                    st.metric(
                        "multi-test tasks", f"{multi_test_perfect}/{multi_test_tasks}"
                    )
                else:
                    st.metric("multi-test tasks", "0/0")

        # interactive table
        st.subheader("📋 task results table")
        st.markdown("click on a row to visualize that task")

        # use st.dataframe with selection
        selected_rows = st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # handle row selection
        if selected_rows.selection.rows:
            selected_idx = selected_rows.selection.rows[0]
            selected_task = results[selected_idx]

            st.subheader(f"🔍 visualizing task {selected_task['task_idx']}")

            # visualize the selected task
            if "sample_data" in selected_task:
                # New format - direct sample data
                sample_data = selected_task["sample_data"]
                test_example_idx = selected_task.get("test_example_idx")
                fig = visualize_prediction_comparison(
                    sample_data,
                    selected_task["predictions"],
                    evaluation_mode,
                    test_example_idx,
                )
                st.pyplot(fig)
            else:
                # Old format - extract from batch
                batch = selected_task["batch"]
                sample_idx = selected_task[
                    "batch_sample_idx"
                ]  # Use stored batch sample index

                evaluation_mode = st.session_state.get("evaluation_mode", "test")
                sample_data = extract_sample_from_batch(
                    batch, sample_idx, evaluation_mode
                )
                fig = visualize_prediction_comparison(
                    sample_data, selected_task["predictions"], evaluation_mode
                )
                st.pyplot(fig)

            # show detailed metrics
            st.subheader("📈 detailed metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "perfect accuracy", f"{selected_task['perfect_accuracy']:.3f}"
                )
            with col2:
                st.metric("pixel accuracy", f"{selected_task['pixel_accuracy']:.3f}")
            with col3:
                st.metric(
                    "near miss accuracy", f"{selected_task['near_miss_accuracy']:.3f}"
                )

    # display combination results if available
    elif (
        "combination_results" in st.session_state
        or "evaluation_results" in st.session_state
    ):
        results = st.session_state.get(
            "combination_results", st.session_state.get("evaluation_results", [])
        )
        current_task_set = st.session_state.get("task_set", "unknown")
        evaluation_mode = st.session_state.get("evaluation_mode", "test")

        # show noise info if applicable
        noise_info = ""
        if (
            st.session_state.get("inject_noise", False)
            or st.session_state.get("noise_a_input", False)
            or st.session_state.get("noise_a_output", False)
            or st.session_state.get("noise_b_input", False)
            or st.session_state.get("noise_b_output", False)
            or st.session_state.get("noise_test_inputs", False)
        ):
            # Create NoiseConfig from session state for display
            noise_config = create_noise_config_from_ui(
                st.session_state.get("inject_noise", False),
                st.session_state.get("noise_type", "gaussian"),
                st.session_state.get("noise_std", 1.0),
                st.session_state.get("noise_range", 1.0),
                st.session_state.get("noise_ratio", 1.0),
                st.session_state.get("noise_a_input", False),
                st.session_state.get("noise_a_input_type", "gaussian"),
                st.session_state.get("noise_a_input_std", 1.0),
                st.session_state.get("noise_a_input_range", 1.0),
                st.session_state.get("noise_a_input_ratio", 1.0),
                st.session_state.get("noise_a_output", False),
                st.session_state.get("noise_a_output_type", "gaussian"),
                st.session_state.get("noise_a_output_std", 1.0),
                st.session_state.get("noise_a_output_range", 1.0),
                st.session_state.get("noise_a_output_ratio", 1.0),
                st.session_state.get("noise_b_input", False),
                st.session_state.get("noise_b_input_type", "gaussian"),
                st.session_state.get("noise_b_input_std", 1.0),
                st.session_state.get("noise_b_input_range", 1.0),
                st.session_state.get("noise_b_input_ratio", 1.0),
                st.session_state.get("noise_b_output", False),
                st.session_state.get("noise_b_output_type", "gaussian"),
                st.session_state.get("noise_b_output_std", 1.0),
                st.session_state.get("noise_b_output_range", 1.0),
                st.session_state.get("noise_b_output_ratio", 1.0),
                st.session_state.get("noise_test_inputs", False),
                st.session_state.get("noise_test_type", "gaussian"),
                st.session_state.get("noise_test_std", 1.0),
                st.session_state.get("noise_test_range", 1.0),
                st.session_state.get("noise_test_ratio", 1.0),
            )
            noise_info = noise_config.get_noise_info_string()

        # show counterfactual info if applicable
        counterfactual_info = ""
        if st.session_state.get("enable_counterfactuals", False):
            counterfactual_transform = st.session_state.get(
                "counterfactual_transform", "rotate_90"
            )
            counterfactual_info = f" (counterfactuals: {counterfactual_transform})"

        # Add test pairs info if applicable
        test_pairs_info = ""
        if st.session_state.get("test_all_test_pairs", False):
            test_pairs_info = " (all test pairs)"

        st.subheader(
            f"🔄 combination test results ({current_task_set}) - {evaluation_mode} mode{test_pairs_info}{noise_info}{counterfactual_info}"
        )

        # create combination results dataframe with consistent indexing
        combo_df_data = []
        combo_index_mapping = []  # Store mapping from dataframe index to (task_idx, combo_idx)

        for result in results:
            # Build combination string
            combination_str = (
                f"({result['pair_indices'][0]}, {result['pair_indices'][1]})"
            )
            if result.get("is_counterfactual", False):
                combination_str += " (counterfactual)"

            # Add test example info if available
            if "test_example_idx" in result:
                combination_str += f" - test {result['test_example_idx']}"

            combo_df_data.append(
                {
                    "idx": result["global_task_index"],  # Global task index
                    "task_id": result["task_id"],  # Task ID (filename)
                    "combination": combination_str,
                    "perfect": f"{result['perfect_accuracy']:.3f}",
                    "pixel": f"{result['pixel_accuracy']:.3f}",
                    "near_miss": f"{result['near_miss_accuracy']:.3f}",
                    "status": (
                        "✅ perfect"
                        if result["perfect_accuracy"] == 1.0
                        else "⚠️ partial"
                        if result["pixel_accuracy"] > 0.5
                        else "❌ failed"
                    ),
                }
            )
            # Store mapping for later lookup
            combo_index_mapping.append(
                (
                    result["global_task_index"],
                    result["combination_idx"],
                    result.get("test_example_idx"),
                )
            )

        combo_df = pd.DataFrame(combo_df_data)

        # display summary stats
        if st.session_state.get("test_all_test_pairs", False):
            col1, col2, col3, col4, col5, col6 = st.columns(6)
        else:
            col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("total combinations", len(combo_df_data))
        with col2:
            perfect_count = sum(1 for r in combo_df_data if float(r["perfect"]) == 1.0)
            st.metric("perfect combinations", f"{perfect_count}/{len(combo_df_data)}")
        with col3:
            # Calculate perfect tasks (tasks with at least one perfect combination)
            perfect_tasks = set()
            for result in results:
                if result["perfect_accuracy"] == 1.0:
                    perfect_tasks.add(result["global_task_index"])
            total_tasks = len(set(r["global_task_index"] for r in results))
            st.metric("perfect tasks", f"{len(perfect_tasks)}/{total_tasks}")
        with col4:
            avg_pixel = np.mean([float(r["pixel"]) for r in combo_df_data])
            st.metric("avg pixel accuracy", f"{avg_pixel:.3f}")
        with col5:
            avg_near_miss = np.mean([float(r["near_miss"]) for r in combo_df_data])
            st.metric("avg near-miss", f"{avg_near_miss:.3f}")

        # Add additional test pairs stats if enabled
        if st.session_state.get("test_all_test_pairs", False):
            with col6:
                # Count combinations with multiple test examples
                multi_test_combos = 0
                multi_test_perfect = 0
                for result in results:
                    sample_data = result.get("sample_data", {})
                    num_test_examples = sample_data.get("num_test_examples", 1)
                    if num_test_examples > 1:
                        multi_test_combos += 1
                        if result["perfect_accuracy"] == 1.0:
                            multi_test_perfect += 1

                if multi_test_combos > 0:
                    st.metric(
                        "multi-test combos", f"{multi_test_perfect}/{multi_test_combos}"
                    )
                else:
                    st.metric("multi-test combos", "0/0")

        # display combination table with selection
        st.subheader("📋 combination results table")
        st.markdown("click on a row to visualize that combination")

        # use st.dataframe with selection
        selected_rows = st.dataframe(
            combo_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"combo_table_{evaluation_mode}_{current_task_set}_{len(combo_df_data)}",  # Add key to force refresh
        )

        # handle row selection for visualization
        if selected_rows.selection.rows:
            selected_idx = selected_rows.selection.rows[0]

            # use the mapping to find the correct combination
            if selected_idx < len(combo_index_mapping):
                global_task_index, combo_idx, test_example_idx = combo_index_mapping[
                    selected_idx
                ]

                # find the selected task and combination
            selected_task = None
            selected_combo = None

            for result in results:
                if (
                    result["global_task_index"] == global_task_index
                    and result["combination_idx"] == combo_idx
                    and result.get("test_example_idx") == test_example_idx
                ):
                    selected_task = result
                    selected_combo = result
                    break

            if selected_combo is not None:
                # Build visualization title
                title = f"🔍 visualizing task {selected_task['task_id']} combination {selected_combo['combination_idx']}"
                if test_example_idx is not None:
                    title += f" test example {test_example_idx}"
                st.subheader(title)

                # visualize the selected combination
                test_example_idx = selected_combo.get("test_example_idx")
                fig = visualize_prediction_comparison(
                    selected_combo["sample_data"],
                    selected_combo["predictions"],
                    evaluation_mode,
                    test_example_idx,
                )
                st.pyplot(fig)

                # show detailed metrics for this combination
                st.subheader("📈 combination metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "perfect accuracy", f"{selected_combo['perfect_accuracy']:.3f}"
                    )
                with col2:
                    st.metric(
                        "pixel accuracy", f"{selected_combo['pixel_accuracy']:.3f}"
                    )
                with col3:
                    st.metric(
                        "near miss accuracy",
                        f"{selected_combo['near_miss_accuracy']:.3f}",
                    )

                # show combination details
                st.subheader("🔧 combination details")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Task ID:** {selected_task['task_id']}")
                    combination_display = f"{selected_combo['pair_indices']}"
                    if selected_combo.get("is_counterfactual", False):
                        combination_display += " (counterfactual)"
                    st.write(f"**Combination:** {combination_display}")
                    st.write(f"**Evaluation Mode:** {evaluation_mode}")
                with col2:
                    st.write(
                        f"**Training Examples Used:** {selected_combo['pair_indices'][0]} and {selected_combo['pair_indices'][1]}"
                    )
                    st.write(
                        f"**Status:** {'✅ perfect' if selected_combo['perfect_accuracy'] == 1.0 else '⚠️ partial' if selected_combo['pixel_accuracy'] > 0.5 else '❌ failed'}"
                    )

        # per-task summary
        st.subheader("📊 per-task summary")
        task_summary = []

        # Group results by task index (global_task_index)
        task_groups = {}
        for result in results:
            global_task_index = result["global_task_index"]
            if global_task_index not in task_groups:
                task_groups[global_task_index] = []
            task_groups[global_task_index].append(result)

        for global_task_index, task_results in task_groups.items():
            task_id = task_results[0]["task_id"]  # Get task_id from first result
            best_perfect = max(result["perfect_accuracy"] for result in task_results)
            best_pixel = max(result["pixel_accuracy"] for result in task_results)
            avg_perfect = np.mean(
                [result["perfect_accuracy"] for result in task_results]
            )
            avg_pixel = np.mean([result["pixel_accuracy"] for result in task_results])

            task_summary.append(
                {
                    "idx": global_task_index,
                    "task_id": task_id,
                    "combinations": len(task_results),
                    "best_perfect": f"{best_perfect:.3f}",
                    "best_pixel": f"{best_pixel:.3f}",
                    "avg_perfect": f"{avg_perfect:.3f}",
                    "avg_pixel": f"{avg_pixel:.3f}",
                }
            )

        task_summary_df = pd.DataFrame(task_summary)
        st.dataframe(task_summary_df, width="stretch", hide_index=True)

    else:
        st.info("👆 click 'evaluate model' to run evaluation and see results")


if __name__ == "__main__":
    main()
