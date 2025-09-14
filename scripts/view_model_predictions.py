#!/usr/bin/env python3
"""
streamlit app for visualizing model predictions from overfitting experiments.

interactive web interface to load checkpoints and visualize model outputs.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

from algo.config import Config
from algo.data import ARCDataset, custom_collate_fn
from algo.models.simple_arc import SimpleARCModel
from torch.utils.data import Subset

# set page config
st.set_page_config(page_title="arc model predictions", page_icon="🤖", layout="wide")

# arc color palette (official 10 colors)
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: fuschia
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: teal
    "#870C25",  # 9: brown
]


def tensor_to_numpy(tensor):
    """convert pytorch tensor to numpy array."""
    if tensor.dim() == 4:  # [b, c, h, w]
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    elif tensor.dim() == 3:  # [c, h, w]
        return tensor.permute(1, 2, 0).cpu().numpy()
    elif tensor.dim() == 2:  # [h, w]
        return tensor.cpu().numpy()
    else:
        return tensor.cpu().numpy()


def tensor_to_grayscale_numpy(tensor):
    """convert pytorch tensor to grayscale numpy array."""
    if tensor.dim() == 4:  # [b, c, h, w]
        return tensor.squeeze(0).squeeze(0).cpu().numpy()
    elif tensor.dim() == 3:  # [c, h, w]
        return tensor.squeeze(0).cpu().numpy()
    elif tensor.dim() == 2:  # [h, w]
        return tensor.cpu().numpy()
    else:
        return tensor.cpu().numpy()


def denormalize_rgb(img_tensor):
    """convert normalized rgb tensor back to [0, 1] range."""
    img = (img_tensor + 1) / 2
    return torch.clamp(img, 0, 1)


def visualize_arc_image(img_tensor, title, is_rgb=True, figsize=(4, 4)):
    """visualize an arc image with proper color mapping."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if is_rgb:
        # rgb image (example images)
        img = denormalize_rgb(img_tensor)
        img_np = tensor_to_numpy(img)
        ax.imshow(img_np)
    else:
        # grayscale image (target images)
        img_np = tensor_to_numpy(img_tensor)
        # create rgb version using arc color palette
        rgb_img = np.zeros((*img_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = img_np == i
            rgb_img[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        ax.imshow(rgb_img)

    ax.set_title(title, fontsize=10)
    ax.axis("off")

    # add grid
    ax.set_xticks(np.arange(-0.5, img_np.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, img_np.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    return fig


def extract_sample_from_batch(batch, sample_idx, evaluation_mode="test"):
    """Extract individual sample data from batched format."""
    sample = {
        "train_examples": [],
        "test_example": {
            "input": batch["test_inputs"][sample_idx],
            "output": batch["test_outputs"][sample_idx],
        },
    }

    # Use holdout data if in holdout mode and available
    if evaluation_mode == "holdout" and batch["has_holdout"][sample_idx]:
        sample["test_example"] = {
            "input": batch["holdout_inputs"][sample_idx],
            "output": batch["holdout_outputs"][sample_idx],
        }

    # Extract training examples - show first 2 examples from all_train_inputs
    num_train = min(2, batch["all_train_inputs"].shape[1])

    for i in range(num_train):
        sample["train_examples"].append(
            {
                "input": batch["all_train_inputs"][sample_idx, i],
                "output": batch["all_train_outputs"][sample_idx, i],
            }
        )

    return sample


def visualize_prediction_comparison(sample, prediction):
    """visualize model predictions compared to ground truth."""

    # create figure with 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("model prediction comparison", fontsize=16)

    # example 1 - handle new data structure with consistent color scheme
    axes[0, 0].set_title("example 1 input", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 1:
        img1_np = tensor_to_grayscale_numpy(sample["train_examples"][0]["input"])
        rgb_img1 = np.zeros((*img1_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = img1_np == i
            rgb_img1[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[0, 0].imshow(rgb_img1)
    else:
        axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 0].axis("off")

    axes[0, 1].set_title("example 1 output", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 1:
        img2_np = tensor_to_grayscale_numpy(sample["train_examples"][0]["output"])
        rgb_img2 = np.zeros((*img2_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = img2_np == i
            rgb_img2[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[0, 1].imshow(rgb_img2)
    else:
        axes[0, 1].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 1].axis("off")

    # example 2
    axes[0, 2].set_title("example 2 input", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 2:
        img3_np = tensor_to_grayscale_numpy(sample["train_examples"][1]["input"])
        rgb_img3 = np.zeros((*img3_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = img3_np == i
            rgb_img3[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[0, 2].imshow(rgb_img3)
    else:
        axes[0, 2].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 2].axis("off")

    axes[0, 3].set_title("example 2 output", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 2:
        img4_np = tensor_to_grayscale_numpy(sample["train_examples"][1]["output"])
        rgb_img4 = np.zeros((*img4_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = img4_np == i
            rgb_img4[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[0, 3].imshow(rgb_img4)
    else:
        axes[0, 3].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 3].axis("off")

    # target input - handle new data structure
    axes[1, 0].set_title("target input", fontsize=12)
    if "test_example" in sample and "input" in sample["test_example"]:
        target_input_np = tensor_to_grayscale_numpy(sample["test_example"]["input"])
        rgb_target_input = np.zeros((*target_input_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = target_input_np == i
            rgb_target_input[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[1, 0].imshow(rgb_target_input)
    else:
        axes[1, 0].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[1, 0].axis("off")

    # ground truth output
    axes[1, 1].set_title("ground truth", fontsize=12)
    if "test_example" in sample and "output" in sample["test_example"]:
        target_output_np = tensor_to_grayscale_numpy(sample["test_example"]["output"])
        rgb_target_output = np.zeros((*target_output_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = target_output_np == i
            rgb_target_output[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[1, 1].imshow(rgb_target_output)
    else:
        axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[1, 1].axis("off")

    # model prediction
    axes[1, 2].set_title("model prediction", fontsize=12)
    pred_np = tensor_to_grayscale_numpy(prediction)
    rgb_pred = np.zeros((*pred_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = pred_np == i
        rgb_pred[mask] = (
            np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
            / 255.0
        )
    axes[1, 2].imshow(rgb_pred)
    axes[1, 2].axis("off")

    # difference visualization
    axes[1, 3].set_title("difference", fontsize=12)
    if "test_example" in sample and "output" in sample["test_example"]:
        target_output_np = tensor_to_grayscale_numpy(sample["test_example"]["output"])
        diff = np.abs(target_output_np.astype(float) - pred_np.astype(float))
        axes[1, 3].imshow(diff, cmap="hot", vmin=0, vmax=9)
    else:
        axes[1, 3].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[1, 3].axis("off")

    # add grid to all subplots
    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 30, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


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


def load_model_checkpoint(checkpoint_path: str, config: Config) -> SimpleARCModel:
    """load model from checkpoint."""
    model = SimpleARCModel(config)

    checkpoint = torch.load(
        checkpoint_path, map_location=config.device, weights_only=False
    )
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
    inject_noise=False,
    noise_type="gaussian",
    noise_std=1.0,
    noise_range=1.0,
    noise_ratio=1.0,
    enable_color_augmentation=False,
    augmentation_variants=1,
    preserve_background=True,
    augmentation_seed=42,
):
    """evaluate model on all tasks in dataset and return results.

    Args:
        model: The model to evaluate
        dataset: Dataset to evaluate on
        config: Configuration object
        evaluation_mode: "test" for test targets, "holdout" for holdout targets
        progress_bar: Streamlit progress bar
        inject_noise: Whether to inject noise into rule latent
        noise_type: Type of noise ("gaussian", "uniform", "zeros", "ones")
        noise_std: Standard deviation for gaussian noise
        noise_range: Range for uniform noise
        noise_ratio: Fraction of rule latent to replace with noise
        enable_color_augmentation: Whether to enable color relabeling
        augmentation_variants: Number of augmented versions per example
        preserve_background: Whether to preserve background color
        augmentation_seed: Random seed for augmentation
    """
    from torch.utils.data import DataLoader

    results = []

    # Create augmented dataset if color augmentation is enabled
    if enable_color_augmentation:
        # Create a copy of the config with augmentation enabled
        augmented_config = Config()
        augmented_config.__dict__.update(config.__dict__)  # Copy all config values
        augmented_config.use_color_relabeling = True
        augmented_config.augmentation_variants = augmentation_variants
        augmented_config.preserve_background = preserve_background
        augmented_config.random_seed = augmentation_seed

        # Create augmented dataset
        from algo.data import ARCDataset

        augmented_dataset = ARCDataset(
            config.arc_agi1_dir,
            augmented_config,
            holdout=True,
            use_first_combination_only=True,
        )

        # If we're using a subset, create the same subset for augmented dataset
        if hasattr(dataset, "dataset"):
            # Get the original indices
            original_indices = dataset.indices
            augmented_dataset = Subset(augmented_dataset, original_indices)

        dataset = augmented_dataset

    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn
    )

    # Set deterministic training for reproducible results
    config.set_deterministic_training()

    with torch.no_grad():
        global_task_idx = 0  # Track actual task index across all batches
        for i, batch in enumerate(dataloader):
            if progress_bar:
                progress_bar.progress((i + 1) / len(dataloader))

            # run model inference with rule latent training
            outputs = model.forward_rule_latent_training(
                batch["rule_latent_inputs"],
                batch["all_train_inputs"],
                batch["num_train"],
            )

            # inject noise into rule latents if requested
            if inject_noise:
                for j in range(outputs["rule_latents"].size(0)):
                    original_latent = outputs["rule_latents"][j : j + 1]
                    noisy_latent = generate_noise_latent(
                        original_latent, noise_type, noise_std, noise_range, noise_ratio
                    )
                    outputs["rule_latents"][j : j + 1] = noisy_latent

            # process each task in the batch
            for j in range(batch["rule_latent_inputs"].size(0)):
                # select target based on evaluation mode
                if evaluation_mode == "test":
                    # use test target
                    target_logits = model.decoder(
                        outputs["rule_latents"][j : j + 1],
                        batch["test_inputs"][j : j + 1],
                    )
                    target_output = batch["test_outputs"][j : j + 1]
                elif evaluation_mode == "holdout" and batch["has_holdout"][j]:
                    # use holdout target
                    target_logits = model.decoder(
                        outputs["rule_latents"][j : j + 1],
                        batch["holdout_inputs"][j : j + 1],
                    )
                    target_output = batch["holdout_outputs"][j : j + 1]
                else:
                    # fallback to test target
                    target_logits = model.decoder(
                        outputs["rule_latents"][j : j + 1],
                        batch["test_inputs"][j : j + 1],
                    )
                    target_output = batch["test_outputs"][j : j + 1]

                # convert to predictions
                predictions = torch.argmax(target_logits, dim=1).squeeze(0)

                # calculate metrics
                metrics = calculate_accuracy_metrics(predictions, target_output)

                # get the global task index (actual ARC task index)
                global_task_index = global_task_idx
                if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
                    # we're using a subset, get the actual task index from the original dataset
                    global_task_index = dataset.indices[global_task_idx]

                # extract task id - try multiple approaches
                task_id = "unknown"

                # approach 1: from combination_info in batch
                if "combination_info" in batch:
                    combo_info = batch["combination_info"]
                    if isinstance(combo_info, dict):
                        task_id = combo_info.get("task_id", "unknown")
                    elif isinstance(combo_info, list) and len(combo_info) > j:
                        task_id = combo_info[j].get("task_id", "unknown")
                    elif isinstance(combo_info, list) and len(combo_info) > 0:
                        task_id = combo_info[0].get("task_id", "unknown")

                # approach 2: if we have a subset, get task id from original dataset
                if task_id == "unknown" and hasattr(dataset, "dataset"):
                    # we're using a subset, get the original dataset
                    original_dataset = dataset.dataset
                    if global_task_index < len(original_dataset.tasks):
                        task_id = original_dataset.tasks[global_task_index]["task_id"]

                # approach 3: if still unknown, use the global task index as task_id
                if task_id == "unknown":
                    task_id = f"task_{global_task_index}"

                # store results
                results.append(
                    {
                        "task_idx": global_task_idx,  # Sequential evaluation index
                        "global_task_index": global_task_index,  # Actual ARC task index
                        "task_id": task_id,  # Task ID (filename)
                        "perfect_accuracy": metrics["perfect_accuracy"],
                        "pixel_accuracy": metrics["pixel_accuracy"],
                        "near_miss_accuracy": metrics["near_miss_accuracy"],
                        "batch": batch,
                        "predictions": predictions,
                        "logits": target_logits,
                        "evaluation_mode": evaluation_mode,
                        "combination_info": batch.get("combination_info", {}),
                        "batch_sample_idx": j,  # Store position within batch for visualization
                    }
                )
                global_task_idx += 1

    return results


def test_all_combinations(
    model,
    dataset,
    config,
    evaluation_mode="test",
    progress_bar=None,
    inject_noise=False,
    noise_type="gaussian",
    noise_std=1.0,
    noise_range=1.0,
    noise_ratio=1.0,
    enable_color_augmentation=False,
    augmentation_variants=1,
    preserve_background=True,
    augmentation_seed=42,
):
    """test all possible combinations of train examples for rule latent creation.

    Args:
        model: The model to evaluate
        dataset: Dataset to test on
        config: Configuration object
        evaluation_mode: "test" for test targets, "holdout" for holdout targets
        progress_bar: Streamlit progress bar
        inject_noise: Whether to inject noise into rule latent
        noise_type: Type of noise ("gaussian", "uniform", "zeros", "ones")
        noise_std: Standard deviation for gaussian noise
        noise_range: Range for uniform noise
        noise_ratio: Fraction of rule latent to replace with noise
        enable_color_augmentation: Whether to enable color relabeling
        augmentation_variants: Number of augmented versions per example
        preserve_background: Whether to preserve background color
        augmentation_seed: Random seed for augmentation
    """
    # Set deterministic training for reproducible results
    config.set_deterministic_training()

    results = []

    # Create augmented dataset if color augmentation is enabled
    if enable_color_augmentation:
        # Create a copy of the config with augmentation enabled
        augmented_config = Config()
        augmented_config.__dict__.update(config.__dict__)  # Copy all config values
        augmented_config.use_color_relabeling = True
        augmented_config.augmentation_variants = augmentation_variants
        augmented_config.preserve_background = preserve_background
        augmented_config.random_seed = augmentation_seed

        # Create augmented dataset
        from algo.data import ARCDataset

        augmented_dataset = ARCDataset(
            config.arc_agi1_dir,
            augmented_config,
            holdout=True,
            use_first_combination_only=True,
        )

        # If we're using a subset, create the same subset for augmented dataset
        if hasattr(dataset, "dataset"):
            # Get the original indices
            original_indices = dataset.indices
            augmented_dataset = Subset(augmented_dataset, original_indices)

        dataset = augmented_dataset

    # Get the underlying dataset if it's a Subset
    if hasattr(dataset, "dataset"):
        underlying_dataset = dataset.dataset
    else:
        underlying_dataset = dataset

    with torch.no_grad():
        for task_idx in range(len(dataset)):
            if progress_bar:
                progress_bar.progress((task_idx + 1) / len(dataset))

            # get the task data
            task_data = dataset[task_idx]

            # get the original task index for combinations
            original_task_index = task_data["combination_info"]["task_idx"]

            # get the global task index for display (same logic as regular evaluation)
            global_task_index = task_idx
            if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
                # we're using a subset, get the actual task index from the original dataset
                global_task_index = dataset.indices[task_idx]

            # get all possible combinations for this task (use original index)
            task_combinations = underlying_dataset.combinations[original_task_index]

            task_results = []
            for combo_idx, (i, j) in enumerate(task_combinations):
                # create rule latent from this combination
                # Get the raw data from the underlying dataset
                raw_task_idx = task_data["combination_info"]["task_idx"]
                raw_task = underlying_dataset.tasks[raw_task_idx]

                # Get the specific training examples for this combination
                # Handle both original and augmented examples
                all_examples = raw_task["train"]
                if "augmented_train" in raw_task:
                    all_examples = raw_task["train"] + raw_task["augmented_train"]

                example1 = all_examples[i]
                example2 = all_examples[j]

                # Preprocess for ResNet encoder
                from algo.data import preprocess_example_image

                example1_input = preprocess_example_image(
                    example1["input"], config
                ).unsqueeze(0)  # [1, 3, 64, 64]
                example1_output = preprocess_example_image(
                    example1["output"], config
                ).unsqueeze(0)  # [1, 3, 64, 64]
                example2_input = preprocess_example_image(
                    example2["input"], config
                ).unsqueeze(0)  # [1, 3, 64, 64]
                example2_output = preprocess_example_image(
                    example2["output"], config
                ).unsqueeze(0)  # [1, 3, 64, 64]

                rule_latent = model.encoder(
                    example1_input, example1_output, example2_input, example2_output
                )

                # inject noise into rule latent if requested
                if inject_noise:
                    rule_latent = generate_noise_latent(
                        rule_latent, noise_type, noise_std, noise_range, noise_ratio
                    )

                # evaluate on target
                if evaluation_mode == "test":
                    target = task_data["training_targets"][-1]  # test target
                elif evaluation_mode == "holdout" and task_data.get("holdout_target"):
                    target = task_data["holdout_target"]
                else:
                    target = task_data["training_targets"][-1]

                logits = model.decoder(rule_latent, target["input"])
                predictions = torch.argmax(logits, dim=1).squeeze(0)
                metrics = calculate_accuracy_metrics(predictions, target["output"])

                # Create sample data for visualization - use decoder preprocessing for training examples
                from algo.data import preprocess_target_image

                sample_data = {
                    "train_examples": [
                        {
                            "input": preprocess_target_image(
                                example1["input"], config
                            ).squeeze(0),  # [1, 30, 30] -> [30, 30]
                            "output": preprocess_target_image(
                                example1["output"], config
                            ).squeeze(0),
                        },
                        {
                            "input": preprocess_target_image(
                                example2["input"], config
                            ).squeeze(0),
                            "output": preprocess_target_image(
                                example2["output"], config
                            ).squeeze(0),
                        },
                    ],
                    "test_example": {
                        "input": target["input"],
                        "output": target["output"],
                    },
                }

                task_results.append(
                    {
                        "combination_idx": combo_idx,
                        "pair_indices": (i, j),
                        "perfect_accuracy": metrics["perfect_accuracy"],
                        "pixel_accuracy": metrics["pixel_accuracy"],
                        "near_miss_accuracy": metrics["near_miss_accuracy"],
                        "predictions": predictions,
                        "logits": logits,
                        "sample_data": sample_data,  # Add visualization data
                    }
                )

            results.append(
                {
                    "task_idx": task_idx,  # Sequential evaluation index
                    "global_task_index": global_task_index,  # Actual ARC task index
                    "task_id": task_data["combination_info"]["task_id"],
                    "combinations": task_results,
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
        config = Config()
        model = SimpleARCModel(config)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        st.sidebar.success("✅ model loaded successfully")
    except Exception as e:
        st.error(f"❌ failed to load model: {e}")
        st.stop()

    config.use_color_relabeling = False
    dataset = ARCDataset(
        config.arc_agi1_dir, config, holdout=True, use_first_combination_only=True
    )
    if task_set == "overfit tasks only":
        if "tasks" in exp_info and "task_indices" in exp_info["tasks"]:
            task_indices = exp_info["tasks"]["task_indices"]
            dataset = Subset(dataset, task_indices)
            st.sidebar.write(f"**evaluating on {len(task_indices)} overfit tasks**")
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

    if st.sidebar.button("🚀 evaluate model", type="primary"):
        if "evaluation_results" in st.session_state:
            del st.session_state.evaluation_results
        if "combination_results" in st.session_state:
            del st.session_state.combination_results

        progress_bar = st.progress(0)
        status_text = st.empty()

        if test_combinations:
            status_text.text("testing all combinations...")
            results = test_all_combinations(
                model,
                dataset,
                config,
                evaluation_mode,
                progress_bar,
                inject_noise,
                noise_type,
                noise_std,
                noise_range,
                noise_ratio,
                enable_color_augmentation,
                augmentation_variants,
                preserve_background,
                augmentation_seed,
            )
            st.session_state.combination_results = results
        else:
            status_text.text("evaluating model on tasks...")
            results = evaluate_model_on_tasks(
                model,
                dataset,
                config,
                evaluation_mode,
                progress_bar,
                inject_noise,
                noise_type,
                noise_std,
                noise_range,
                noise_ratio,
                enable_color_augmentation,
                augmentation_variants,
                preserve_background,
                augmentation_seed,
            )
            st.session_state.evaluation_results = results

        st.session_state.task_set = task_set
        st.session_state.evaluation_mode = evaluation_mode
        st.session_state.test_combinations = test_combinations
        st.session_state.inject_noise = inject_noise
        st.session_state.noise_type = noise_type
        st.session_state.noise_std = noise_std
        st.session_state.noise_range = noise_range
        st.session_state.noise_ratio = noise_ratio
        st.session_state.enable_color_augmentation = enable_color_augmentation
        st.session_state.augmentation_variants = augmentation_variants
        st.session_state.preserve_background = preserve_background
        st.session_state.augmentation_seed = augmentation_seed

        progress_bar.empty()
        status_text.text("evaluation complete!")
        st.rerun()

    if "evaluation_results" in st.session_state:
        results = st.session_state.evaluation_results
        current_task_set = st.session_state.get("task_set", "unknown")

        # show noise info if applicable
        noise_info = ""
        if st.session_state.get("inject_noise", False):
            noise_type = st.session_state.get("noise_type", "gaussian")
            noise_ratio = st.session_state.get("noise_ratio", 1.0)
            if noise_type == "gaussian":
                noise_std = st.session_state.get("noise_std", 1.0)
                noise_info = f" (noise: {noise_type}, std={noise_std:.1f}, ratio={noise_ratio:.1f})"
            elif noise_type == "uniform":
                noise_range = st.session_state.get("noise_range", 1.0)
                noise_info = f" (noise: {noise_type}, range={noise_range:.1f}, ratio={noise_ratio:.1f})"
            else:
                noise_info = f" (noise: {noise_type}, ratio={noise_ratio:.1f})"

        st.subheader(f"📊 evaluation results ({current_task_set}){noise_info}")

        # create results dataframe
        df_data = []
        for result in results:
            df_data.append(
                {
                    "idx": result["global_task_index"],
                    "task_id": result["task_id"],
                    "perfect": f"{result['perfect_accuracy']:.3f}",
                    "pixel": f"{result['pixel_accuracy']:.3f}",
                    "near_miss": f"{result['near_miss_accuracy']:.3f}",
                    "status": "✅ perfect"
                    if result["perfect_accuracy"] > 0.99
                    else "⚠️ partial"
                    if result["pixel_accuracy"] > 0.5
                    else "❌ failed",
                }
            )

        df = pd.DataFrame(df_data)

        # display summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("total tasks", len(results))
        with col2:
            perfect_count = sum(1 for r in results if r["perfect_accuracy"] > 0.99)
            st.metric("perfect tasks", f"{perfect_count}/{len(results)}")
        with col3:
            avg_pixel = np.mean([r["pixel_accuracy"] for r in results])
            st.metric("avg pixel accuracy", f"{avg_pixel:.3f}")
        with col4:
            avg_near_miss = np.mean([r["near_miss_accuracy"] for r in results])
            st.metric("avg near-miss", f"{avg_near_miss:.3f}")

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
            # Extract the correct sample from the batch
            batch = selected_task["batch"]
            sample_idx = selected_task[
                "batch_sample_idx"
            ]  # Use stored batch sample index

            evaluation_mode = st.session_state.get("evaluation_mode", "test")
            sample_data = extract_sample_from_batch(batch, sample_idx, evaluation_mode)
            fig = visualize_prediction_comparison(
                sample_data, selected_task["predictions"]
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
    elif "combination_results" in st.session_state:
        results = st.session_state.combination_results
        current_task_set = st.session_state.get("task_set", "unknown")
        evaluation_mode = st.session_state.get("evaluation_mode", "test")

        # show noise info if applicable
        noise_info = ""
        if st.session_state.get("inject_noise", False):
            noise_type = st.session_state.get("noise_type", "gaussian")
            noise_ratio = st.session_state.get("noise_ratio", 1.0)
            if noise_type == "gaussian":
                noise_std = st.session_state.get("noise_std", 1.0)
                noise_info = f" (noise: {noise_type}, std={noise_std:.1f}, ratio={noise_ratio:.1f})"
            elif noise_type == "uniform":
                noise_range = st.session_state.get("noise_range", 1.0)
                noise_info = f" (noise: {noise_type}, range={noise_range:.1f}, ratio={noise_ratio:.1f})"
            else:
                noise_info = f" (noise: {noise_type}, ratio={noise_ratio:.1f})"

        st.subheader(
            f"🔄 combination test results ({current_task_set}) - {evaluation_mode} mode{noise_info}"
        )

        # create combination results dataframe
        combo_df_data = []
        for task_result in results:
            global_task_index = task_result[
                "global_task_index"
            ]  # Use global task index
            task_id = task_result["task_id"]  # Use actual task ID
            for combo in task_result["combinations"]:
                combo_df_data.append(
                    {
                        "idx": global_task_index,  # Global task index
                        "task_id": task_id,  # Task ID (filename)
                        "combination": f"({combo['pair_indices'][0]}, {combo['pair_indices'][1]})",
                        "perfect": f"{combo['perfect_accuracy']:.3f}",
                        "pixel": f"{combo['pixel_accuracy']:.3f}",
                        "near_miss": f"{combo['near_miss_accuracy']:.3f}",
                        "status": "✅ perfect"
                        if combo["perfect_accuracy"] > 0.99
                        else "⚠️ partial"
                        if combo["pixel_accuracy"] > 0.5
                        else "❌ failed",
                    }
                )

        combo_df = pd.DataFrame(combo_df_data)

        # display summary stats
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("total combinations", len(combo_df_data))
        with col2:
            perfect_count = sum(1 for r in combo_df_data if float(r["perfect"]) > 0.99)
            st.metric("perfect combinations", f"{perfect_count}/{len(combo_df_data)}")
        with col3:
            # Calculate perfect tasks (tasks with at least one perfect combination)
            perfect_tasks = set()
            for task_result in results:
                has_perfect = any(
                    combo["perfect_accuracy"] > 0.99
                    for combo in task_result["combinations"]
                )
                if has_perfect:
                    perfect_tasks.add(task_result["task_id"])
            st.metric("perfect tasks", f"{len(perfect_tasks)}/{len(results)}")
        with col4:
            avg_pixel = np.mean([float(r["pixel"]) for r in combo_df_data])
            st.metric("avg pixel accuracy", f"{avg_pixel:.3f}")
        with col5:
            avg_near_miss = np.mean([float(r["near_miss"]) for r in combo_df_data])
            st.metric("avg near-miss", f"{avg_near_miss:.3f}")

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

            # find the selected combination
            combo_idx = 0
            selected_task = None
            selected_combo = None

            for task_result in results:
                for combo in task_result["combinations"]:
                    if combo_idx == selected_idx:
                        selected_task = task_result
                        selected_combo = combo
                        break
                    combo_idx += 1
                if selected_combo is not None:
                    break

            if selected_combo is not None:
                st.subheader(
                    f"🔍 visualizing task {selected_task['task_id']} combination {selected_combo['combination_idx']}"
                )

                # visualize the selected combination
                fig = visualize_prediction_comparison(
                    selected_combo["sample_data"], selected_combo["predictions"]
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
                    st.write(f"**Combination:** {selected_combo['pair_indices']}")
                    st.write(f"**Evaluation Mode:** {evaluation_mode}")
                with col2:
                    st.write(
                        f"**Training Examples Used:** {selected_combo['pair_indices'][0]} and {selected_combo['pair_indices'][1]}"
                    )
                    st.write(
                        f"**Status:** {'✅ perfect' if selected_combo['perfect_accuracy'] > 0.99 else '⚠️ partial' if selected_combo['pixel_accuracy'] > 0.5 else '❌ failed'}"
                    )

        # per-task summary
        st.subheader("📊 per-task summary")
        task_summary = []
        for task_result in results:
            global_task_index = task_result["global_task_index"]
            task_id = task_result["task_id"]
            combinations = task_result["combinations"]
            best_perfect = max(combo["perfect_accuracy"] for combo in combinations)
            best_pixel = max(combo["pixel_accuracy"] for combo in combinations)
            avg_perfect = np.mean([combo["perfect_accuracy"] for combo in combinations])
            avg_pixel = np.mean([combo["pixel_accuracy"] for combo in combinations])

            task_summary.append(
                {
                    "idx": global_task_index,
                    "task_id": task_id,
                    "combinations": len(combinations),
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
