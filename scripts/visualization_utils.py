#!/usr/bin/env python3
"""
Shared visualization and type conversion utilities for ARC-AGI dataset viewers.

Common functions used by both view_dataset.py and view_model_predictions.py.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ARC color palette (official 10 colors)
ARC_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Grey
    "#F012BE",  # 6: Fuschia
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Teal
    "#870C25",  # 9: Brown
]


def tensor_to_numpy(tensor):
    """Convert PyTorch tensor to numpy array."""
    # Handle numpy arrays (already converted)
    if isinstance(tensor, np.ndarray):
        return tensor

    # Handle pytorch tensors
    if tensor.dim() == 4:  # [B, C, H, W]
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    elif tensor.dim() == 3:  # [C, H, W]
        return tensor.permute(1, 2, 0).cpu().numpy()
    elif tensor.dim() == 2:  # [H, W]
        return tensor.cpu().numpy()
    else:
        return tensor.cpu().numpy()


def tensor_to_grayscale_numpy(tensor):
    """Convert PyTorch tensor to grayscale numpy array."""
    # Handle numpy arrays (already converted)
    if isinstance(tensor, np.ndarray):
        return tensor

    # Handle pytorch tensors
    if tensor.dim() == 4:  # [B, C, H, W]
        return tensor.squeeze(0).squeeze(0).cpu().numpy()
    elif tensor.dim() == 3:  # [C, H, W]
        return tensor.squeeze(0).cpu().numpy()
    elif tensor.dim() == 2:  # [H, W]
        return tensor.cpu().numpy()
    else:
        return tensor.cpu().numpy()


def denormalize_rgb(img_tensor):
    """Convert normalized RGB tensor back to [0, 1] range."""
    # Convert from [-1, 1] to [0, 1]
    img = (img_tensor + 1) / 2
    return torch.clamp(img, 0, 1)


def apply_arc_color_palette(img_np):
    """Apply ARC color palette to grayscale image."""
    rgb_img = np.zeros((*img_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = img_np == i
        rgb_img[mask] = (
            np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
            / 255.0
        )
    return rgb_img


def visualize_arc_image(img_tensor, title, is_rgb=True, figsize=(4, 4)):
    """Visualize an ARC image with proper color mapping."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if is_rgb:
        # RGB image (example images)
        img = denormalize_rgb(img_tensor)
        img_np = tensor_to_numpy(img)
        ax.imshow(img_np)
    else:
        # Grayscale image (target images)
        img_np = tensor_to_grayscale_numpy(img_tensor)
        rgb_img = apply_arc_color_palette(img_np)
        ax.imshow(rgb_img)

    ax.set_title(title, fontsize=10)
    ax.axis("off")

    # Add grid
    ax.set_xticks(np.arange(-0.5, img_np.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, img_np.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    return fig


def show_color_palette():
    """Display the ARC color palette."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    for i, color in enumerate(ARC_COLORS):
        ax.add_patch(
            patches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="black")
        )
        ax.text(
            i + 0.5,
            0.5,
            str(i),
            ha="center",
            va="center",
            fontsize=12,
            color="white" if i in [0, 9] else "black",
            weight="bold",
        )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_title("ARC Color Palette (0-9)", fontsize=14)
    ax.set_xticks(range(11))
    ax.set_yticks([])
    ax.set_xlabel("Color Index")

    return fig


def visualize_task_combination(
    task_data, task_idx=0, combination_idx=0, show_holdout=False
):
    """Visualize a single task combination."""
    # determine grid size based on whether we have holdout data
    is_counterfactual = task_data.get("combination_info", {}).get(
        "is_counterfactual", False
    )
    # create figure - always use 2x4 grid for consistency
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"arc task {task_idx} - combination {combination_idx}{' (counterfactual)' if is_counterfactual else ''}",
        fontsize=16,
    )

    # get rule latent inputs and test example
    rule_latent_examples = task_data["rule_latent_examples"]
    test_example = task_data["test_example"]

    # 1. ex1 input
    axes[0, 0].set_title("ex1 input", fontsize=12)
    img1 = denormalize_rgb(rule_latent_examples[0]["input"])
    img1_np = tensor_to_numpy(img1)
    axes[0, 0].imshow(img1_np)
    axes[0, 0].axis("off")

    # 2. ex1 output (to be rotated with cf)
    axes[0, 1].set_title("ex1 output", fontsize=12)
    img2 = denormalize_rgb(rule_latent_examples[0]["output"])
    img2_np = tensor_to_numpy(img2)
    axes[0, 1].imshow(img2_np)
    axes[0, 1].axis("off")

    # 3. ex2 input
    axes[0, 2].set_title("ex2 input", fontsize=12)
    img3 = denormalize_rgb(rule_latent_examples[1]["input"])
    img3_np = tensor_to_numpy(img3)
    axes[0, 2].imshow(img3_np)
    axes[0, 2].axis("off")

    # 4. holdout input (if available)
    if show_holdout and task_data.get("holdout_example") is not None:
        axes[0, 3].set_title("holdout input", fontsize=12)
        holdout_input_np = tensor_to_grayscale_numpy(
            task_data["holdout_example"]["input"]
        )
        rgb_holdout_input = apply_arc_color_palette(holdout_input_np)
        axes[0, 3].imshow(rgb_holdout_input)
        axes[0, 3].axis("off")
    else:
        # Hide holdout input if not available
        axes[0, 3].set_title("holdout input (n/a)", fontsize=12)
        axes[0, 3].text(
            0.5,
            0.5,
            "No holdout data",
            ha="center",
            va="center",
            transform=axes[0, 3].transAxes,
        )
        axes[0, 3].axis("off")

    # 5. ex2 output (to be rotated with cf) - show training target with counterfactual
    axes[1, 0].set_title(
        "ex2 output (cf)" if is_counterfactual else "ex2 output", fontsize=12
    )
    img4 = denormalize_rgb(rule_latent_examples[1]["output"])
    img4_np = tensor_to_numpy(img4)
    axes[1, 0].imshow(img4_np)
    axes[1, 0].axis("off")

    # 6. test input
    axes[1, 1].set_title("test input", fontsize=12)
    test_input_np = tensor_to_grayscale_numpy(test_example["input"])
    rgb_test_input = apply_arc_color_palette(test_input_np)
    axes[1, 1].imshow(rgb_test_input)
    axes[1, 1].axis("off")

    # 7. test output (to be rotated with cf)
    axes[1, 2].set_title(
        "test output (cf)" if is_counterfactual else "test output", fontsize=12
    )
    test_output_np = tensor_to_grayscale_numpy(test_example["output"])
    rgb_test_output = apply_arc_color_palette(test_output_np)
    axes[1, 2].imshow(rgb_test_output)
    axes[1, 2].axis("off")

    # 8. holdout output (if available)
    if show_holdout and task_data.get("holdout_example") is not None:
        axes[1, 3].set_title(
            "holdout output (cf)" if is_counterfactual else "holdout output",
            fontsize=12,
        )
        holdout_output_np = tensor_to_grayscale_numpy(
            task_data["holdout_example"]["output"]
        )
        rgb_holdout_output = apply_arc_color_palette(holdout_output_np)
        axes[1, 3].imshow(rgb_holdout_output)
        axes[1, 3].axis("off")
    else:
        # Hide holdout output if not available
        axes[1, 3].set_title("holdout output (n/a)", fontsize=12)
        axes[1, 3].text(
            0.5,
            0.5,
            "No holdout data",
            ha="center",
            va="center",
            transform=axes[1, 3].transAxes,
        )
        axes[1, 3].axis("off")

    # add grid to all subplots
    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 30, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_prediction_comparison(sample, prediction, evaluation_mode="test"):
    """Visualize model predictions compared to ground truth."""
    # create figure with 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"model prediction comparison ({evaluation_mode} mode)", fontsize=16)

    # example 1 - handle RGB images from rule latent inputs
    axes[0, 0].set_title("example 1 input", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 1:
        img1_np = tensor_to_numpy(sample["train_examples"][0]["input"])
        if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
            # RGB image - display directly
            axes[0, 0].imshow(img1_np)
        else:
            # Grayscale image - apply ARC color palette
            img1_np = tensor_to_grayscale_numpy(sample["train_examples"][0]["input"])
            rgb_img1 = apply_arc_color_palette(img1_np)
            axes[0, 0].imshow(rgb_img1)
    else:
        axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 0].axis("off")

    axes[0, 1].set_title("example 1 output", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 1:
        img2_np = tensor_to_numpy(sample["train_examples"][0]["output"])
        if len(img2_np.shape) == 3 and img2_np.shape[2] == 3:
            # RGB image - display directly
            axes[0, 1].imshow(img2_np)
        else:
            # Grayscale image - apply ARC color palette
            img2_np = tensor_to_grayscale_numpy(sample["train_examples"][0]["output"])
            rgb_img2 = apply_arc_color_palette(img2_np)
            axes[0, 1].imshow(rgb_img2)
    else:
        axes[0, 1].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 1].axis("off")

    # example 2
    axes[0, 2].set_title("example 2 input", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 2:
        img3_np = tensor_to_numpy(sample["train_examples"][1]["input"])
        if len(img3_np.shape) == 3 and img3_np.shape[2] == 3:
            # RGB image - display directly
            axes[0, 2].imshow(img3_np)
        else:
            # Grayscale image - apply ARC color palette
            img3_np = tensor_to_grayscale_numpy(sample["train_examples"][1]["input"])
            rgb_img3 = apply_arc_color_palette(img3_np)
            axes[0, 2].imshow(rgb_img3)
    else:
        axes[0, 2].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 2].axis("off")

    axes[0, 3].set_title("example 2 output", fontsize=12)
    if "train_examples" in sample and len(sample["train_examples"]) >= 2:
        img4_np = tensor_to_numpy(sample["train_examples"][1]["output"])
        if len(img4_np.shape) == 3 and img4_np.shape[2] == 3:
            # RGB image - display directly
            axes[0, 3].imshow(img4_np)
        else:
            # Grayscale image - apply ARC color palette
            img4_np = tensor_to_grayscale_numpy(sample["train_examples"][1]["output"])
            rgb_img4 = apply_arc_color_palette(img4_np)
            axes[0, 3].imshow(rgb_img4)
    else:
        axes[0, 3].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[0, 3].axis("off")

    # target input - use appropriate data based on evaluation mode
    target_data = None
    if evaluation_mode == "holdout" and "holdout_example" in sample:
        target_data = sample["holdout_example"]
        axes[1, 0].set_title("holdout input", fontsize=12)
    else:
        target_data = sample.get("test_example")
        axes[1, 0].set_title("test input", fontsize=12)

    if target_data and "input" in target_data:
        target_input_np = tensor_to_grayscale_numpy(target_data["input"])
        rgb_target_input = apply_arc_color_palette(target_input_np)
        axes[1, 0].imshow(rgb_target_input)
    else:
        axes[1, 0].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[1, 0].axis("off")

    # ground truth output
    if evaluation_mode == "holdout" and "holdout_example" in sample:
        axes[1, 1].set_title("holdout ground truth", fontsize=12)
    else:
        axes[1, 1].set_title("test ground truth", fontsize=12)

    if target_data and "output" in target_data:
        target_output_np = tensor_to_grayscale_numpy(target_data["output"])
        rgb_target_output = apply_arc_color_palette(target_output_np)
        axes[1, 1].imshow(rgb_target_output)
    else:
        axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[1, 1].axis("off")

    # model prediction
    axes[1, 2].set_title("model prediction", fontsize=12)
    pred_np = tensor_to_grayscale_numpy(prediction)
    rgb_pred = apply_arc_color_palette(pred_np)
    axes[1, 2].imshow(rgb_pred)
    axes[1, 2].axis("off")

    # difference visualization
    axes[1, 3].set_title("difference", fontsize=12)
    if target_data and "output" in target_data:
        target_output_np = tensor_to_grayscale_numpy(target_data["output"])
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
