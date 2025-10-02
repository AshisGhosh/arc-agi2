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

    # Check if this is cycling format (has target_example)
    is_cycling = "target_example" in task_data

    # create figure - use 3x4 grid for more compact layout
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Update title to show cycling format
    title = f"arc task {task_idx} - combination {combination_idx}"
    if is_cycling:
        cycling_indices = task_data.get("cycling_indices", (0, 1, 2))
        title += f" (cycling: {cycling_indices[0]},{cycling_indices[1]} -> {cycling_indices[2]})"
    if is_counterfactual:
        title += " (counterfactual)"

    fig.suptitle(title, fontsize=16)

    # get support examples and test examples
    support_examples = task_data["support_examples"]
    test_examples = task_data["test_examples"]
    test_example = test_examples[0]  # Use first test example for visualization

    # Check if we have RGB support examples (ResNet) or just grayscale (Patch)
    has_rgb_support = (
        "support_examples_rgb" in task_data
        and task_data["support_examples_rgb"] is not None
    )

    if has_rgb_support:
        # Use RGB support examples for visualization (ResNet dataset)
        rgb_support_examples = task_data["support_examples_rgb"]

        # Row 1: Example 1 input/output pair (RGB)
        # 1. ex1 input
        axes[0, 0].set_title("ex1 input (RGB)", fontsize=10)
        img1 = denormalize_rgb(rgb_support_examples[0]["input"])
        img1_np = tensor_to_numpy(img1)
        axes[0, 0].imshow(img1_np)
        axes[0, 0].axis("off")

        # 2. ex1 output
        axes[0, 1].set_title("ex1 output (RGB)", fontsize=10)
        img2 = denormalize_rgb(rgb_support_examples[0]["output"])
        img2_np = tensor_to_numpy(img2)
        axes[0, 1].imshow(img2_np)
        axes[0, 1].axis("off")

        # 3. ex2 input
        axes[0, 2].set_title("ex2 input (RGB)", fontsize=10)
        img3 = denormalize_rgb(rgb_support_examples[1]["input"])
        img3_np = tensor_to_numpy(img3)
        axes[0, 2].imshow(img3_np)
        axes[0, 2].axis("off")

        # 4. ex2 output
        axes[0, 3].set_title(
            "ex2 output (RGB, cf)" if is_counterfactual else "ex2 output (RGB)",
            fontsize=10,
        )
        img4 = denormalize_rgb(rgb_support_examples[1]["output"])
        img4_np = tensor_to_numpy(img4)
        axes[0, 3].imshow(img4_np)
        axes[0, 3].axis("off")
    else:
        # Use grayscale support examples for visualization (Patch dataset)
        # Row 1: Example 1 input/output pair (Grayscale)
        # 1. ex1 input
        axes[0, 0].set_title("ex1 input (Gray)", fontsize=10)
        img1 = support_examples[0]["input"]
        img1_np = tensor_to_grayscale_numpy(img1)
        img1_rgb = apply_arc_color_palette(img1_np)
        axes[0, 0].imshow(img1_rgb)
        axes[0, 0].axis("off")

        # 2. ex1 output
        axes[0, 1].set_title("ex1 output (Gray)", fontsize=10)
        img2 = support_examples[0]["output"]
        img2_np = tensor_to_grayscale_numpy(img2)
        img2_rgb = apply_arc_color_palette(img2_np)
        axes[0, 1].imshow(img2_rgb)
        axes[0, 1].axis("off")

        # 3. ex2 input
        axes[0, 2].set_title("ex2 input (Gray)", fontsize=10)
        img3 = support_examples[1]["input"]
        img3_np = tensor_to_grayscale_numpy(img3)
        img3_rgb = apply_arc_color_palette(img3_np)
        axes[0, 2].imshow(img3_rgb)
        axes[0, 2].axis("off")

        # 4. ex2 output
        axes[0, 3].set_title(
            "ex2 output (Gray, cf)" if is_counterfactual else "ex2 output (Gray)",
            fontsize=10,
        )
        img4 = support_examples[1]["output"]
        img4_np = tensor_to_grayscale_numpy(img4)
        img4_rgb = apply_arc_color_palette(img4_np)
        axes[0, 3].imshow(img4_rgb)
        axes[0, 3].axis("off")

    # Row 2: Target example (cycling) or Test example (fallback)
    if is_cycling:
        # Show target example for cycling format
        target_example = task_data["target_example"]

        # 5. target input
        axes[1, 0].set_title("target input (cycling)", fontsize=10)
        target_input_np = tensor_to_grayscale_numpy(target_example["input"])
        rgb_target_input = apply_arc_color_palette(target_input_np)
        axes[1, 0].imshow(rgb_target_input)
        axes[1, 0].axis("off")

        # 6. target output
        axes[1, 1].set_title(
            "target output (cycling, cf)"
            if is_counterfactual
            else "target output (cycling)",
            fontsize=10,
        )
        target_output_np = tensor_to_grayscale_numpy(target_example["output"])
        rgb_target_output = apply_arc_color_palette(target_output_np)
        axes[1, 1].imshow(rgb_target_output)
        axes[1, 1].axis("off")
    else:
        # This should not happen in cycling format
        # 5. test input
        axes[1, 0].set_title("test input", fontsize=10)
        test_input_np = tensor_to_grayscale_numpy(test_example["input"])
        rgb_test_input = apply_arc_color_palette(test_input_np)
        axes[1, 0].imshow(rgb_test_input)
        axes[1, 0].axis("off")

        # 6. test output (to be rotated with cf)
        axes[1, 1].set_title(
            "test output (cf)" if is_counterfactual else "test output", fontsize=10
        )
        test_output_np = tensor_to_grayscale_numpy(test_example["output"])
        rgb_test_output = apply_arc_color_palette(test_output_np)
        axes[1, 1].imshow(rgb_test_output)
        axes[1, 1].axis("off")

    # 7. Additional test examples (if any)
    if len(test_examples) > 1:
        axes[1, 2].set_title("test input 2", fontsize=10)
        test2_input_np = tensor_to_grayscale_numpy(test_examples[1]["input"])
        rgb_test2_input = apply_arc_color_palette(test2_input_np)
        axes[1, 2].imshow(rgb_test2_input)
        axes[1, 2].axis("off")

        axes[1, 3].set_title("test output 2", fontsize=10)
        test2_output_np = tensor_to_grayscale_numpy(test_examples[1]["output"])
        rgb_test2_output = apply_arc_color_palette(test2_output_np)
        axes[1, 3].imshow(rgb_test2_output)
        axes[1, 3].axis("off")
    else:
        # Show explanation for cycling format or hide additional test examples
        if is_cycling:
            axes[1, 2].set_title("cycling format", fontsize=10)
            axes[1, 2].text(
                0.5,
                0.5,
                "Model uses\nsupport examples\nto predict target",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
                fontsize=9,
            )
        else:
            axes[1, 2].set_title("test input 2 (n/a)", fontsize=10)
            axes[1, 2].text(
                0.5,
                0.5,
                "Only 1 test",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
        axes[1, 2].axis("off")

        if is_cycling:
            axes[1, 3].set_title("test examples", fontsize=10)
            axes[1, 3].text(
                0.5,
                0.5,
                "Test examples\nare separate\nfor evaluation",
                ha="center",
                va="center",
                transform=axes[1, 3].transAxes,
                fontsize=9,
            )
        else:
            axes[1, 3].set_title("test output 2 (n/a)", fontsize=10)
        axes[1, 3].text(
            0.5,
            0.5,
            "Only 1 test",
            ha="center",
            va="center",
            transform=axes[1, 3].transAxes,
        )
        axes[1, 3].axis("off")

    # Row 3: Holdout input/output pair (if available)
    if show_holdout and task_data.get("holdout_example") is not None:
        axes[2, 0].set_title("holdout input", fontsize=10)
        holdout_input_np = tensor_to_grayscale_numpy(
            task_data["holdout_example"]["input"]
        )
        rgb_holdout_input = apply_arc_color_palette(holdout_input_np)
        axes[2, 0].imshow(rgb_holdout_input)
        axes[2, 0].axis("off")

        axes[2, 1].set_title(
            "holdout output (cf)" if is_counterfactual else "holdout output",
            fontsize=10,
        )
        holdout_output_np = tensor_to_grayscale_numpy(
            task_data["holdout_example"]["output"]
        )
        rgb_holdout_output = apply_arc_color_palette(holdout_output_np)
        axes[2, 1].imshow(rgb_holdout_output)
        axes[2, 1].axis("off")

        # Hide unused cells
        axes[2, 2].axis("off")
        axes[2, 3].axis("off")
    else:
        # Hide holdout if not available
        axes[2, 0].set_title("holdout input (n/a)", fontsize=10)
        axes[2, 0].text(
            0.5,
            0.5,
            "No holdout data",
            ha="center",
            va="center",
            transform=axes[2, 0].transAxes,
        )
        axes[2, 0].axis("off")

        axes[2, 1].set_title("holdout output (n/a)", fontsize=10)
        axes[2, 1].text(
            0.5,
            0.5,
            "No holdout data",
            ha="center",
            va="center",
            transform=axes[2, 1].transAxes,
        )
        axes[2, 1].axis("off")

        # Hide unused cells
        axes[2, 2].axis("off")
        axes[2, 3].axis("off")

    # add grid to all subplots
    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 30, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_prediction_comparison(
    sample, prediction, evaluation_mode="test", test_example_idx=None
):
    """Visualize model predictions compared to ground truth."""
    # create figure with 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Add test example info to title if available
    title = f"model prediction comparison ({evaluation_mode} mode)"
    if test_example_idx is not None:
        title += f" - test example {test_example_idx}"
    elif "test_examples" in sample and len(sample["test_examples"]) > 1:
        title += f" - test example 0 (showing first of {len(sample['test_examples'])} test examples)"

    fig.suptitle(title, fontsize=16)

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
    elif "target_example" in sample:
        # Use target_example from cycling format (highest priority)
        target_data = sample["target_example"]
        axes[1, 0].set_title("target input (cycling)", fontsize=12)
    else:
        # Fallback to test_examples format
        if "test_examples" in sample and len(sample["test_examples"]) > 0:
            # Use specified test example index or first one
            if test_example_idx is not None and test_example_idx < len(
                sample["test_examples"]
            ):
                target_data = sample["test_examples"][test_example_idx]
            else:
                target_data = sample["test_examples"][0]
        else:
            # This should not happen in cycling format
            target_data = None
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
    elif "target_example" in sample:
        axes[1, 1].set_title("target ground truth (cycling)", fontsize=12)
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
