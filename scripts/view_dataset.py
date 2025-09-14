#!/usr/bin/env python3
"""
Streamlit dataset viewer for ARC-AGI dataset.

Interactive web interface to explore preprocessed ARC data batches.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from algo.config import Config
from algo.data import ARCDataset

# Set page config
st.set_page_config(page_title="ARC Dataset Viewer", page_icon="🔍", layout="wide")

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
    if tensor.dim() == 4:  # [B, C, H, W]
        return (
            tensor.squeeze(0).squeeze(0).cpu().numpy()
        )  # Remove batch and channel dims
    elif tensor.dim() == 3:  # [C, H, W]
        return tensor.squeeze(0).cpu().numpy()  # Remove channel dim
    elif tensor.dim() == 2:  # [H, W]
        return tensor.cpu().numpy()
    else:
        return tensor.cpu().numpy()


def denormalize_rgb(img_tensor):
    """Convert normalized RGB tensor back to [0, 1] range."""
    # Convert from [-1, 1] to [0, 1]
    img = (img_tensor + 1) / 2
    return torch.clamp(img, 0, 1)


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
        img_np = tensor_to_numpy(img_tensor)
        # Create RGB version using ARC color palette
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

    # Add grid
    ax.set_xticks(np.arange(-0.5, img_np.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, img_np.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    return fig


def visualize_task(task, task_idx=0, show_holdout=False):
    """Visualize a single task."""
    # Determine grid size based on whether we have holdout data
    if show_holdout and task.get("holdout_target") is not None:
        # 2x4 grid to include holdout
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"ARC Task {task_idx} (with Holdout)", fontsize=16)
    else:
        # 2x3 grid for normal view
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"ARC Task {task_idx}", fontsize=16)

    # Rule latent inputs (2 examples)
    axes[0, 0].set_title("Example 1 Input", fontsize=12)
    img1 = denormalize_rgb(task["rule_latent_inputs"][0]["input"])
    img1_np = tensor_to_numpy(img1)
    axes[0, 0].imshow(img1_np)
    axes[0, 0].axis("off")

    axes[0, 1].set_title("Example 1 Output", fontsize=12)
    img2 = denormalize_rgb(task["rule_latent_inputs"][0]["output"])
    img2_np = tensor_to_numpy(img2)
    axes[0, 1].imshow(img2_np)
    axes[0, 1].axis("off")

    # Example 2
    axes[0, 2].set_title("Example 2 Input", fontsize=12)
    img3 = denormalize_rgb(task["rule_latent_inputs"][1]["input"])
    img3_np = tensor_to_numpy(img3)
    axes[0, 2].imshow(img3_np)
    axes[0, 2].axis("off")

    axes[1, 0].set_title("Example 2 Output", fontsize=12)
    img4 = denormalize_rgb(task["rule_latent_inputs"][1]["output"])
    img4_np = tensor_to_numpy(img4)
    axes[1, 0].imshow(img4_np)
    axes[1, 0].axis("off")

    # Test target (last in training_targets)
    axes[1, 1].set_title("Test Input", fontsize=12)
    test_input_np = tensor_to_grayscale_numpy(task["training_targets"][-1]["input"])
    rgb_test_input = np.zeros((*test_input_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = test_input_np == i
        rgb_test_input[mask] = (
            np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
            / 255.0
        )
    axes[1, 1].imshow(rgb_test_input)
    axes[1, 1].axis("off")

    axes[1, 2].set_title("Test Output", fontsize=12)
    test_output_np = tensor_to_grayscale_numpy(task["training_targets"][-1]["output"])
    rgb_test_output = np.zeros((*test_output_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = test_output_np == i
        rgb_test_output[mask] = (
            np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
            / 255.0
        )
    axes[1, 2].imshow(rgb_test_output)
    axes[1, 2].axis("off")

    # Add holdout visualization if available
    if show_holdout and task.get("holdout_target") is not None:
        # Holdout input in position 4 (row 0, col 3)
        axes[0, 3].set_title("Holdout Input", fontsize=12)
        holdout_input_np = tensor_to_grayscale_numpy(task["holdout_target"]["input"])
        rgb_holdout_input = np.zeros((*holdout_input_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = holdout_input_np == i
            rgb_holdout_input[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[0, 3].imshow(rgb_holdout_input)
        axes[0, 3].axis("off")

        # Holdout output in position 8 (row 1, col 3)
        axes[1, 3].set_title("Holdout Output", fontsize=12)
        holdout_output_np = tensor_to_grayscale_numpy(task["holdout_target"]["output"])
        rgb_holdout_output = np.zeros((*holdout_output_np.shape, 3))
        for i, color in enumerate(ARC_COLORS):
            mask = holdout_output_np == i
            rgb_holdout_output[mask] = (
                np.array(
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                )
                / 255.0
            )
        axes[1, 3].imshow(rgb_holdout_output)
        axes[1, 3].axis("off")

    # Add grid to all subplots
    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 30, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
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


def main():
    """Main Streamlit app."""
    st.title("🔍 ARC Dataset Viewer")
    st.markdown("Interactive exploration of preprocessed ARC-AGI dataset")

    # Sidebar controls
    st.sidebar.header("Dataset Controls")

    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset", ["arc_agi1", "arc_agi2"], index=0
    )

    # Holdout mode
    holdout_mode = st.sidebar.checkbox(
        "Enable Holdout Mode",
        value=True,
        help="When enabled, tasks with 3+ train examples will have holdout data",
    )

    # Load dataset
    try:
        config = Config()
        config.training_dataset = dataset_choice

        with st.spinner(f"Loading {dataset_choice} dataset..."):
            if dataset_choice == "arc_agi1":
                dataset = ARCDataset(config.arc_agi1_dir, config, holdout=holdout_mode)
            else:
                dataset = ARCDataset(config.processed_dir, config)

        st.success(f"✅ Loaded {len(dataset)} samples from {dataset_choice}")

        # Task selection (after dataset is loaded)
        task_idx = st.sidebar.number_input(
            "Task Index",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1,
            help=f"Enter task index (0 to {len(dataset) - 1})",
        )

        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tasks", len(dataset))
        with col2:
            st.metric("Current Task", task_idx)
        with col3:
            if holdout_mode and dataset_choice == "arc_agi1":
                # Count tasks with holdout data
                holdout_count = 0
                for i in range(min(100, len(dataset))):  # Check first 100 samples
                    sample = dataset[i]
                    if sample.get("holdout_target") is not None:
                        holdout_count += 1
                st.metric("Tasks w/ Holdout", f"{holdout_count}/100")
            else:
                st.metric("Holdout Mode", "Disabled")

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()

    # Load a task
    try:
        task = dataset[task_idx]
        st.success(f"✅ Loaded task {task_idx}")
    except Exception as e:
        st.error(f"❌ Error loading task: {e}")
        st.stop()

    # Main content
    st.header("📊 Task Visualization")

    # Color palette
    st.subheader("🎨 ARC Color Palette")
    palette_fig = show_color_palette()
    st.pyplot(palette_fig)

    # Task visualization
    st.subheader(f"🔍 Task {task_idx}")

    # Check if this task has holdout data
    has_holdout = task.get("holdout_target") is not None

    if has_holdout:
        st.info(f"✅ Task {task_idx} has holdout data")
    else:
        st.info(f"ℹ️ Task {task_idx} has no holdout data")

    task_fig = visualize_task(task, task_idx, show_holdout=has_holdout)
    st.pyplot(task_fig)

    # Holdout vs Test comparison
    if has_holdout and holdout_mode and dataset_choice == "arc_agi1":
        st.subheader("🔄 Holdout vs Test Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Test Input**")
            test_input_np = tensor_to_grayscale_numpy(
                task["training_targets"][-1]["input"]
            )
            rgb_test_input = np.zeros((*test_input_np.shape, 3))
            for i, color in enumerate(ARC_COLORS):
                mask = test_input_np == i
                rgb_test_input[mask] = (
                    np.array(
                        [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                    )
                    / 255.0
                )
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(rgb_test_input)
            ax.set_title("Test Input", fontsize=12)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.write("**Holdout Input**")
            holdout_input_np = tensor_to_grayscale_numpy(
                task["holdout_target"]["input"]
            )
            rgb_holdout_input = np.zeros((*holdout_input_np.shape, 3))
            for i, color in enumerate(ARC_COLORS):
                mask = holdout_input_np == i
                rgb_holdout_input[mask] = (
                    np.array(
                        [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                    )
                    / 255.0
                )
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(rgb_holdout_input)
            ax.set_title("Holdout Input", fontsize=12)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        # Check if they're the same
        test_input = task["training_targets"][-1]["input"]
        holdout_input = task["holdout_target"]["input"]
        are_same = torch.equal(test_input, holdout_input)

        if are_same:
            st.error("⚠️ Holdout and Test inputs are the same! This indicates a bug.")
        else:
            st.success(
                "✅ Holdout and Test inputs are different - holdout is working correctly!"
            )

    # Task information
    st.subheader("📋 Task Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Training Examples**")
        st.write(f"Number: {len(task['training_targets'])}")
        st.write(f"Rule latent pairs: {len(task['rule_latent_inputs'])}")

    with col2:
        st.write("**Holdout Status**")
        if has_holdout:
            st.write("✅ Has holdout data")
            st.write(f"Holdout input shape: {task['holdout_target']['input'].shape}")
        else:
            st.write("❌ No holdout data")

    with col3:
        st.write("**Combination Info**")
        if "combination_info" in task:
            combo_info = task["combination_info"]
            st.write(f"Task ID: {combo_info.get('task_id', 'N/A')}")
            st.write(f"Combination: {combo_info.get('pair_indices', 'N/A')}")
            st.write(
                f"Total combinations: {combo_info.get('total_combinations', 'N/A')}"
            )
        else:
            st.write("No combination info")

    # Data statistics
    st.subheader("📈 Data Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Rule Latent Images (RGB)**")
        example_img = task["rule_latent_inputs"][0]["input"]
        st.write(f"- Shape: {example_img.shape}")
        st.write(f"- Data type: {example_img.dtype}")
        st.write(f"- Value range: [{example_img.min():.3f}, {example_img.max():.3f}]")

    with col2:
        st.write("**Test Images (Grayscale)**")
        test_img = task["training_targets"][-1]["input"]
        st.write(f"- Shape: {test_img.shape}")
        st.write(f"- Data type: {test_img.dtype}")
        st.write(f"- Value range: [{test_img.min():.0f}, {test_img.max():.0f}]")

    with col3:
        if has_holdout:
            st.write("**Holdout Images (Grayscale)**")
            holdout_img = task["holdout_target"]["input"]
            st.write(f"- Shape: {holdout_img.shape}")
            st.write(f"- Data type: {holdout_img.dtype}")
            st.write(
                f"- Value range: [{holdout_img.min():.0f}, {holdout_img.max():.0f}]"
            )
        else:
            st.write("**Holdout Status**")
            st.write("No holdout data available")

    # Refresh button
    if st.button("🔄 Refresh Task"):
        st.rerun()


if __name__ == "__main__":
    main()
