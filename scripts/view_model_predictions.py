#!/usr/bin/env python3
"""
streamlit app for visualizing model predictions from overfitting experiments.

interactive web interface to load checkpoints and visualize model outputs.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from algo.config import Config
from algo.data import ARCDataset
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


def visualize_prediction_comparison(sample, prediction):
    """visualize model predictions compared to ground truth."""

    # create figure with 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("model prediction comparison", fontsize=16)

    # example 1
    axes[0, 0].set_title("example 1 input", fontsize=12)
    img1 = denormalize_rgb(sample["example1_input"])
    img1_np = tensor_to_numpy(img1)
    axes[0, 0].imshow(img1_np)
    axes[0, 0].axis("off")

    axes[0, 1].set_title("example 1 output", fontsize=12)
    img2 = denormalize_rgb(sample["example1_output"])
    img2_np = tensor_to_numpy(img2)
    axes[0, 1].imshow(img2_np)
    axes[0, 1].axis("off")

    # example 2
    axes[0, 2].set_title("example 2 input", fontsize=12)
    img3 = denormalize_rgb(sample["example2_input"])
    img3_np = tensor_to_numpy(img3)
    axes[0, 2].imshow(img3_np)
    axes[0, 2].axis("off")

    axes[0, 3].set_title("example 2 output", fontsize=12)
    img4 = denormalize_rgb(sample["example2_output"])
    img4_np = tensor_to_numpy(img4)
    axes[0, 3].imshow(img4_np)
    axes[0, 3].axis("off")

    # target input
    axes[1, 0].set_title("target input", fontsize=12)
    target_input_np = tensor_to_grayscale_numpy(sample["target_input"])
    rgb_target_input = np.zeros((*target_input_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = target_input_np == i
        rgb_target_input[mask] = (
            np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
            / 255.0
        )
    axes[1, 0].imshow(rgb_target_input)
    axes[1, 0].axis("off")

    # ground truth output
    axes[1, 1].set_title("ground truth", fontsize=12)
    target_output_np = tensor_to_grayscale_numpy(sample["target_output"])
    rgb_target_output = np.zeros((*target_output_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = target_output_np == i
        rgb_target_output[mask] = (
            np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
            / 255.0
        )
    axes[1, 1].imshow(rgb_target_output)
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
    diff = np.abs(target_output_np.astype(float) - pred_np.astype(float))
    axes[1, 3].imshow(diff, cmap="hot", vmin=0, vmax=9)
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

    # load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=config.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    return model


def get_available_experiments(logs_dir: Path) -> List[Tuple[str, Path]]:
    """get list of available overfitting experiments."""
    experiments = []

    for item in logs_dir.iterdir():
        if item.is_dir() and item.name.startswith("overfit_"):
            # check if it has a model checkpoint
            if (item / "best_model.pt").exists():
                experiments.append((item.name, item))

    # sort by name (which includes timestamp)
    experiments.sort(key=lambda x: x[0])
    return experiments


def calculate_accuracy_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """calculate accuracy metrics for predictions."""
    with torch.no_grad():
        # perfect accuracy (exact match)
        perfect_matches = (predictions == targets).all(dim=(1, 2, 3))
        perfect_accuracy = perfect_matches.float().mean().item()

        # pixel accuracy
        pixel_matches = predictions == targets
        pixel_accuracy = pixel_matches.float().mean().item()

        # near miss accuracy (within 1 pixel)
        diff = torch.abs(predictions.float() - targets.float())
        near_miss = (diff <= 1.0).all(dim=(1, 2, 3))
        near_miss_accuracy = near_miss.float().mean().item()

        # l1 and l2 losses
        l1_loss = torch.abs(predictions.float() - targets.float()).mean().item()
        l2_loss = torch.pow(predictions.float() - targets.float(), 2).mean().item()

        return {
            "perfect_accuracy": perfect_accuracy,
            "pixel_accuracy": pixel_accuracy,
            "near_miss_accuracy": near_miss_accuracy,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
        }


def main():
    """main streamlit app."""
    st.title("🤖 arc model predictions")
    st.markdown("visualize model predictions from overfitting experiments")

    # sidebar controls
    st.sidebar.header("experiment controls")

    # get available experiments
    logs_dir = Path("logs")
    if not logs_dir.exists():
        st.error("❌ logs directory not found")
        st.stop()

    experiments = get_available_experiments(logs_dir)
    if not experiments:
        st.error("❌ no overfitting experiments found")
        st.stop()

    # experiment selection
    experiment_names = [name for name, _ in experiments]
    selected_exp_name = st.sidebar.selectbox(
        "select experiment", experiment_names, index=0
    )
    selected_exp_path = next(
        path for name, path in experiments if name == selected_exp_name
    )

    # load experiment info
    exp_info = load_experiment_info(selected_exp_path)

    # display experiment info
    st.sidebar.subheader("experiment info")
    if "training" in exp_info:
        st.sidebar.write(
            f"**best epoch:** {exp_info['training'].get('best_epoch', 'n/a')}"
        )
        st.sidebar.write(
            f"**best loss:** {exp_info['training'].get('best_loss', 'n/a'):.4f}"
        )
        st.sidebar.write(
            f"**total epochs:** {exp_info['training'].get('total_epochs', 'n/a')}"
        )

    if "tasks" in exp_info:
        st.sidebar.write(f"**tasks:** {exp_info['tasks'].get('n_tasks', 'n/a')}")
        st.sidebar.write(
            f"**task indices:** {exp_info['tasks'].get('task_indices', [])}"
        )

    # dataset selection
    dataset_choice = st.sidebar.selectbox(
        "select dataset", ["arc_agi1", "arc_agi2"], index=0
    )

    # task selection mode
    task_mode = st.sidebar.selectbox(
        "task selection",
        ["all tasks (generalization)", "training tasks only (overfitting)"],
        index=1,
    )

    # load dataset and model
    try:
        config = Config()
        config.training_dataset = dataset_choice

        with st.spinner(f"loading {dataset_choice} dataset..."):
            full_dataset = ARCDataset(config.processed_dir, config)

            # create task subset if training tasks only is selected
            if task_mode == "training tasks only (overfitting)" and "tasks" in exp_info:
                task_indices = exp_info["tasks"].get("task_indices", [])
                if task_indices:
                    dataset = Subset(full_dataset, task_indices)
                    st.info(f"using training tasks only: {task_indices}")
                else:
                    dataset = full_dataset
                    st.warning("no training task indices found, using full dataset")
            else:
                dataset = full_dataset
                st.info("using all tasks (generalization test)")

        st.success(f"✅ loaded {len(dataset)} samples from {dataset_choice}")

        # task selection (after dataset is loaded)
        if task_mode == "training tasks only (overfitting)" and "tasks" in exp_info:
            # show only training tasks
            task_indices = exp_info["tasks"].get("task_indices", [])
            if task_indices:
                task_options = [f"task {idx}" for idx in task_indices]
                selected_task_idx = st.sidebar.selectbox(
                    "select task", task_options, index=0
                )
                sample_idx = task_indices.index(int(selected_task_idx.split()[-1]))
            else:
                st.sidebar.error("no training task indices found")
                sample_idx = 0
        else:
            # show all tasks
            total_tasks = len(full_dataset)
            sample_idx = st.sidebar.slider(
                "select task", min_value=0, max_value=total_tasks - 1, value=0, step=1
            )

        with st.spinner("loading model checkpoint..."):
            checkpoint_path = selected_exp_path / "best_model.pt"
            model = load_model_checkpoint(str(checkpoint_path), config)

        st.success(f"✅ loaded model from {selected_exp_name}")

    except Exception as e:
        st.error(f"❌ error loading dataset/model: {e}")
        st.stop()

    # load selected task and get prediction
    try:
        # get single sample
        sample = dataset[sample_idx]

        # move to device and add batch dimension
        batch = {}
        for key in sample:
            batch[key] = sample[key].unsqueeze(0).to(config.device)

        # get model prediction
        with torch.no_grad():
            prediction = model(
                batch["example1_input"],
                batch["example1_output"],
                batch["example2_input"],
                batch["example2_output"],
                batch["target_input"],
            )

        # convert back to single sample format and round to discrete values
        predictions = torch.round(prediction).squeeze(
            0
        )  # remove batch dimension and round
        batch = {k: v.squeeze(0) for k, v in batch.items()}  # remove batch dimension

        st.success(f"✅ generated prediction for task {sample_idx}")

    except Exception as e:
        st.error(f"❌ error generating prediction: {e}")
        st.stop()

    # main content
    st.header("📊 prediction visualization")

    # show current mode
    if task_mode == "training tasks only (overfitting)":
        st.info(
            f"🔬 **overfitting mode**: testing on training tasks only {exp_info.get('tasks', {}).get('task_indices', [])}"
        )
    else:
        st.info(
            "🌐 **generalization mode**: testing on all tasks (including unseen ones)"
        )

    # debug info
    st.subheader("🔍 debug info")
    st.info(
        "ℹ️ model outputs continuous values which are rounded to discrete color indices (0-9)"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(
            f"**prediction range:** [{predictions.min():.0f}, {predictions.max():.0f}]"
        )
    with col2:
        st.write(
            f"**target range:** [{batch['target_output'].min():.0f}, {batch['target_output'].max():.0f}]"
        )
    with col3:
        st.write(f"**unique values in prediction:** {len(torch.unique(predictions))}")

    # color palette
    st.subheader("🎨 arc color palette")
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
    ax.set_title("arc color palette (0-9)", fontsize=14)
    ax.set_xticks(range(11))
    ax.set_yticks([])
    ax.set_xlabel("color index")
    st.pyplot(fig)

    # task visualization
    st.subheader(f"🔍 task {sample_idx} prediction comparison")

    sample_fig = visualize_prediction_comparison(batch, predictions)
    st.pyplot(sample_fig)

    # accuracy metrics for current task
    st.subheader("📈 task accuracy metrics")
    metrics = calculate_accuracy_metrics(
        predictions.unsqueeze(0), batch["target_output"].unsqueeze(0)
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("perfect accuracy", f"{metrics['perfect_accuracy']:.3f}")
    with col2:
        st.metric("pixel accuracy", f"{metrics['pixel_accuracy']:.3f}")
    with col3:
        st.metric("near miss accuracy", f"{metrics['near_miss_accuracy']:.3f}")
    with col4:
        st.metric("l1 loss", f"{metrics['l1_loss']:.3f}")
    with col5:
        st.metric("l2 loss", f"{metrics['l2_loss']:.3f}")

    # refresh button
    if st.button("🔄 refresh task"):
        st.rerun()


if __name__ == "__main__":
    main()
