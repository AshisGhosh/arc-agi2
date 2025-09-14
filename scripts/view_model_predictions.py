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
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

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
        # add batch dimension if needed
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)

        # perfect accuracy (exact match)
        perfect_matches = (predictions == targets).all(dim=(1, 2))
        perfect_accuracy = perfect_matches.float().mean().item()

        # pixel accuracy
        pixel_matches = predictions == targets
        pixel_accuracy = pixel_matches.float().mean().item()

        # near miss accuracy (within 1 pixel)
        diff = torch.abs(predictions.float() - targets.float())
        near_miss = (diff <= 1.0).all(dim=(1, 2))
        near_miss_accuracy = near_miss.float().mean().item()

        return {
            "perfect_accuracy": perfect_accuracy,
            "pixel_accuracy": pixel_accuracy,
            "near_miss_accuracy": near_miss_accuracy,
        }


def evaluate_model_on_tasks(model, dataset, config, progress_bar=None):
    """evaluate model on all tasks in dataset and return results."""
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if progress_bar:
                progress_bar.progress((i + 1) / len(dataset))
            
            # ensure batch has proper structure
            if isinstance(batch, dict):
                # single sample, add batch dimension
                batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # run model inference
            logits = model(
                batch["example1_input"],
                batch["example1_output"],
                batch["example2_input"],
                batch["example2_output"],
                batch["target_input"],
            )
            
            # convert to predictions
            predictions = torch.argmax(logits, dim=1).squeeze(0)
            
            # calculate metrics
            metrics = calculate_accuracy_metrics(predictions, batch["target_output"])
            
            # store results
            results.append({
                "task_idx": i,
                "perfect_accuracy": metrics["perfect_accuracy"],
                "pixel_accuracy": metrics["pixel_accuracy"],
                "near_miss_accuracy": metrics["near_miss_accuracy"],
                "batch": batch,
                "predictions": predictions,
                "logits": logits
            })
    
    return results

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

    # task set selection
    st.sidebar.subheader("task set selection")
    task_set = st.sidebar.radio(
        "evaluate on:",
        ["overfit tasks only", "all test tasks"],
        index=0,
    )

    # load model
    model_path = selected_exp_path / "best_model.pt"
    if not model_path.exists():
        st.error(f"❌ model checkpoint not found: {model_path}")
        st.stop()

    try:
        config = Config()
        model = SimpleARCModel(config)
        
        # load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        st.sidebar.success("✅ model loaded successfully")
    except Exception as e:
        st.error(f"❌ failed to load model: {e}")
        st.stop()

    # load dataset
    dataset = ARCDataset(config.processed_dir, config)

    # create task subset based on selection
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

    # evaluation button
    if st.sidebar.button("🚀 evaluate model", type="primary"):
        # create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("evaluating model on tasks...")
        
        # evaluate model
        results = evaluate_model_on_tasks(model, dataset, config, progress_bar)
        
        # store results in session state
        st.session_state.evaluation_results = results
        st.session_state.task_set = task_set
        
        progress_bar.empty()
        status_text.text("evaluation complete!")
        
        # auto-rerun to show results
        st.rerun()

    # display results if available
    if "evaluation_results" in st.session_state:
        results = st.session_state.evaluation_results
        current_task_set = st.session_state.get("task_set", "unknown")
        
        st.subheader(f"📊 evaluation results ({current_task_set})")
        
        # create results dataframe
        df_data = []
        for result in results:
            df_data.append({
                "task": result["task_idx"],
                "perfect": f"{result['perfect_accuracy']:.3f}",
                "pixel": f"{result['pixel_accuracy']:.3f}",
                "near_miss": f"{result['near_miss_accuracy']:.3f}",
                "status": "✅ perfect" if result['perfect_accuracy'] > 0.99 else "⚠️ partial" if result['pixel_accuracy'] > 0.5 else "❌ failed"
            })
        
        df = pd.DataFrame(df_data)
        
        # display summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("total tasks", len(results))
        with col2:
            perfect_count = sum(1 for r in results if r['perfect_accuracy'] > 0.99)
            st.metric("perfect tasks", f"{perfect_count}/{len(results)}")
        with col3:
            avg_pixel = np.mean([r['pixel_accuracy'] for r in results])
            st.metric("avg pixel accuracy", f"{avg_pixel:.3f}")
        with col4:
            avg_near_miss = np.mean([r['near_miss_accuracy'] for r in results])
            st.metric("avg near-miss", f"{avg_near_miss:.3f}")
        
        # interactive table
        st.subheader("📋 task results table")
        st.markdown("click on a row to visualize that task")
        
        # use st.dataframe with selection
        selected_rows = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # handle row selection
        if selected_rows.selection.rows:
            selected_idx = selected_rows.selection.rows[0]
            selected_task = results[selected_idx]
            
            st.subheader(f"🔍 visualizing task {selected_task['task_idx']}")
            
            # visualize the selected task
            fig = visualize_prediction_comparison(
                selected_task["batch"], 
                selected_task["predictions"]
            )
            st.pyplot(fig)
            
            # show detailed metrics
            st.subheader("📈 detailed metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("perfect accuracy", f"{selected_task['perfect_accuracy']:.3f}")
            with col2:
                st.metric("pixel accuracy", f"{selected_task['pixel_accuracy']:.3f}")
            with col3:
                st.metric("near miss accuracy", f"{selected_task['near_miss_accuracy']:.3f}")
    else:
        st.info("👆 click 'evaluate model' to run evaluation and see results")


if __name__ == "__main__":
    main()
