#!/usr/bin/env python3
"""
streamlit app for testing hrm model on arc-agi data
integrates with existing testing interface functionality
"""

import streamlit as st
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Optional
import random

# add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algo.hrm import HRM, HRMCarry, hrm_loss
from utils.hrm_dataset import HRMDataset
from utils.data_loader import load_arc_json, pad_grid_center
import yaml

# color mapping for arc-agi visualization (0-9)
COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue  
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#FF851B',  # 5: orange
    '#B10DC9',  # 6: purple
    '#FF69B4',  # 7: pink
    '#39CCCC',  # 8: teal
    '#F012BE',  # 9: magenta
]

@st.cache_data
def load_config(config_path: str) -> Dict:
    """load hrm configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_arc_task(task_path: str) -> Dict:
    """load arc task from json file"""
    with open(task_path, 'r') as f:
        return json.load(f)

@st.cache_data
def get_available_tasks(data_dir: str) -> List[str]:
    """get list of available arc tasks"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    # get both raw json files and processed pt files
    json_files = list(data_path.glob("**/*.json"))
    pt_files = list(data_path.glob("**/*.pt"))
    
    all_files = []
    for f in json_files + pt_files:
        # return relative path from data_dir, not from parent
        all_files.append(str(f.relative_to(data_path)))
    
    return sorted(all_files)

def visualize_grid(grid: np.ndarray, title: str = "", figsize: Tuple[int, int] = (4, 4)) -> plt.Figure:
    """visualize arc grid with colors"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(grid.shape) == 1:
        # flattened grid, try to reshape
        size = int(np.sqrt(len(grid)))
        if size * size == len(grid):
            grid = grid.reshape(size, size)
        else:
            # assume 30x30 padded
            grid = grid.reshape(30, 30)
    
    height, width = grid.shape
    
    # create grid visualization
    for i in range(height):
        for j in range(width):
            color_idx = int(grid[i, j])
            color = COLORS[color_idx % len(COLORS)]
            
            rect = patches.Rectangle((j, height-1-i), 1, 1, 
                                   linewidth=0.5, edgecolor='white', 
                                   facecolor=color)
            ax.add_patch(rect)
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks(range(width+1))
    ax.set_yticks(range(height+1))
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    return fig

def load_model(config: Dict, checkpoint_path: Optional[str] = None) -> Tuple[HRM, Dict]:
    """load hrm model and return model with loading info"""
    # check for gpu availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        st.success(f"🚀 using GPU: {torch.cuda.get_device_name()}")
    else:
        st.warning("⚠️ using CPU - flash attention will not work, consider using GPU")
    
    model = HRM(config)
    loading_info = {
        "checkpoint_loaded": False,
        "checkpoint_path": checkpoint_path,
        "checkpoint_keys": [],
        "model_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": device
    }
    
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            loading_info["checkpoint_keys"] = list(checkpoint.keys())
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                loading_info["checkpoint_loaded"] = True
                st.success(f"✅ loaded model from {checkpoint_path}")
                
                # show additional checkpoint info
                if 'epoch' in checkpoint:
                    st.info(f"📊 checkpoint epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    st.info(f"📉 checkpoint loss: {checkpoint['loss']:.4f}")
                if 'optimizer_state_dict' in checkpoint:
                    st.info("🔧 optimizer state included")
                    
            else:
                model.load_state_dict(checkpoint)
                loading_info["checkpoint_loaded"] = True
                st.success(f"✅ loaded model from {checkpoint_path}")
                
        except Exception as e:
            st.error(f"❌ failed to load checkpoint: {e}")
            loading_info["checkpoint_loaded"] = False
    else:
        if checkpoint_path:
            st.error(f"❌ checkpoint file not found: {checkpoint_path}")
        else:
            st.warning("⚠️ no checkpoint provided, using random weights")
    
    # move model to device
    model = model.to(device)
    return model, loading_info

def process_arc_task(task_data: Dict, task_id: int = 0) -> Dict:
    """process arc task into hrm format - matches training preprocessing"""
    train_examples = task_data["train"]
    test_examples = task_data.get("test", [])
    
    # combine all examples like in preprocessing
    all_examples = train_examples + test_examples
    
    if len(all_examples) < 3:
        raise ValueError(f"need at least 3 examples, got {len(all_examples)}")
    
    # use first 2 as support, 3rd as query (like in preprocessing)
    support_examples = all_examples[:2]
    query_example = all_examples[2]
    
    # process query example
    query_input = np.array(query_example["input"], dtype=np.int64)
    query_output = np.array(query_example["output"], dtype=np.int64)
    
    # pad to 30x30 and flatten
    query_input_padded = pad_grid_center(query_input, target_size=30)
    query_output_padded = pad_grid_center(query_output, target_size=30)
    
    query_input_flat = torch.from_numpy(query_input_padded.ravel()).long()
    query_output_flat = torch.from_numpy(query_output_padded.ravel()).long()
    
    # create support pairs (exactly 2, like in training)
    support_pairs = []
    for example in support_examples:
        inp = np.array(example["input"], dtype=np.int64)
        out = np.array(example["output"], dtype=np.int64)
        inp_padded = pad_grid_center(inp, target_size=30)
        out_padded = pad_grid_center(out, target_size=30)
        
        support_pairs.append({
            "inp": torch.from_numpy(inp_padded.ravel()).long(),
            "out": torch.from_numpy(out_padded.ravel()).long(),
        })
    
    return {
        "inputs": query_input_flat.unsqueeze(0),  # add batch dimension
        "labels": query_output_flat.unsqueeze(0),
        "puzzle_identifiers": torch.tensor([task_id], dtype=torch.long),
        "support_pairs": support_pairs,  # exactly 2 support examples
        "raw_input": query_input,
        "raw_output": query_output,
        "raw_input_padded": query_input_padded,
        "raw_output_padded": query_output_padded,
        "support_examples": support_examples,  # for display
        "query_example": query_example,  # for display
    }

def run_model_inference(model: HRM, batch: Dict, config: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """run model inference and return predictions and metrics"""
    model.eval()
    
    # create model batch with only tensor items and move to device
    model_batch = {
        "inputs": batch["inputs"].to(device),
        "labels": batch["labels"].to(device), 
        "puzzle_identifiers": batch["puzzle_identifiers"].to(device)
    }
    
    # get support pairs and move to device
    support_pairs = batch.get("support_pairs", [])
    if not support_pairs:
        raise ValueError("Support pairs are required for model inference")
    
    # move support pairs to device
    device_support_pairs = []
    for support_pair in support_pairs:
        device_support_pair = {
            "inp": support_pair["inp"].to(device),
            "out": support_pair["out"].to(device)
        }
        device_support_pairs.append(device_support_pair)
    
    with torch.no_grad():
        # initialize carry
        carry = model.initial_carry(model_batch)
        
        # run model with support examples
        outputs = {}
        for step in range(config.get("halt_max_steps", 4)):
            carry, step_outputs = model(carry, model_batch, [device_support_pairs])
            outputs.update(step_outputs)
            
            # check if all sequences are halted
            if carry.halted.all():
                break
        
        # get final predictions
        logits = outputs["logits"]  # (B, L, vocab_size)
        predictions = torch.argmax(logits, dim=-1)  # (B, L)
        
        # calculate loss
        loss, metrics = hrm_loss(outputs, model_batch["labels"], carry, config)
        
        return predictions, {
            "loss": loss.item(),
            "metrics": metrics,
            "logits": logits,
            "q_halt_logits": outputs.get("q_halt_logits"),
            "q_continue_logits": outputs.get("q_continue_logits"),
            "steps": carry.steps,
        }

def main():
    st.set_page_config(page_title="HRM Testing Interface", layout="wide")
    
    st.title("HRM Model Testing Interface")
    st.markdown("test your hrm model on arc-agi data with visual feedback")
    
    # sidebar for configuration
    with st.sidebar:
        st.header("configuration")
        
        # model config
        config_path = st.text_input("config path", value="configs/hrm.yaml")
        
        # checkpoint selection with container file browser
        st.subheader("model checkpoint")
        checkpoint_option = st.radio(
            "checkpoint source",
            ["no checkpoint (random weights)", "browse container files", "file path"],
            help="select how to load the model checkpoint"
        )
        
        checkpoint_path = ""
        if checkpoint_option == "browse container files":
            # find checkpoint files in container (works with docker-compose mount)
            checkpoint_files = []
            search_dirs = ["/workspace/src/checkpoints", "/workspace/src", "/workspace/checkpoints", "/workspace", "/checkpoints"]
            
            for search_dir in search_dirs:
                if Path(search_dir).exists():
                    for pattern in ["*.pt", "*.pth", "checkpoint*", "model*", "best*", "last*", "epoch*"]:
                        checkpoint_files.extend(Path(search_dir).glob(pattern))
            
            # also search recursively in workspace/src (docker-compose mount)
            if Path("/workspace/src").exists():
                for pattern in ["**/*.pt", "**/*.pth"]:
                    checkpoint_files.extend(Path("/workspace/src").glob(pattern))
            
            # filter and deduplicate
            checkpoint_files = list(set([f for f in checkpoint_files if f.is_file()]))
            checkpoint_files = sorted(checkpoint_files)
            
            if checkpoint_files:
                checkpoint_options = ["none"] + [str(f) for f in checkpoint_files]
                selected_checkpoint = st.selectbox("select checkpoint file", checkpoint_options)
                if selected_checkpoint != "none":
                    checkpoint_path = selected_checkpoint
                    checkpoint_file = Path(checkpoint_path)
                    if checkpoint_file.exists():
                        file_size = checkpoint_file.stat().st_size / (1024 * 1024)  # MB
                        st.success(f"✅ found: {checkpoint_file.name} ({file_size:.1f} MB)")
            else:
                st.info("no checkpoint files found in container")
                st.text("searched in: /workspace/src/checkpoints, /workspace/src, /workspace/checkpoints")
        elif checkpoint_option == "file path":
            checkpoint_path = st.text_input("checkpoint path", value="", help="path to .pt or .pth checkpoint file")
        
        # display checkpoint info
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.exists():
                file_size = checkpoint_file.stat().st_size / (1024 * 1024)  # MB
                st.success(f"✅ checkpoint found: {checkpoint_file.name} ({file_size:.1f} MB)")
            else:
                st.error(f"❌ checkpoint not found: {checkpoint_path}")
        else:
            st.info("ℹ️ using random weights (no checkpoint)")
        
        # data selection
        data_source = st.selectbox("data source", ["raw json", "processed pt"])
        
        if data_source == "raw json":
            data_dir = st.text_input("raw data directory", 
                                   value="HRM/dataset/raw-data/ARC-AGI-2/data/training")
        else:
            data_dir = st.text_input("processed data directory", 
                                   value="data/agi2/train")
        
        # load available tasks
        available_tasks = get_available_tasks(data_dir)
        if available_tasks:
            selected_task = st.selectbox("select task", available_tasks)
        else:
            st.error(f"no tasks found in {data_dir}")
            return
    
    # main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("task visualization")
        
        # load and display task
        try:
            if data_source == "raw json":
                # fix path duplication issue
                task_path = Path(data_dir) / selected_task
                if not task_path.exists():
                    # try without the duplicated path
                    task_path = Path(selected_task)
                task_data = load_arc_task(task_path)
                
                # display support and query examples
                st.subheader("support examples (for few-shot learning)")
                for i, example in enumerate(processed_batch["support_examples"]):
                    input_grid = np.array(example["input"])
                    output_grid = np.array(example["output"])
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig = visualize_grid(input_grid, f"support {i+1} input")
                        st.pyplot(fig)
                    with col_b:
                        fig = visualize_grid(output_grid, f"support {i+1} output")
                        st.pyplot(fig)
                
                st.subheader("query example (what we're predicting)")
                query_example = processed_batch["query_example"]
                input_grid = np.array(query_example["input"])
                output_grid = np.array(query_example["output"])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = visualize_grid(input_grid, "query input")
                    st.pyplot(fig)
                with col_b:
                    fig = visualize_grid(output_grid, "query output (ground truth)")
                    st.pyplot(fig)
                
                # display test examples if available
                if task_data.get("test"):
                    st.subheader("test examples")
                    for i, example in enumerate(task_data["test"]):
                        input_grid = np.array(example["input"])
                        output_grid = np.array(example["output"])
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            fig = visualize_grid(input_grid, f"test input {i+1}")
                            st.pyplot(fig)
                        with col_b:
                            fig = visualize_grid(output_grid, f"test output {i+1}")
                            st.pyplot(fig)
                
                # process task for model
                processed_batch = process_arc_task(task_data)
                
            else:
                # load processed pt file
                pt_path = Path(data_dir) / selected_task
                data = torch.load(pt_path, map_location='cpu')
                
                # display support examples
                st.subheader("support examples (for few-shot learning)")
                support_pairs = data.get("support_pairs", [])
                for i, pair in enumerate(support_pairs):
                    support_input = pair["inp"].numpy().reshape(30, 30)
                    support_output = pair["out"].numpy().reshape(30, 30)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig = visualize_grid(support_input, f"support {i+1} input")
                        st.pyplot(fig)
                    with col_b:
                        fig = visualize_grid(support_output, f"support {i+1} output")
                        st.pyplot(fig)
                
                # display query example
                st.subheader("query example (what we're predicting)")
                query_input = data["query_inp"].numpy().reshape(30, 30)
                query_output = data["query_out"].numpy().reshape(30, 30)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = visualize_grid(query_input, "query input")
                    st.pyplot(fig)
                with col_b:
                    fig = visualize_grid(query_output, "query output (ground truth)")
                    st.pyplot(fig)
                
                # create batch (already in correct format from preprocessing)
                processed_batch = {
                    "inputs": data["query_inp"].unsqueeze(0),
                    "labels": data["query_out"].unsqueeze(0),
                    "puzzle_identifiers": torch.tensor([data["task_id"]], dtype=torch.long),
                    "support_pairs": support_pairs,
                }
            
        except Exception as e:
            st.error(f"error loading task: {e}")
            return
    
    with col2:
        st.header("model testing")
        
        # load model
        try:
            config = load_config(config_path)
            model, loading_info = load_model(config, checkpoint_path)
            
            # display model info
            st.subheader("model information")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("total parameters", f"{loading_info['model_params']:,}")
            with col_b:
                st.metric("trainable parameters", f"{loading_info['trainable_params']:,}")
            
            if loading_info["checkpoint_loaded"]:
                st.success("✅ model checkpoint loaded successfully")
                if loading_info["checkpoint_keys"]:
                    with st.expander("checkpoint contents"):
                        for key in loading_info["checkpoint_keys"]:
                            st.text(f"• {key}")
            else:
                st.warning("⚠️ using random weights")
            
            # run inference
            if st.button("run model inference"):
                with st.spinner("running model..."):
                    predictions, results = run_model_inference(model, processed_batch, config, loading_info["device"])
                
                # display results
                st.subheader("results")
                
                # loss and metrics
                st.metric("total loss", f"{results['loss']:.4f}")
                
                metrics = results["metrics"]
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("accuracy", f"{metrics['accuracy']:.4f}")
                    st.metric("exact accuracy", f"{metrics['exact_accuracy']:.4f}")
                with col_b:
                    st.metric("q halt accuracy", f"{metrics['q_halt_accuracy']:.4f}")
                    st.metric("steps", f"{metrics['steps']:.0f}")
                
                # detailed losses
                st.subheader("detailed losses")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("lm loss", f"{metrics['lm_loss']:.4f}")
                with col_b:
                    st.metric("q halt loss", f"{metrics['q_halt_loss']:.4f}")
                with col_c:
                    st.metric("q continue loss", f"{metrics['q_continue_loss']:.4f}")
                
                # predictions vs ground truth
                st.subheader("predictions vs ground truth")
                
                # reshape predictions and labels (move to cpu first)
                pred_grid = predictions[0].cpu().numpy().reshape(30, 30)
                true_grid = processed_batch["labels"][0].cpu().numpy().reshape(30, 30)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    fig = visualize_grid(pred_grid, "model prediction")
                    st.pyplot(fig)
                with col_b:
                    fig = visualize_grid(true_grid, "ground truth")
                    st.pyplot(fig)
                with col_c:
                    # difference visualization
                    diff_grid = (pred_grid != true_grid).astype(int)
                    fig = visualize_grid(diff_grid, "differences (red=wrong)")
                    st.pyplot(fig)
                
                # accuracy per position (content-weighted like in loss function)
                true_flat = true_grid.ravel()
                pred_flat = pred_grid.ravel()
                
                # separate content vs background
                content_mask = (true_flat != 0)  # non-background tokens
                background_mask = (true_flat == 0)  # background tokens
                
                # calculate separate accuracies
                content_correct = content_mask & (pred_flat == true_flat)
                background_correct = background_mask & (pred_flat == true_flat)
                
                content_accuracy = content_correct.sum() / content_mask.sum() if content_mask.sum() > 0 else 0.0
                background_accuracy = background_correct.sum() / background_mask.sum() if background_mask.sum() > 0 else 0.0
                overall_accuracy = (pred_flat == true_flat).mean()
                
                # weighted accuracy (same as loss function)
                weighted_accuracy = 0.8 * content_accuracy + 0.2 * background_accuracy
                
                # display all accuracies
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("overall accuracy", f"{overall_accuracy:.4f}")
                with col_b:
                    st.metric("content accuracy", f"{content_accuracy:.4f}")
                with col_c:
                    st.metric("background accuracy", f"{background_accuracy:.4f}")
                with col_d:
                    st.metric("weighted accuracy", f"{weighted_accuracy:.4f}")
                
                # show content coverage
                content_coverage = content_mask.sum() / len(true_flat)
                st.metric("content coverage", f"{content_coverage:.4f}", help="fraction of positions that are content (non-background)")
                
                # q-values
                if results["q_halt_logits"] is not None:
                    st.subheader("q-values")
                    q_halt = torch.sigmoid(results["q_halt_logits"][0].cpu()).item()
                    q_continue = torch.sigmoid(results["q_continue_logits"][0].cpu()).item()
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("q halt", f"{q_halt:.4f}")
                    with col_b:
                        st.metric("q continue", f"{q_continue:.4f}")
                        
        except Exception as e:
            st.error(f"error running model: {e}")
            import traceback
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()