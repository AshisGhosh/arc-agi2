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
from pathlib import Path
import argparse

from algo.config import Config
from algo.data import ARCDataset

# Set page config
st.set_page_config(
    page_title="ARC Dataset Viewer",
    page_icon="🔍",
    layout="wide"
)

# ARC color palette (official 10 colors)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Grey
    '#F012BE',  # 6: Fuschia
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Teal
    '#870C25',  # 9: Brown
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
        return tensor.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dims
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
            rgb_img[mask] = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
        ax.imshow(rgb_img)
    
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, img_np.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, img_np.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    return fig

def visualize_batch(batch, batch_idx=0):
    """Visualize a single sample from a batch."""
    sample = {k: v[batch_idx] for k, v in batch.items()}
    
    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'ARC Task Sample {batch_idx}', fontsize=16)
    
    # Example 1
    axes[0, 0].set_title('Example 1 Input', fontsize=12)
    img1 = denormalize_rgb(sample['example1_input'])
    img1_np = tensor_to_numpy(img1)
    axes[0, 0].imshow(img1_np)
    axes[0, 0].axis('off')
    
    axes[0, 1].set_title('Example 1 Output', fontsize=12)
    img2 = denormalize_rgb(sample['example1_output'])
    img2_np = tensor_to_numpy(img2)
    axes[0, 1].imshow(img2_np)
    axes[0, 1].axis('off')
    
    # Example 2
    axes[0, 2].set_title('Example 2 Input', fontsize=12)
    img3 = denormalize_rgb(sample['example2_input'])
    img3_np = tensor_to_numpy(img3)
    axes[0, 2].imshow(img3_np)
    axes[0, 2].axis('off')
    
    axes[1, 0].set_title('Example 2 Output', fontsize=12)
    img4 = denormalize_rgb(sample['example2_output'])
    img4_np = tensor_to_numpy(img4)
    axes[1, 0].imshow(img4_np)
    axes[1, 0].axis('off')
    
    # Target
    axes[1, 1].set_title('Target Input', fontsize=12)
    target_input_np = tensor_to_grayscale_numpy(sample['target_input'])
    rgb_target_input = np.zeros((*target_input_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = target_input_np == i
        rgb_target_input[mask] = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
    axes[1, 1].imshow(rgb_target_input)
    axes[1, 1].axis('off')
    
    axes[1, 2].set_title('Target Output', fontsize=12)
    target_output_np = tensor_to_grayscale_numpy(sample['target_output'])
    rgb_target_output = np.zeros((*target_output_np.shape, 3))
    for i, color in enumerate(ARC_COLORS):
        mask = target_output_np == i
        rgb_target_output[mask] = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
    axes[1, 2].imshow(rgb_target_output)
    axes[1, 2].axis('off')
    
    # Add grid to all subplots
    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 30, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    return fig

def show_color_palette():
    """Display the ARC color palette."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    
    for i, color in enumerate(ARC_COLORS):
        ax.add_patch(patches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black'))
        ax.text(i + 0.5, 0.5, str(i), ha='center', va='center', fontsize=12, 
                color='white' if i in [0, 9] else 'black', weight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_title('ARC Color Palette (0-9)', fontsize=14)
    ax.set_xticks(range(11))
    ax.set_yticks([])
    ax.set_xlabel('Color Index')
    
    return fig

def main():
    """Main Streamlit app."""
    st.title("🔍 ARC Dataset Viewer")
    st.markdown("Interactive exploration of preprocessed ARC-AGI dataset")
    
    # Sidebar controls
    st.sidebar.header("Dataset Controls")
    
    # Dataset selection
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        ["arc_agi1", "arc_agi2"],
        index=0
    )
    
    # Batch size selection
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=32,
        value=4,
        step=1
    )
    
    # Sample selection within batch
    sample_idx = st.sidebar.slider(
        "Sample Index in Batch",
        min_value=0,
        max_value=batch_size-1,
        value=0,
        step=1
    )
    
    # Load dataset
    try:
        config = Config()
        config.training_dataset = dataset_choice
        
        with st.spinner(f"Loading {dataset_choice} dataset..."):
            dataset = ARCDataset(config.processed_dir, config)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
        
        st.success(f"✅ Loaded {len(dataset)} samples from {dataset_choice}")
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(dataset))
        with col2:
            st.metric("Batch Size", batch_size)
        with col3:
            st.metric("Batches", len(dataloader))
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()
    
    # Load a batch
    try:
        batch = next(iter(dataloader))
        st.success(f"✅ Loaded batch with {len(batch['example1_input'])} samples")
    except Exception as e:
        st.error(f"❌ Error loading batch: {e}")
        st.stop()
    
    # Main content
    st.header("📊 Batch Visualization")
    
    # Color palette
    st.subheader("🎨 ARC Color Palette")
    palette_fig = show_color_palette()
    st.pyplot(palette_fig)
    
    # Sample visualization
    st.subheader(f"🔍 Sample {sample_idx} in Current Batch")
    
    if sample_idx >= len(batch['example1_input']):
        st.warning(f"Sample index {sample_idx} is out of range for batch size {len(batch['example1_input'])}")
        sample_idx = 0
    
    sample_fig = visualize_batch(batch, sample_idx)
    st.pyplot(sample_fig)
    
    # Batch grid view
    st.subheader("📋 Batch Grid View")
    
    # Show all samples in batch as thumbnails (6 images per sample)
    for i in range(min(4, batch_size)):  # Show max 4 samples
        st.write(f"**Sample {i}**")
        sample = {k: v[i] for k, v in batch.items()}
        
        # Create 6-column grid for all images
        cols = st.columns(6)
        
        # Example 1 Input
        with cols[0]:
            st.write("**Ex1 In**")
            img1 = denormalize_rgb(sample['example1_input'])
            img1_np = tensor_to_numpy(img1)
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.imshow(img1_np)
            ax.set_title('Ex1 Input', fontsize=8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        # Example 1 Output
        with cols[1]:
            st.write("**Ex1 Out**")
            img2 = denormalize_rgb(sample['example1_output'])
            img2_np = tensor_to_numpy(img2)
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.imshow(img2_np)
            ax.set_title('Ex1 Output', fontsize=8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        # Example 2 Input
        with cols[2]:
            st.write("**Ex2 In**")
            img3 = denormalize_rgb(sample['example2_input'])
            img3_np = tensor_to_numpy(img3)
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.imshow(img3_np)
            ax.set_title('Ex2 Input', fontsize=8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        # Example 2 Output
        with cols[3]:
            st.write("**Ex2 Out**")
            img4 = denormalize_rgb(sample['example2_output'])
            img4_np = tensor_to_numpy(img4)
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.imshow(img4_np)
            ax.set_title('Ex2 Output', fontsize=8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        # Target Input
        with cols[4]:
            st.write("**Target In**")
            target_input_np = tensor_to_grayscale_numpy(sample['target_input'])
            rgb_target_input = np.zeros((*target_input_np.shape, 3))
            for j, color in enumerate(ARC_COLORS):
                mask = target_input_np == j
                rgb_target_input[mask] = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.imshow(rgb_target_input)
            ax.set_title('Target Input', fontsize=8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        # Target Output
        with cols[5]:
            st.write("**Target Out**")
            target_output_np = tensor_to_grayscale_numpy(sample['target_output'])
            rgb_target_output = np.zeros((*target_output_np.shape, 3))
            for j, color in enumerate(ARC_COLORS):
                mask = target_output_np == j
                rgb_target_output[mask] = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.imshow(rgb_target_output)
            ax.set_title('Target Output', fontsize=8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("---")  # Separator between samples
    
    # Data statistics
    st.subheader("📈 Data Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Example Images (RGB)**")
        example_img = batch['example1_input'][0]
        st.write(f"- Shape: {example_img.shape}")
        st.write(f"- Data type: {example_img.dtype}")
        st.write(f"- Value range: [{example_img.min():.3f}, {example_img.max():.3f}]")
    
    with col2:
        st.write("**Target Images (Grayscale)**")
        target_img = batch['target_input'][0]
        st.write(f"- Shape: {target_img.shape}")
        st.write(f"- Data type: {target_img.dtype}")
        st.write(f"- Value range: [{target_img.min():.0f}, {target_img.max():.0f}]")
    
    # Refresh button
    if st.button("🔄 Load New Batch"):
        st.rerun()

if __name__ == "__main__":
    main()
