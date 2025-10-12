#!/usr/bin/env python3
"""
attention analysis utilities for the transformer arc model.
"""

import torch
import plotly.graph_objects as go
from typing import Dict, Any


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    title: str = "attention heatmap",
    width: int = 400,
    height: int = 400,
) -> go.Figure:
    """
    plot attention weights as a heatmap using plotly.

    args:
        attention_weights: [seq_len, seq_len] attention matrix
        title: plot title
        width: figure width
        height: figure height

    returns:
        plotly figure object
    """
    # convert to numpy for plotting
    weights_np = attention_weights.detach().cpu().numpy()

    # create plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=weights_np, colorscale="viridis", showscale=True, hoverongaps=False
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="key positions",
        yaxis_title="query positions",
        width=width,
        height=height,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )

    return fig


def plot_attention_entropy_analysis(analysis_data: Dict[str, Any]) -> go.Figure:
    """
    plot attention entropy analysis across layers and types using plotly.

    args:
        analysis_data: output from model.get_attention_analysis()

    returns:
        plotly figure object
    """
    if "error" in analysis_data:
        # return empty figure if error
        return go.Figure()

    # extract data for plotting
    layers = []
    entropies = []
    sparsities = []
    attn_types = []

    for key, data in analysis_data.items():
        layers.append(data["layer"])
        entropies.append(data["avg_entropy"])
        sparsities.append(data["sparsity"])
        attn_types.append(data["type"])

    # create subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("attention entropy by layer", "attention sparsity by layer"),
        horizontal_spacing=0.1,
    )

    # entropy plot
    unique_layers = sorted(set(layers))
    self_entropies = []
    cross_entropies = []

    for unique_layer in unique_layers:
        self_ent = [
            entropies[i]
            for i, (layer, t) in enumerate(zip(layers, attn_types))
            if layer == unique_layer and t == "self_attention"
        ]
        cross_ent = [
            entropies[i]
            for i, (layer, t) in enumerate(zip(layers, attn_types))
            if layer == unique_layer and t == "cross_attention"
        ]

        self_entropies.append(self_ent[0] if self_ent else 0)
        cross_entropies.append(cross_ent[0] if cross_ent else 0)

    # add entropy traces
    fig.add_trace(
        go.Scatter(
            x=unique_layers,
            y=self_entropies,
            mode="lines+markers",
            name="self-attention",
            line=dict(width=2),
            marker=dict(symbol="circle"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=unique_layers,
            y=cross_entropies,
            mode="lines+markers",
            name="cross-attention",
            line=dict(width=2),
            marker=dict(symbol="square"),
        ),
        row=1,
        col=1,
    )

    # sparsity plot
    self_sparsities = []
    cross_sparsities = []

    for unique_layer in unique_layers:
        self_sp = [
            sparsities[i]
            for i, (layer, t) in enumerate(zip(layers, attn_types))
            if layer == unique_layer and t == "self_attention"
        ]
        cross_sp = [
            sparsities[i]
            for i, (layer, t) in enumerate(zip(layers, attn_types))
            if layer == unique_layer and t == "cross_attention"
        ]

        self_sparsities.append(self_sp[0] if self_sp else 0)
        cross_sparsities.append(cross_sp[0] if cross_sp else 0)

    # add sparsity traces
    fig.add_trace(
        go.Scatter(
            x=unique_layers,
            y=self_sparsities,
            mode="lines+markers",
            name="self-attention",
            line=dict(width=2),
            marker=dict(symbol="circle"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=unique_layers,
            y=cross_sparsities,
            mode="lines+markers",
            name="cross-attention",
            line=dict(width=2),
            marker=dict(symbol="square"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # update layout
    fig.update_layout(
        width=800, height=400, title_text="attention analysis", showlegend=True
    )

    # update axes
    fig.update_xaxes(title_text="layer", row=1, col=1)
    fig.update_yaxes(title_text="average entropy", row=1, col=1)
    fig.update_xaxes(title_text="layer", row=1, col=2)
    fig.update_yaxes(title_text="sparsity", row=1, col=2)

    return fig


def analyze_attention_patterns(
    attention_weights: torch.Tensor, patch_size: int = 3, grid_size: int = 10
) -> Dict[str, Any]:
    """
    analyze spatial attention patterns in the attention weights.

    args:
        attention_weights: [seq_len, seq_len] attention matrix
        patch_size: size of each patch
        grid_size: size of the patch grid (10x10 for 30x30 image with 3x3 patches)

    returns:
        dictionary with spatial analysis results
    """
    weights_np = attention_weights.detach().cpu().numpy()
    seq_len = weights_np.shape[0]

    # reshape to spatial grid for analysis
    if seq_len == grid_size * grid_size:
        # self-attention: patches attend to patches
        spatial_weights = weights_np.reshape(grid_size, grid_size, grid_size, grid_size)

        # compute spatial statistics
        # 1. local vs global attention
        local_radius = 1  # attend to neighboring patches
        local_attention = 0
        total_attention = 0

        for i in range(grid_size):
            for j in range(grid_size):
                for ki in range(
                    max(0, i - local_radius), min(grid_size, i + local_radius + 1)
                ):
                    for kj in range(
                        max(0, j - local_radius), min(grid_size, j + local_radius + 1)
                    ):
                        local_attention += spatial_weights[i, j, ki, kj]
                total_attention += spatial_weights[i, j, :, :].sum()

        local_ratio = local_attention / total_attention if total_attention > 0 else 0

        # 2. attention to center vs edges
        center_start = grid_size // 3
        center_end = 2 * grid_size // 3
        center_attention = 0
        edge_attention = 0

        for i in range(grid_size):
            for j in range(grid_size):
                for ki in range(grid_size):
                    for kj in range(grid_size):
                        if (
                            center_start <= ki < center_end
                            and center_start <= kj < center_end
                        ):
                            center_attention += spatial_weights[i, j, ki, kj]
                        else:
                            edge_attention += spatial_weights[i, j, ki, kj]

        center_ratio = (
            center_attention / (center_attention + edge_attention)
            if (center_attention + edge_attention) > 0
            else 0
        )

        return {
            "local_attention_ratio": local_ratio,
            "center_attention_ratio": center_ratio,
            "spatial_weights_shape": spatial_weights.shape,
            "analysis_type": "self_attention",
        }

    else:
        # cross-attention: patches attend to rule tokens
        # assume first dimension is patches, second is rule tokens
        patch_attention = weights_np.sum(axis=1)  # [num_patches]
        rule_attention = weights_np.sum(axis=0)  # [num_rule_tokens]

        # analyze which patches attend most to rules
        patch_attention_reshaped = patch_attention.reshape(grid_size, grid_size)

        # compute spatial statistics for patch attention
        center_start = grid_size // 3
        center_end = 2 * grid_size // 3
        center_patch_attention = patch_attention_reshaped[
            center_start:center_end, center_start:center_end
        ].sum()
        total_patch_attention = patch_attention.sum()
        center_patch_ratio = (
            center_patch_attention / total_patch_attention
            if total_patch_attention > 0
            else 0
        )

        return {
            "center_patch_attention_ratio": center_patch_ratio,
            "rule_attention_distribution": rule_attention,
            "patch_attention_shape": patch_attention_reshaped.shape,
            "analysis_type": "cross_attention",
        }


def compare_attention_across_heads(
    attention_weights: torch.Tensor, num_heads: int, layer_name: str = "layer"
) -> go.Figure:
    """
    compare attention patterns across different heads using plotly.

    args:
        attention_weights: [num_heads, seq_len, seq_len] attention weights
        num_heads: number of attention heads
        layer_name: name for the layer (for plotting)

    returns:
        plotly figure object
    """
    if attention_weights.dim() != 3:
        print(f"expected 3d tensor [heads, seq, seq], got {attention_weights.shape}")
        return go.Figure()

    # create subplots
    from plotly.subplots import make_subplots

    cols = min(4, num_heads)  # max 4 columns
    rows = (num_heads + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{layer_name} - head {i}" for i in range(num_heads)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    for head_idx in range(num_heads):
        row = head_idx // cols + 1
        col = head_idx % cols + 1

        weights_np = attention_weights[head_idx].detach().cpu().numpy()

        fig.add_trace(
            go.Heatmap(
                z=weights_np, colorscale="viridis", showscale=False, hoverongaps=False
            ),
            row=row,
            col=col,
        )

    # update layout
    fig.update_layout(
        width=200 * cols,
        height=200 * rows,
        title_text=f"attention patterns across heads - {layer_name}",
        showlegend=False,
    )

    # hide tick labels for all subplots
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(showticklabels=False, row=i, col=j)
            fig.update_yaxes(showticklabels=False, row=i, col=j)

    return fig
