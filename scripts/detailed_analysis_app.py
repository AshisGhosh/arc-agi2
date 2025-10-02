#!/usr/bin/env python3
"""
streamlit app for detailed model analysis with forward passes, intermediate outputs, and losses.

focused on deep analysis of model behavior with interactive visualizations.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

from algo.config.config import Config
from algo.data import create_dataset
from algo.data.task_subset import TaskSubset
from algo.models import create_model
from scripts.view_model_predictions import (
    get_available_experiments,
    load_experiment_info,
    NoiseConfig,
    calculate_accuracy_metrics,
    generate_noise_rule_tokens,
    apply_noise_to_examples,
    apply_noise_to_test_inputs,
)
from scripts.visualization_utils import (
    visualize_prediction_comparison,
    apply_arc_color_palette,
    tensor_to_grayscale_numpy,
)


# set page config
st.set_page_config(
    page_title="detailed model analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


class DetailedModelAnalyzer:
    """detailed model analysis with intermediate output capture."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Loss weights (same as transformer trainer)
        self.support_reconstruction_weight = getattr(
            config, "support_reconstruction_weight", 0.1
        )
        self.cls_regularization_weight = getattr(
            config, "cls_regularization_weight", 0.01
        )
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.07)

    def _calculate_support_reconstruction_loss(
        self, support1_input, support1_output, support2_input, support2_output
    ):
        """Calculate support reconstruction loss (single sample version) with individual losses."""
        # Get rule tokens for the support pair
        support_inputs = torch.stack(
            [support1_input, support2_input], dim=1
        )  # [1, 2, 30, 30]
        support_outputs = torch.stack(
            [support1_output, support2_output], dim=1
        )  # [1, 2, 30, 30]
        rule_tokens = self.model.get_rule_tokens(support_inputs, support_outputs)

        # Reshape for processing
        all_support_inputs = support_inputs.view(2, 30, 30)  # [2, 30, 30]
        all_support_outputs = support_outputs.view(2, 30, 30)  # [2, 30, 30]

        # Expand rule tokens for each support example
        expanded_rule_tokens = rule_tokens.repeat_interleave(
            2, dim=0
        )  # [2, num_rule_tokens, d_model]

        # Reconstruct all support examples
        support_processed = self.model.cross_attention_decoder(
            all_support_inputs, expanded_rule_tokens
        )
        support_pred = self.model.output_head(support_processed)  # [2, 10, 30, 30]

        # Calculate individual losses for each support example
        from algo.training.losses import calculate_classification_loss

        # Support 1 loss
        support1_pred = support_pred[0:1]  # [1, 10, 30, 30]
        support1_target = all_support_outputs[0:1]  # [1, 30, 30]
        support1_loss, _ = calculate_classification_loss(
            support1_pred, support1_target, self.config
        )

        # Support 2 loss
        support2_pred = support_pred[1:2]  # [1, 10, 30, 30]
        support2_target = all_support_outputs[1:2]  # [1, 30, 30]
        support2_loss, _ = calculate_classification_loss(
            support2_pred, support2_target, self.config
        )

        # Total loss (average of both)
        total_support_loss = (support1_loss + support2_loss) / 2

        return total_support_loss, support1_loss, support2_loss

    def _calculate_cls_regularization_loss(
        self, support1_input, support1_output, support2_input, support2_output
    ):
        """Calculate CLS regularization loss (single sample version)."""
        # Get pair summaries
        support_inputs = torch.stack(
            [support1_input, support2_input], dim=1
        )  # [1, 2, 30, 30]
        support_outputs = torch.stack(
            [support1_output, support2_output], dim=1
        )  # [1, 2, 30, 30]
        pair_summaries = self.model.get_pair_summaries(support_inputs, support_outputs)

        # Split into R_1 and R_2
        R_1 = pair_summaries[:, 0, :]  # [1, d_model]
        R_2 = pair_summaries[:, 1, :]  # [1, d_model]

        # Normalize embeddings
        R_1_norm = F.normalize(R_1, p=2, dim=1)
        R_2_norm = F.normalize(R_2, p=2, dim=1)

        # Calculate similarity
        similarity = torch.sum(R_1_norm * R_2_norm, dim=1)  # [1]

        # L2 regularization
        l2_loss = torch.mean(torch.norm(R_1, p=2, dim=1) + torch.norm(R_2, p=2, dim=1))

        # Contrastive loss: encourage R_1 and R_2 to be similar
        contrastive_loss = -torch.mean(similarity)

        return contrastive_loss + 0.01 * l2_loss

    def _calculate_rule_token_consistency_loss(
        self,
        rule_tokens_list,
        task_indices,
        augmentation_groups,
        regularization_weight=0.1,
    ):
        """
        Calculate rule token consistency loss across augmentation groups (prototype version).

        This is a simplified version of the ResNet trainer's rule latent regularization,
        adapted for rule tokens in transformer models.

        Args:
            rule_tokens_list: List of rule token tensors [num_samples, num_rule_tokens, d_model]
            task_indices: List of task indices for each sample
            augmentation_groups: List of augmentation groups for each sample
            regularization_weight: Weight for the regularization loss

        Returns:
            Tuple of (consistency_loss, loss_components)
        """
        if len(rule_tokens_list) < 2:
            return torch.tensor(0.0), {
                "rule_token_consistency": 0.0,
                "active_groups": 0,
                "total_pairs": 0,
            }

        # Group rule tokens by (task_idx, augmentation_group)
        groups = {}
        for i, (task_idx, aug_group) in enumerate(
            zip(task_indices, augmentation_groups)
        ):
            key = (task_idx, aug_group)
            if key not in groups:
                groups[key] = []
            groups[key].append(rule_tokens_list[i])

        # Debug: print group information
        print(f"Debug: Created {len(groups)} groups")
        for key, group_tokens in groups.items():
            print(f"  Group {key}: {len(group_tokens)} samples")

        # Calculate within-group similarity loss
        total_loss = 0.0
        group_count = 0
        total_pairs = 0

        for group_tokens in groups.values():
            if len(group_tokens) > 1:  # Need at least 2 samples for regularization
                group_tensor = torch.stack(
                    group_tokens
                )  # [N, num_rule_tokens, d_model]

                # Flatten rule tokens for comparison
                group_flat = group_tensor.view(
                    group_tensor.size(0), -1
                )  # [N, num_rule_tokens * d_model]

                # Calculate pairwise cosine similarity (want high similarity)
                similarities = F.cosine_similarity(
                    group_flat.unsqueeze(1), group_flat.unsqueeze(0), dim=2
                )
                # We want similarities close to 1, so loss is 1 - similarity
                distances = 1 - similarities
                # Only consider upper triangle (avoid duplicates and diagonal)
                mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
                group_loss = distances[mask].mean()

                total_loss += group_loss
                group_count += 1
                total_pairs += mask.sum().item()

        # Normalize by number of groups with multiple samples
        if group_count > 0:
            avg_group_loss = total_loss / group_count
            consistency_loss = regularization_weight * avg_group_loss
            avg_group_loss_value = avg_group_loss.item()
        else:
            consistency_loss = torch.tensor(0.0, device=rule_tokens_list[0].device)
            avg_group_loss_value = 0.0

        loss_components = {
            "rule_token_consistency": consistency_loss.item(),
            "avg_group_loss": avg_group_loss_value,
            "active_groups": group_count,
            "total_pairs": total_pairs,
        }

        return consistency_loss, loss_components

    def run_detailed_forward_pass(self, sample_data, noise_config=None):
        """run detailed forward pass with intermediate output capture."""
        if noise_config is None:
            noise_config = NoiseConfig()

        # extract inputs (already in correct shape from view_model_predictions.py format)
        support1_input = sample_data["support1_input"].to(self.device)
        support1_output = sample_data["support1_output"].to(self.device)
        support2_input = sample_data["support2_input"].to(self.device)
        support2_output = sample_data["support2_output"].to(self.device)
        target_input = sample_data["target_input"].to(self.device)
        target_output = sample_data["target_output"].to(self.device)

        # apply noise to support examples if requested
        if noise_config.has_any_noise():
            support_examples = [
                {
                    "input": support1_input.squeeze(0),
                    "output": support1_output.squeeze(0),
                },
                {
                    "input": support2_input.squeeze(0),
                    "output": support2_output.squeeze(0),
                },
            ]
            noisy_support = apply_noise_to_examples(support_examples, noise_config)
            support1_input = noisy_support[0]["input"].unsqueeze(0)
            support1_output = noisy_support[0]["output"].unsqueeze(0)
            support2_input = noisy_support[1]["input"].unsqueeze(0)
            support2_output = noisy_support[1]["output"].unsqueeze(0)

        # apply noise to target input if requested
        if noise_config.noise_test_inputs:
            test_examples = [
                {"input": target_input.squeeze(0), "output": target_output}
            ]
            noisy_test = apply_noise_to_test_inputs(
                test_examples,
                noise_config.noise_test_type,
                noise_config.noise_test_std,
                noise_config.noise_test_range,
                noise_config.noise_test_ratio,
            )
            target_input = noisy_test[0]["input"].unsqueeze(0)

        # run forward pass with gradient enabled for detailed analysis
        self.model.eval()
        with torch.enable_grad():
            if hasattr(self.model, "get_rule_tokens"):
                # transformer model - use exact same logic as view_model_predictions.py
                if noise_config.inject_noise:
                    # Get rule tokens from the model
                    support_inputs = torch.stack(
                        [support1_input, support2_input], dim=1
                    )  # [1, 2, 30, 30]
                    support_outputs = torch.stack(
                        [support1_output, support2_output], dim=1
                    )  # [1, 2, 30, 30]
                    rule_tokens = self.model.get_rule_tokens(
                        support_inputs, support_outputs
                    )  # [1, num_rule_tokens, d_model]

                    # Apply noise to rule tokens
                    rule_tokens = generate_noise_rule_tokens(
                        rule_tokens,
                        noise_config.noise_type,
                        noise_config.noise_std,
                        noise_config.noise_range,
                        noise_config.noise_ratio,
                    )

                    # Use noisy rule tokens in cross-attention decoder
                    processed_patches = self.model.cross_attention_decoder(
                        target_input, rule_tokens
                    )
                    target_logits = self.model.output_head(processed_patches)
                else:
                    # Normal forward pass - call model directly like view_model_predictions.py
                    target_logits = self.model(
                        support1_input,
                        support1_output,
                        support2_input,
                        support2_output,
                        target_input,
                    )

                    # Get rule tokens and processed patches for analysis (separate call)
                    support_inputs = torch.stack(
                        [support1_input, support2_input], dim=1
                    )  # [1, 2, 30, 30]
                    support_outputs = torch.stack(
                        [support1_output, support2_output], dim=1
                    )  # [1, 2, 30, 30]

                    # Get rule tokens (this returns expanded tokens after bottleneck)
                    rule_tokens = self.model.get_rule_tokens(
                        support_inputs, support_outputs
                    )  # [1, num_rule_tokens, d_model]

                    # Get compressed bottleneck tokens if bottleneck is enabled
                    rule_tokens_compressed = None
                    if (
                        hasattr(self.model, "rule_bottleneck")
                        and self.model.rule_bottleneck is not None
                    ):
                        # Get the rule tokens before bottleneck expansion
                        pair_summaries = self.model.get_pair_summaries(
                            support_inputs, support_outputs
                        )
                        rule_tokens_before_bottleneck = self.model.pma(pair_summaries)
                        # Apply only the down-projection to get compressed tokens
                        rule_tokens_compressed = self.model.rule_bottleneck.down_proj(
                            rule_tokens_before_bottleneck
                        )

                # Get processed patches for analysis
                processed_patches = self.model.cross_attention_decoder(
                    target_input, rule_tokens
                )

                # Get pairwise encoder outputs for CLS loss analysis
                pair_summaries = self.model.get_pair_summaries(
                    support_inputs, support_outputs
                )

                # calculate losses - match view_model_predictions.py exactly
                # target_logits is [1, 10, 30, 30], target_output is [1, 30, 30]
                predictions = torch.argmax(target_logits, dim=1).squeeze(0)  # [30, 30]

                # calculate accuracy metrics using the same logic as view_model_predictions.py
                # target_output is [1, 30, 30], predictions is [30, 30]
                # use calculate_accuracy_metrics function from view_model_predictions.py
                metrics = calculate_accuracy_metrics(predictions, target_output)
                perfect_matches = metrics["perfect_accuracy"] == 1.0
                pixel_accuracy = metrics["pixel_accuracy"]

                # calculate main classification loss using the same method as training
                from algo.training.losses import calculate_classification_loss

                main_loss, main_loss_components = calculate_classification_loss(
                    target_logits, target_output, self.config
                )

                # calculate auxiliary losses (fixed implementation)
                support_loss = torch.tensor(0.0)
                cls_loss = torch.tensor(0.0)
                try:
                    support_loss, support1_loss, support2_loss = (
                        self._calculate_support_reconstruction_loss(
                            support1_input,
                            support1_output,
                            support2_input,
                            support2_output,
                        )
                    )
                except Exception as e:
                    print(f"Warning: Support reconstruction loss failed: {e}")
                    support_loss = torch.tensor(0.0)
                    support1_loss = torch.tensor(0.0)
                    support2_loss = torch.tensor(0.0)

                try:
                    cls_loss = self._calculate_cls_regularization_loss(
                        support1_input, support1_output, support2_input, support2_output
                    )
                except Exception as e:
                    print(f"Warning: CLS regularization loss failed: {e}")

                # total loss (weighted sum like in training)
                total_loss = (
                    main_loss
                    + self.support_reconstruction_weight * support_loss
                    + self.cls_regularization_weight * cls_loss
                )

                # foreground accuracy (excluding background)
                target_output_flat = target_output.squeeze(0)  # [30, 30]
                foreground_mask = target_output_flat != 0
                if foreground_mask.any():
                    # flatten both tensors for indexing
                    predictions_flat = predictions.flatten()
                    target_flat = target_output_flat.flatten()
                    foreground_mask_flat = foreground_mask.flatten()
                    foreground_accuracy = (
                        (
                            predictions_flat[foreground_mask_flat]
                            == target_flat[foreground_mask_flat]
                        )
                        .float()
                        .mean()
                    )
                else:
                    foreground_accuracy = torch.tensor(0.0)

                # Get support reconstruction outputs for analysis
                support_reconstruction = None
                try:
                    # Reconstruct support examples using rule tokens
                    all_support_inputs = support_inputs.view(2, 30, 30)  # [2, 30, 30]
                    expanded_rule_tokens = rule_tokens.repeat_interleave(
                        2, dim=0
                    )  # [2, num_rule_tokens, d_model]
                    support_processed = self.model.cross_attention_decoder(
                        all_support_inputs, expanded_rule_tokens
                    )
                    support_reconstruction = self.model.output_head(
                        support_processed
                    )  # [2, 10, 30, 30]
                except Exception as e:
                    print(f"Warning: Could not get support reconstruction: {e}")

                return {
                    "target_logits": target_logits.detach().cpu(),
                    "predictions": predictions.detach().cpu(),
                    "target_output": target_output_flat.detach().cpu(),
                    # main loss components
                    "main_loss": main_loss.item(),
                    "main_loss_components": main_loss_components,
                    "support_loss": support_loss.item(),
                    "support1_loss": support1_loss.item(),
                    "support2_loss": support2_loss.item(),
                    "cls_loss": cls_loss.item(),
                    "total_loss": total_loss.item(),
                    # legacy field for compatibility
                    "ce_loss": main_loss.item(),
                    "perfect_match": perfect_matches,
                    "pixel_accuracy": pixel_accuracy,
                    "foreground_accuracy": foreground_accuracy.item(),
                    "logits_norm": torch.norm(target_logits).item(),
                    "logits_std": torch.std(target_logits).item(),
                    "logits_min": torch.min(target_logits).item(),
                    "logits_max": torch.max(target_logits).item(),
                    "logits_entropy": -torch.sum(
                        torch.softmax(target_logits, dim=-1)
                        * torch.log_softmax(target_logits, dim=-1)
                    ).item(),
                    # store input data for visualization
                    "support1_input": support1_input.detach().cpu(),
                    "support1_output": support1_output.detach().cpu(),
                    "support2_input": support2_input.detach().cpu(),
                    "support2_output": support2_output.detach().cpu(),
                    "target_input": target_input.detach().cpu(),
                    # store intermediate outputs for transformer models
                    "rule_tokens": rule_tokens.detach().cpu(),
                    "rule_tokens_compressed": rule_tokens_compressed.detach().cpu()
                    if rule_tokens_compressed is not None
                    else None,
                    "processed_patches": processed_patches.detach().cpu(),
                    "pairwise_embeddings": pair_summaries.detach().cpu(),
                    "support_reconstruction": support_reconstruction.detach().cpu()
                    if support_reconstruction is not None
                    else None,
                    "rule_tokens_norm": torch.norm(rule_tokens).item(),
                    "processed_patches_norm": torch.norm(processed_patches).item(),
                    "support_reconstruction_weight": self.support_reconstruction_weight,
                    "cls_regularization_weight": self.cls_regularization_weight,
                }
            else:
                # simple model - no rule tokens or processed patches
                target_logits = self.model(
                    support1_input,
                    support1_output,
                    support2_input,
                    support2_output,
                    target_input,
                )
                # Set dummy values for non-transformer models
                rule_tokens = None
                processed_patches = None

                # calculate losses - match view_model_predictions.py exactly
                # target_logits is [1, 10, 30, 30], target_output is [1, 30, 30]
                predictions = torch.argmax(target_logits, dim=1).squeeze(0)  # [30, 30]

                # calculate accuracy metrics using the same logic as view_model_predictions.py
                # target_output is [1, 30, 30], predictions is [30, 30]
                # use calculate_accuracy_metrics function from view_model_predictions.py
                metrics = calculate_accuracy_metrics(predictions, target_output)
                perfect_matches = metrics["perfect_accuracy"] == 1.0
                pixel_accuracy = metrics["pixel_accuracy"]

                # calculate cross-entropy loss
                target_logits_flat = target_logits.reshape(
                    -1, target_logits.size(1)
                )  # [900, 10]
                target_output_flat = target_output.reshape(-1)  # [900]
                ce_loss = torch.nn.CrossEntropyLoss()(
                    target_logits_flat, target_output_flat.long()
                )

                # foreground accuracy (excluding background)
                target_output_flat = target_output.squeeze(0)  # [30, 30]
                foreground_mask = target_output_flat != 0
                if foreground_mask.any():
                    # flatten both tensors for indexing
                    predictions_flat = predictions.flatten()
                    target_flat = target_output_flat.flatten()
                    foreground_mask_flat = foreground_mask.flatten()
                    foreground_accuracy = (
                        (
                            predictions_flat[foreground_mask_flat]
                            == target_flat[foreground_mask_flat]
                        )
                        .float()
                        .mean()
                    )
                else:
                    foreground_accuracy = torch.tensor(0.0)

                return {
                    "target_logits": target_logits.detach().cpu(),
                    "predictions": predictions.detach().cpu(),
                    "target_output": target_output_flat.detach().cpu(),
                    "ce_loss": ce_loss.item(),
                    "support1_loss": 0.0,
                    "support2_loss": 0.0,
                    "perfect_match": perfect_matches,
                    "pixel_accuracy": pixel_accuracy,
                    "foreground_accuracy": foreground_accuracy.item(),
                    "logits_norm": torch.norm(target_logits).item(),
                    "logits_std": torch.std(target_logits).item(),
                    "logits_min": torch.min(target_logits).item(),
                    "logits_max": torch.max(target_logits).item(),
                    "logits_entropy": -torch.sum(
                        torch.softmax(target_logits, dim=-1)
                        * torch.log_softmax(target_logits, dim=-1)
                    ).item(),
                    # store input data for visualization
                    "support1_input": support1_input.detach().cpu(),
                    "support1_output": support1_output.detach().cpu(),
                    "support2_input": support2_input.detach().cpu(),
                    "support2_output": support2_output.detach().cpu(),
                    "target_input": target_input.detach().cpu(),
                    # no intermediate outputs for simple models
                    "rule_tokens": None,
                    "processed_patches": None,
                    "rule_tokens_norm": 0.0,
                    "processed_patches_norm": 0.0,
                    "support_reconstruction_weight": 0.0,
                    "cls_regularization_weight": 0.0,
                }


def create_arc_heatmap_plot(data, title):
    """create an ARC-style heatmap plot using plotly with proper color mapping."""
    # convert to numpy if it's a tensor
    if hasattr(data, "numpy"):
        data_np = data.numpy()
    else:
        data_np = data

    # apply ARC color palette
    rgb_data = apply_arc_color_palette(data_np)

    fig = go.Figure(data=go.Image(z=rgb_data))
    fig.update_layout(
        title=title, xaxis_title="width", yaxis_title="height", width=400, height=400
    )
    return fig


def create_distribution_plot(data, title, x_label="value"):
    """create a distribution plot using plotly."""
    fig = go.Figure(data=go.Histogram(x=data, nbinsx=30, opacity=0.7))
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title="frequency", width=400, height=300
    )
    return fig


def create_scatter_plot(x_data, y_data, title, x_label, y_label):
    """create a scatter plot using plotly."""
    fig = go.Figure(
        data=go.Scatter(
            x=x_data, y=y_data, mode="markers", marker=dict(size=8, opacity=0.7)
        )
    )
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label, width=400, height=300
    )
    return fig


def main():
    """main streamlit app."""
    st.title("🔬 detailed model analysis")
    st.markdown(
        "deep analysis of model forward passes, intermediate outputs, and losses"
    )

    # sidebar controls
    st.sidebar.header("model selection")
    logs_dir = Path("logs")
    if not logs_dir.exists():
        st.error("❌ logs directory not found")
        st.stop()

    experiments = get_available_experiments(logs_dir)
    if not experiments:
        st.error("❌ no experiments found")
        st.stop()

    experiment_names = [name for name, _ in experiments]
    selected_exp_name = st.sidebar.selectbox(
        "select experiment", experiment_names, index=0
    )
    selected_exp_path = next(
        path for name, path in experiments if name == selected_exp_name
    )

    # load experiment info
    exp_info = load_experiment_info(selected_exp_path)

    # display model type information
    config_info = exp_info.get("full_training_info", {}).get("config", {})
    model_info = exp_info.get("full_training_info", {}).get("model", {})

    # try to get model type from model section first, then config
    model_type = model_info.get("model_type") or config_info.get(
        "model_type", "simple_arc"
    )

    # normalize model type names
    model_type_normalized = {
        "patch_cross_attention": "patch_attention",
        "patch_attention": "patch_attention",
        "simple_arc": "simple_arc",
        "transformer_arc": "transformer_arc",
    }.get(model_type, model_type)

    model_type_display = {
        "simple_arc": "SimpleARC (ResNet + MLP)",
        "patch_attention": "PatchCrossAttention",
        "transformer_arc": "TransformerARC (Transformer + Cross-Attention)",
    }.get(model_type_normalized, model_type)

    st.sidebar.subheader("model info")
    st.sidebar.write(f"**model type:** {model_type_display}")

    # show additional model-specific info
    if model_type_normalized == "patch_attention":
        st.sidebar.write("**architecture:** patch-based cross-attention")
        st.sidebar.write(f"**patch size:** {config_info.get('patch_size', 3)}")
        st.sidebar.write(f"**model dim:** {config_info.get('model_dim', 128)}")
        if "total_parameters" in model_info:
            st.sidebar.write(f"**parameters:** {model_info['total_parameters']:,}")
        # show support-as-test info for patch models
        use_support_as_test = config_info.get("use_support_as_test", False)
        st.sidebar.write(
            f"**support as test:** {'✅ enabled' if use_support_as_test else '❌ disabled'}"
        )
    elif model_type_normalized == "transformer_arc":
        st.sidebar.write(
            "**architecture:** transformer encoder + cross-attention decoder"
        )
        st.sidebar.write(f"**patch size:** {config_info.get('patch_size', 3)}")
        # for transformer models, use d_model from config or model_dim from model_info
        model_dim = config_info.get("d_model", model_info.get("model_dim", 128))
        st.sidebar.write(f"**model dim:** {model_dim}")
        st.sidebar.write(
            f"**num rule tokens:** {config_info.get('num_rule_tokens', 4)}"
        )
        st.sidebar.write(
            f"**num encoder layers:** {config_info.get('num_encoder_layers', 2)}"
        )
        # show rule bottleneck info if enabled
        use_bottleneck = config_info.get("use_rule_bottleneck", False)
        if use_bottleneck:
            bottleneck_dim = config_info.get("rule_bottleneck_dim", 32)
            st.sidebar.write(
                f"**rule bottleneck:** {model_dim} → {bottleneck_dim} → {model_dim}"
            )
        else:
            st.sidebar.write("**rule bottleneck:** disabled")
        if "total_parameters" in model_info:
            st.sidebar.write(f"**parameters:** {model_info['total_parameters']:,}")
    else:
        st.sidebar.write("**architecture:** resnet encoder + mlp decoder")
        st.sidebar.write(f"**rule dim:** {config_info.get('rule_dim', 32)}")
        if "total_parameters" in model_info:
            st.sidebar.write(f"**parameters:** {model_info['total_parameters']:,}")

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

    # load model
    model_path = selected_exp_path / "best_model.pt"
    if not model_path.exists():
        st.error(f"❌ model checkpoint not found: {model_path}")
        st.stop()

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            # convert dict to Config object
            if isinstance(config_dict, dict):
                config = Config(**config_dict)
            else:
                config = config_dict
            st.sidebar.info("✅ loaded config from checkpoint")
        else:
            config = Config()
            # only override with model_info if config is missing
            config.use_rule_bottleneck = model_info.get("use_rule_bottleneck", False)
            config.rule_bottleneck_dim = model_info.get("rule_bottleneck_dim", 16)
            st.sidebar.warning("⚠️ no config in checkpoint, using default config")

        model = create_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        st.sidebar.success("✅ model loaded successfully")
    except Exception as e:
        st.error(f"❌ failed to load model: {e}")
        st.stop()

    # create analyzer
    analyzer = DetailedModelAnalyzer(model, config)

    # dataset selection
    st.sidebar.header("dataset selection")
    task_set = st.sidebar.radio(
        "evaluate on:", ["overfit tasks only", "all test tasks"], index=0
    )

    # dataset augmentation options
    st.sidebar.header("dataset options")
    enable_color_augmentation = st.sidebar.checkbox(
        "enable color augmentation",
        value=False,
        help="apply color relabeling to test model robustness",
    )
    enable_counterfactuals = st.sidebar.checkbox(
        "enable counterfactuals",
        value=False,
        help="include counterfactual (rotated) examples in evaluation",
    )

    # counterfactual options (only show if enabled)
    if enable_counterfactuals:
        counterfactual_Y = st.sidebar.checkbox(
            "counterfactual Y (output)",
            value=True,
            help="apply transformation to output (Y) - original behavior",
        )
        counterfactual_X = st.sidebar.checkbox(
            "counterfactual X (input)",
            value=True,
            help="apply transformation to input (X) - new feature",
        )
        counterfactual_transform = st.sidebar.selectbox(
            "counterfactual transform",
            ["rotate_90", "rotate_180", "rotate_270", "reflect_h", "reflect_v"],
            index=0,
            help="type of transformation to apply",
        )
    else:
        counterfactual_Y = True
        counterfactual_X = True
        counterfactual_transform = "rotate_90"

    # create dataset with selected options
    config.use_color_relabeling = enable_color_augmentation
    config.enable_counterfactuals = enable_counterfactuals
    if enable_counterfactuals:
        config.counterfactual_Y = counterfactual_Y
        config.counterfactual_X = counterfactual_X
        config.counterfactual_transform = counterfactual_transform

    if task_set == "overfit tasks only":
        if "tasks" in exp_info and "task_indices" in exp_info["tasks"]:
            task_indices = exp_info["tasks"]["task_indices"]

            # filter to only the overfit tasks
            dataset = TaskSubset(
                task_indices=task_indices,
                config=config,
                arc_agi1_dir=config.arc_agi1_dir,
                holdout=True,
                use_first_combination_only=False,
            )
            st.sidebar.write(f"**evaluating on {len(task_indices)} overfit tasks**")
        else:
            st.error("❌ no task indices found in experiment info")
            st.stop()
    else:
        dataset = create_dataset(
            config.arc_agi1_dir, config, holdout=True, use_first_combination_only=False
        )
        st.sidebar.write(f"**evaluating on all {len(dataset)} test tasks**")

    # task selection
    st.sidebar.header("task selection")

    # get all unique tasks in the dataset
    all_tasks = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        all_tasks.add(sample["task_idx"])
    all_tasks = sorted(list(all_tasks))

    selected_tasks = st.sidebar.multiselect(
        "select tasks to analyze",
        all_tasks,
        default=[],  # start with no tasks selected - user must choose
        help="select which tasks to include in the analysis",
    )

    # count samples for selected tasks
    filtered_samples = []
    if selected_tasks:
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["task_idx"] in selected_tasks:
                filtered_samples.append((i, sample))
        st.sidebar.info(f"selected {len(selected_tasks)} tasks: {selected_tasks}")
        st.sidebar.info(f"found {len(filtered_samples)} samples for selected tasks")
    else:
        st.sidebar.warning("please select at least one task")
        st.sidebar.info("available tasks: " + ", ".join(map(str, all_tasks)))

    # analysis options
    st.sidebar.header("analysis options")
    evaluation_mode = st.sidebar.selectbox(
        "evaluation mode",
        ["test", "holdout"],
        index=0,
        help="test: evaluate on test examples, holdout: evaluate on holdout examples",
    )
    # analyze all combinations for selected tasks

    # noise configuration
    st.sidebar.header("noise analysis")
    enable_noise = st.sidebar.checkbox("enable noise analysis", value=False)

    noise_config = NoiseConfig()
    if enable_noise:
        noise_config.inject_noise = st.sidebar.checkbox(
            "inject noise into rule tokens", value=False
        )
        if noise_config.inject_noise:
            noise_config.noise_type = st.sidebar.selectbox(
                "noise type", ["gaussian", "uniform", "zeros", "ones"]
            )
            if noise_config.noise_type == "gaussian":
                noise_config.noise_std = st.sidebar.slider(
                    "noise std", 0.1, 2.0, 1.0, 0.1
                )
            elif noise_config.noise_type == "uniform":
                noise_config.noise_range = st.sidebar.slider(
                    "noise range", 0.1, 2.0, 1.0, 0.1
                )
            noise_config.noise_ratio = st.sidebar.slider(
                "noise ratio", 0.0, 1.0, 1.0, 0.1
            )

    # run analysis button
    if st.sidebar.button("🚀 run detailed analysis", type="primary"):
        if not selected_tasks:
            st.error("❌ please select at least one task first")
            st.stop()
        st.session_state.analysis_results = []
        st.session_state.analysis_config = {
            "noise_config": noise_config,
            "enable_noise": enable_noise,
        }

        progress_bar = st.progress(0)
        status_text = st.empty()

        # run analysis
        task_count = 0
        total_combinations = 0

        try:
            # iterate through all filtered samples
            samples_to_process = len(filtered_samples)
            st.write(f"processing {samples_to_process} samples from selected tasks...")

            for sample_idx in range(samples_to_process):
                original_idx, sample = filtered_samples[sample_idx]
                status_text.text(
                    f"analyzing sample {sample_idx} (task {sample['task_idx']}, combo {sample['combination_idx']})..."
                )

                try:
                    # extract task and combination info from the sample
                    task_idx = sample.get("task_idx", 0)
                    combo_idx = sample.get("combination_idx", 0)
                    is_counterfactual = sample.get("is_counterfactual", False)
                    counterfactual_type = sample.get("counterfactual_type", "original")
                    cycling_indices = sample.get("cycling_indices", (0, 0, 0))

                    # determine model type and use appropriate data format
                    # use the loaded config (which might be different from the original config)
                    model_type = (
                        config.model_type
                        if hasattr(config, "model_type")
                        else "transformer_arc"
                    )

                    if model_type in [
                        "simple_arc",
                        "patch_cross_attention",
                        "patch_attention",
                    ]:
                        # use RGB format for ResNet-based models
                        support_examples = sample["support_examples_rgb"]
                        target_example = sample["target_example"]

                        # extract support examples (should be 2 examples)
                        if len(support_examples) < 2:
                            continue

                        support1_input = support_examples[0]["input"].squeeze(
                            0
                        )  # [3, 64, 64]
                        support1_output = support_examples[0]["output"].squeeze(
                            0
                        )  # [3, 64, 64]
                        support2_input = support_examples[1]["input"].squeeze(
                            0
                        )  # [3, 64, 64]
                        support2_output = support_examples[1]["output"].squeeze(
                            0
                        )  # [3, 64, 64]

                        # choose evaluation target based on mode
                        # in cycling format, we should use target_example for the cycled target
                        # test_examples are separate and used for different purposes
                        if evaluation_mode == "test":
                            # for cycling format, use target_example (the cycled target) for evaluation
                            # this matches view_model_predictions.py behavior
                            target_input = target_example["input"].squeeze(
                                1
                            )  # [1, 30, 30]
                            target_output = target_example[
                                "output"
                            ].squeeze(
                                0
                            )  # [1, 30, 30] - squeeze first dim like view_model_predictions.py
                        else:
                            # use target_example (holdout) for evaluation
                            target_input = target_example["input"].squeeze(
                                1
                            )  # [1, 30, 30]
                            target_output = target_example[
                                "output"
                            ].squeeze(
                                0
                            )  # [1, 30, 30] - squeeze first dim like view_model_predictions.py

                    else:
                        # use grayscale format for transformer models
                        support_examples = sample["support_examples"]
                        target_example = sample["target_example"]

                        # extract support examples (should be 2 examples)
                        if len(support_examples) < 2:
                            continue

                        support1_input = support_examples[0]["input"].squeeze(
                            1
                        )  # [1, 30, 30]
                        support1_output = support_examples[0]["output"].squeeze(
                            1
                        )  # [1, 30, 30]
                        support2_input = support_examples[1]["input"].squeeze(
                            1
                        )  # [1, 30, 30]
                        support2_output = support_examples[1]["output"].squeeze(
                            1
                        )  # [1, 30, 30]

                        # choose evaluation target based on mode
                        # in cycling format, we should use target_example for the cycled target
                        # test_examples are separate and used for different purposes
                        if evaluation_mode == "test":
                            # for cycling format, use target_example (the cycled target) for evaluation
                            # this matches view_model_predictions.py behavior
                            target_input = target_example["input"].squeeze(
                                1
                            )  # [1, 30, 30]
                            target_output = target_example[
                                "output"
                            ].squeeze(
                                0
                            )  # [1, 30, 30] - squeeze first dim like view_model_predictions.py
                        else:
                            # use target_example (holdout) for evaluation
                            target_input = target_example["input"].squeeze(
                                1
                            )  # [1, 30, 30]
                            target_output = target_example[
                                "output"
                            ].squeeze(
                                0
                            )  # [1, 30, 30] - squeeze first dim like view_model_predictions.py

                    sample_data = {
                        "support1_input": support1_input,
                        "support1_output": support1_output,
                        "support2_input": support2_input,
                        "support2_output": support2_output,
                        "target_input": target_input,
                        "target_output": target_output,
                    }

                    # run detailed forward pass
                    result = analyzer.run_detailed_forward_pass(
                        sample_data, noise_config
                    )

                    # add metadata
                    result.update(
                        {
                            "task_idx": task_idx,
                            "combination_idx": combo_idx,
                            "is_counterfactual": is_counterfactual,
                            "counterfactual_type": counterfactual_type,
                            "cycling_indices": cycling_indices,
                            "augmentation_group": sample.get("augmentation_group", 0),
                        }
                    )

                    st.session_state.analysis_results.append(result)
                    total_combinations += 1

                except Exception as e:
                    st.warning(f"error analyzing sample {sample_idx}: {e}")
                    import traceback

                    st.code(traceback.format_exc())
                    continue

                # update progress
                progress_bar.progress((sample_idx + 1) / samples_to_process)

            progress_bar.empty()
            status_text.text(
                f"analysis complete! analyzed {total_combinations} combinations"
            )
            st.success(
                f"✅ analysis complete! processed {total_combinations} combinations from {task_count} tasks"
            )

            # update sidebar with task breakdown
            results = st.session_state.analysis_results
            task_breakdown = {}
            for result in results:
                task_idx = result["task_idx"]
                if task_idx not in task_breakdown:
                    task_breakdown[task_idx] = {"total": 0, "perfect": 0}
                task_breakdown[task_idx]["total"] += 1
                if result["perfect_match"]:
                    task_breakdown[task_idx]["perfect"] += 1

            st.sidebar.subheader("📊 analysis summary")
            st.sidebar.write(f"**tasks selected:** {len(selected_tasks)}")
            st.sidebar.write(f"**samples processed:** {len(results)}")
            st.sidebar.write(
                f"**perfect matches:** {sum(1 for r in results if r['perfect_match'])}/{len(results)}"
            )
            st.sidebar.write(
                f"**avg pixel accuracy:** {np.mean([r['pixel_accuracy'] for r in results]):.3f}"
            )
            st.sidebar.write(
                f"**avg ce loss:** {np.mean([r['ce_loss'] for r in results]):.3f}"
            )

            st.sidebar.subheader("📋 task breakdown")
            for task_idx in sorted(task_breakdown.keys()):
                breakdown = task_breakdown[task_idx]
                st.sidebar.write(
                    f"**task {task_idx}:** {breakdown['perfect']}/{breakdown['total']} perfect"
                )

        except Exception as e:
            st.error(f"❌ analysis failed: {e}")
            import traceback

            st.code(traceback.format_exc())

    # display results
    if "analysis_results" in st.session_state and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        config_info = st.session_state.analysis_config

        st.header("📊 analysis results")

        # summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("total combinations", len(results))
        with col2:
            perfect_count = sum(1 for r in results if r["perfect_match"])
            st.metric("perfect matches", f"{perfect_count}/{len(results)}")
        with col3:
            avg_loss = np.mean([r["ce_loss"] for r in results])
            st.metric("avg loss", f"{avg_loss:.4f}")
        with col4:
            avg_pixel_acc = np.mean([r["pixel_accuracy"] for r in results])
            st.metric("avg pixel accuracy", f"{avg_pixel_acc:.3f}")

        # loss distribution
        st.subheader("📈 loss distribution")
        losses = [r["ce_loss"] for r in results]
        fig_loss = create_distribution_plot(
            losses, "cross-entropy loss distribution", "loss"
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # accuracy by task
        st.subheader("🎯 accuracy by task")
        task_accuracies = {}
        for result in results:
            task_idx = result["task_idx"]
            if task_idx not in task_accuracies:
                task_accuracies[task_idx] = []
            task_accuracies[task_idx].append(result["pixel_accuracy"])

        tasks = sorted(task_accuracies.keys())
        avg_accuracies = [np.mean(task_accuracies[task]) for task in tasks]

        fig_acc = go.Figure(data=go.Bar(x=tasks, y=avg_accuracies))
        fig_acc.update_layout(
            title="average pixel accuracy by task",
            xaxis_title="task index",
            yaxis_title="average pixel accuracy",
            width=800,
            height=400,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # intermediate outputs analysis (for transformer models)
        if "rule_tokens" in results[0]:
            st.subheader("🧠 intermediate outputs analysis")

            # rule tokens norm distribution
            rule_norms = [r["rule_tokens_norm"] for r in results]
            fig_rule_norm = create_distribution_plot(
                rule_norms, "rule tokens norm distribution", "norm"
            )
            st.plotly_chart(fig_rule_norm, use_container_width=True)

            # processed patches norm distribution
            patch_norms = [r["processed_patches_norm"] for r in results]
            fig_patch_norm = create_distribution_plot(
                patch_norms, "processed patches norm distribution", "norm"
            )
            st.plotly_chart(fig_patch_norm, use_container_width=True)

            # logits statistics
            st.subheader("📊 logits statistics")
            col1, col2 = st.columns(2)

            with col1:
                logits_norms = [r["logits_norm"] for r in results]
                fig_logits_norm = create_distribution_plot(
                    logits_norms, "logits norm distribution", "norm"
                )
                st.plotly_chart(fig_logits_norm, use_container_width=True)

            with col2:
                logits_entropy = [r["logits_entropy"] for r in results]
                fig_logits_entropy = create_distribution_plot(
                    logits_entropy, "logits entropy distribution", "entropy"
                )
                st.plotly_chart(fig_logits_entropy, use_container_width=True)

        # detailed results table with row selection
        st.subheader("📋 detailed results")
        st.markdown("click on a row to visualize that combination")

        df_data = []
        for i, result in enumerate(results):
            df_data.append(
                {
                    "task": result["task_idx"],
                    "combo": result["combination_idx"],
                    "cycling": str(result["cycling_indices"]),
                    "counterfactual": "✅" if result["is_counterfactual"] else "❌",
                    "total_loss": f"{result['total_loss']:.4f}",
                    "main_loss": f"{result['main_loss']:.4f}",
                    "support_loss": f"{result['support_loss']:.4f}",
                    "cls_loss": f"{result['cls_loss']:.4f}",
                    "pixel_acc": f"{result['pixel_accuracy']:.3f}",
                    "perfect": "✅" if result["perfect_match"] else "❌",
                    "logits_norm": f"{result['logits_norm']:.3f}",
                    "logits_entropy": f"{result['logits_entropy']:.3f}",
                }
            )

        df = pd.DataFrame(df_data)

        # use st.dataframe with selection like view_model_predictions.py
        selected_rows = st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"detailed_table_{len(results)}",
        )

        # handle row selection for visualization
        if selected_rows.selection.rows:
            selected_idx = selected_rows.selection.rows[0]
            selected_result = results[selected_idx]

            # build visualization title
            title = f"🔍 visualizing task {selected_result['task_idx']} combination {selected_result['combination_idx']}"
            if selected_result.get("is_counterfactual", False):
                title += " (counterfactual)"
            st.subheader(title)

            # show detailed metrics for this combination
            st.subheader("📈 combination metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "perfect accuracy",
                    f"{selected_result['pixel_accuracy']:.3f}"
                    if selected_result["perfect_match"]
                    else "0.000",
                )
            with col2:
                st.metric("pixel accuracy", f"{selected_result['pixel_accuracy']:.3f}")
            with col3:
                st.metric("total loss", f"{selected_result['total_loss']:.4f}")
            with col4:
                st.metric(
                    "foreground accuracy",
                    f"{selected_result['foreground_accuracy']:.3f}",
                )

            # loss breakdown
            st.subheader("📊 loss breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("main loss", f"{selected_result['main_loss']:.4f}")
            with col2:
                st.metric("support loss", f"{selected_result['support_loss']:.4f}")
            with col3:
                st.metric("cls loss", f"{selected_result['cls_loss']:.4f}")

            # show combination details
            st.subheader("🔧 combination details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Task ID:** {selected_result['task_idx']}")
                combination_display = f"{selected_result['cycling_indices']}"
                if selected_result.get("is_counterfactual", False):
                    combination_display += " (counterfactual)"
                st.write(f"**Combination:** {combination_display}")
                st.write(f"**Evaluation Mode:** {evaluation_mode}")
            with col2:
                st.write(f"**Logits Norm:** {selected_result['logits_norm']:.3f}")
                st.write(f"**Logits Entropy:** {selected_result['logits_entropy']:.3f}")
                st.write(
                    f"**Status:** {'✅ perfect' if selected_result['perfect_match'] else '⚠️ partial' if selected_result['pixel_accuracy'] > 0.5 else '❌ failed'}"
                )

            # visualization section
            st.subheader("🎨 visualization")

            # prediction comparison using view_model_predictions.py style
            # create sample data structure for visualize_prediction_comparison
            sample_data = {
                "train_examples": [
                    {
                        "input": selected_result.get(
                            "support1_input", torch.zeros(1, 30, 30)
                        ),
                        "output": selected_result.get(
                            "support1_output", torch.zeros(1, 30, 30)
                        ),
                    },
                    {
                        "input": selected_result.get(
                            "support2_input", torch.zeros(1, 30, 30)
                        ),
                        "output": selected_result.get(
                            "support2_output", torch.zeros(1, 30, 30)
                        ),
                    },
                ],
                "target_example": {
                    "input": selected_result.get(
                        "target_input", torch.zeros(1, 30, 30)
                    ),
                    "output": selected_result["target_output"],
                },
            }

            # use the same visualization as view_model_predictions.py
            try:
                import matplotlib.pyplot as plt

                fig = visualize_prediction_comparison(
                    sample_data, selected_result["predictions"], evaluation_mode
                )
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"could not create prediction comparison: {e}")

            # detailed analysis section
            st.subheader("🔍 detailed analysis")
            st.info("ℹ️ Use the prediction comparison above for detailed visualization")

            # logits heatmap (keep as regular heatmap since it's not an image)
            st.write("**logits heatmap**")
            logits_fig = go.Figure(
                data=go.Heatmap(
                    z=selected_result["target_logits"].numpy().squeeze(),
                    colorscale="Viridis",
                    showscale=True,
                )
            )
            logits_fig.update_layout(
                title="logits",
                xaxis_title="width",
                yaxis_title="height",
                width=400,
                height=400,
            )
            st.plotly_chart(logits_fig, use_container_width=True)

            # intermediate outputs (for transformer models)
            if selected_result.get("rule_tokens") is not None:
                st.subheader("🧠 intermediate outputs")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**rule tokens**")

                    # Show compressed tokens if available, otherwise show expanded tokens
                    if (
                        "rule_tokens_compressed" in selected_result
                        and selected_result["rule_tokens_compressed"] is not None
                    ):
                        rule_tokens = (
                            selected_result["rule_tokens_compressed"].numpy().squeeze()
                        )
                        st.write("**Compressed Bottleneck Tokens:**")
                        num_rule_tokens = rule_tokens.shape[0]
                        bottleneck_dim = rule_tokens.shape[1]
                        st.write(
                            f"Shape: {rule_tokens.shape} (num_rule_tokens × rule_bottleneck_dim)"
                        )
                        st.write(f"Dimensions: {num_rule_tokens} × {bottleneck_dim}")
                    else:
                        rule_tokens = selected_result["rule_tokens"].numpy().squeeze()
                        st.write("**Expanded Rule Tokens:**")
                        num_rule_tokens = rule_tokens.shape[0]
                        d_model = rule_tokens.shape[1]
                        st.write(
                            f"Shape: {rule_tokens.shape} (num_rule_tokens × d_model)"
                        )
                        st.write(f"Dimensions: {num_rule_tokens} × {d_model}")

                    if rule_tokens.ndim == 2:
                        rule_fig = go.Figure(
                            data=go.Heatmap(
                                z=rule_tokens, colorscale="Viridis", showscale=True
                            )
                        )
                        rule_fig.update_layout(
                            title=f"rule tokens {rule_tokens.shape}",
                            width=400,
                            height=400,
                        )
                        st.plotly_chart(rule_fig, use_container_width=True)

                with col2:
                    st.write("**processed patches**")
                    processed_patches = (
                        selected_result["processed_patches"].numpy().squeeze()
                    )
                    if processed_patches.ndim == 2:
                        st.write(f"Shape: {processed_patches.shape}")
                        patch_fig = go.Figure(
                            data=go.Heatmap(
                                z=processed_patches,
                                colorscale="Viridis",
                                showscale=True,
                            )
                        )
                        patch_fig.update_layout(
                            title=f"processed patches {processed_patches.shape}",
                            width=400,
                            height=400,
                        )
                        st.plotly_chart(patch_fig, use_container_width=True)

                # CLS loss details
                st.subheader("🔍 CLS loss analysis")
                st.write("**CLS loss breakdown:**")
                st.write(
                    "- **Purpose:** Contrastive loss encouraging support example embeddings to be similar"
                )
                st.write(
                    "- **Source:** Pairwise encoder outputs (R_1, R_2) from support examples"
                )
                st.write(
                    "- **Negative values:** Normal! Means embeddings are similar (good)"
                )
                st.write("- **Formula:** -mean(similarity) + 0.01 × L2_regularization")

                # Show detailed mathematical breakdown
                if (
                    "pairwise_embeddings" in selected_result
                    and selected_result["pairwise_embeddings"] is not None
                ):
                    pairwise_emb = (
                        selected_result["pairwise_embeddings"].numpy().squeeze()
                    )
                    R_1 = pairwise_emb[0]  # [d_model]
                    R_2 = pairwise_emb[1]  # [d_model]

                    # Calculate the actual values used in the loss
                    R_1_norm = R_1 / np.linalg.norm(R_1)
                    R_2_norm = R_2 / np.linalg.norm(R_2)
                    similarity = np.dot(R_1_norm, R_2_norm)

                    l2_norm_1 = np.linalg.norm(R_1)
                    l2_norm_2 = np.linalg.norm(R_2)
                    l2_loss = l2_norm_1 + l2_norm_2

                    contrastive_loss = -similarity
                    regularization_loss = 0.01 * l2_loss
                    total_cls_loss = contrastive_loss + regularization_loss

                    st.write("**Mathematical Breakdown:**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**R_1 norm:** {l2_norm_1:.6f}")
                        st.write(f"**R_2 norm:** {l2_norm_2:.6f}")
                        st.write(f"**L2 regularization:** {l2_loss:.6f}")
                        st.write(f"**0.01 × L2:** {regularization_loss:.6f}")

                    with col2:
                        st.write(f"**R_1 normalized:** {np.linalg.norm(R_1_norm):.6f}")
                        st.write(f"**R_2 normalized:** {np.linalg.norm(R_2_norm):.6f}")
                        st.write(f"**Cosine similarity:** {similarity:.6f}")
                        st.write(f"**Contrastive loss:** {contrastive_loss:.6f}")

                    st.write(
                        f"**Total CLS Loss:** {contrastive_loss:.6f} + {regularization_loss:.6f} = **{total_cls_loss:.6f}**"
                    )

                    # Interpretation
                    if similarity > 0.9:
                        st.success(
                            f"✅ High similarity ({similarity:.3f}) - embeddings are very similar"
                        )
                    elif similarity > 0.5:
                        st.info(
                            f"ℹ️ Moderate similarity ({similarity:.3f}) - embeddings are somewhat similar"
                        )
                    else:
                        st.warning(
                            f"⚠️ Low similarity ({similarity:.3f}) - embeddings are quite different"
                        )

                # Show pairwise encoder outputs if available
                if "pairwise_embeddings" in selected_result:
                    st.write("**Pairwise encoder outputs:**")
                    pairwise_emb = (
                        selected_result["pairwise_embeddings"].numpy().squeeze()
                    )
                    st.write(f"Shape: {pairwise_emb.shape} (2 × d_model)")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**R_1 (first support example)**")
                        r1_fig = go.Figure(
                            data=go.Heatmap(
                                z=pairwise_emb[0:1],
                                colorscale="Viridis",
                                showscale=True,
                            )
                        )
                        r1_fig.update_layout(
                            title="R_1 embedding", width=300, height=200
                        )
                        st.plotly_chart(r1_fig, use_container_width=True)

                    with col2:
                        st.write("**R_2 (second support example)**")
                        r2_fig = go.Figure(
                            data=go.Heatmap(
                                z=pairwise_emb[1:2],
                                colorscale="Viridis",
                                showscale=True,
                            )
                        )
                        r2_fig.update_layout(
                            title="R_2 embedding", width=300, height=200
                        )
                        st.plotly_chart(r2_fig, use_container_width=True)

                # Rule token consistency analysis across augmentation groups
                st.subheader("🔄 rule token consistency analysis")
                st.write("**Rule token consistency across augmentation groups:**")
                st.write(
                    "- **Purpose:** Check if rule tokens are consistent across different augmentations of the same task"
                )
                st.write(
                    "- **Current Status:** No explicit regularization in training (potential improvement opportunity)"
                )
                st.write(
                    "- **Analysis:** Compare rule tokens from different samples of the same task"
                )

                # Find other samples from the same task
                same_task_samples = [
                    r for r in results if r["task_idx"] == selected_result["task_idx"]
                ]

                if len(same_task_samples) > 1:
                    st.write(
                        f"**Found {len(same_task_samples)} samples for task {selected_result['task_idx']}**"
                    )

                    # Check if we have rule tokens for analysis
                    current_rule_tokens = selected_result.get("rule_tokens_compressed")
                    if current_rule_tokens is not None:
                        # Calculate proposed consistency loss
                        st.write("**Proposed Rule Token Consistency Loss:**")
                        try:
                            # Collect rule tokens and metadata for loss calculation
                            rule_tokens_list = []
                            task_indices = []
                            augmentation_groups = []

                            for result in same_task_samples:
                                if (
                                    "rule_tokens_compressed" in result
                                    and result["rule_tokens_compressed"] is not None
                                ):
                                    rule_tokens_list.append(
                                        result["rule_tokens_compressed"]
                                    )
                                    task_indices.append(result["task_idx"])
                                    # Use the augmentation group that's already calculated by the dataset
                                    # This uses get_augmentation_group() which properly handles all cases
                                    aug_group = result.get("augmentation_group", 0)
                                    augmentation_groups.append(aug_group)

                            if len(rule_tokens_list) >= 2:
                                # Calculate consistency loss
                                consistency_loss, loss_components = (
                                    analyzer._calculate_rule_token_consistency_loss(
                                        rule_tokens_list,
                                        task_indices,
                                        augmentation_groups,
                                        regularization_weight=0.1,
                                    )
                                )

                                # Show group statistics
                                st.write("**Group Statistics:**")

                                # Group samples by augmentation group
                                group_samples = {}
                                for i, (result, aug_group) in enumerate(
                                    zip(same_task_samples, augmentation_groups)
                                ):
                                    if (
                                        "rule_tokens_compressed" in result
                                        and result["rule_tokens_compressed"] is not None
                                    ):
                                        if aug_group not in group_samples:
                                            group_samples[aug_group] = []
                                        group_samples[aug_group].append((result, i))

                                # Calculate stats for each group
                                for aug_group in sorted(group_samples.keys()):
                                    group_result_indices = group_samples[aug_group]
                                    if (
                                        len(group_result_indices) >= 2
                                    ):  # Need at least 2 samples for comparison
                                        group_rule_tokens = [
                                            rule_tokens_list[i]
                                            for _, i in group_result_indices
                                        ]

                                        # Calculate pairwise similarities within this group
                                        similarities = []
                                        cosine_similarities = []
                                        l2_distances = []

                                        for i in range(len(group_rule_tokens)):
                                            for j in range(
                                                i + 1, len(group_rule_tokens)
                                            ):
                                                current_tokens = (
                                                    group_rule_tokens[i]
                                                    .numpy()
                                                    .squeeze()
                                                )
                                                other_tokens = (
                                                    group_rule_tokens[j]
                                                    .numpy()
                                                    .squeeze()
                                                )

                                                # Flatten for comparison
                                                current_flat = current_tokens.flatten()
                                                other_flat = other_tokens.flatten()

                                                # Cosine similarity
                                                cos_sim = np.dot(
                                                    current_flat, other_flat
                                                ) / (
                                                    np.linalg.norm(current_flat)
                                                    * np.linalg.norm(other_flat)
                                                )
                                                cosine_similarities.append(cos_sim)

                                                # L2 distance
                                                l2_dist = np.linalg.norm(
                                                    current_flat - other_flat
                                                )
                                                l2_distances.append(l2_dist)

                                                # Overall similarity (1 - normalized distance)
                                                max_possible_dist = np.linalg.norm(
                                                    current_flat
                                                ) + np.linalg.norm(other_flat)
                                                similarity = (
                                                    1 - (l2_dist / max_possible_dist)
                                                    if max_possible_dist > 0
                                                    else 0
                                                )
                                                similarities.append(similarity)

                                        if cosine_similarities:
                                            avg_cosine_sim = np.mean(
                                                cosine_similarities
                                            )
                                            avg_l2_dist = np.mean(l2_distances)
                                            avg_similarity = np.mean(similarities)

                                            # Get group type description
                                            sample_result = group_result_indices[0][0]
                                            counterfactual_type = sample_result.get(
                                                "counterfactual_type", "original"
                                            )
                                            is_counterfactual = sample_result.get(
                                                "is_counterfactual", False
                                            )

                                            if is_counterfactual:
                                                group_desc = f"Group {aug_group} ({counterfactual_type} counterfactuals)"
                                            else:
                                                group_desc = (
                                                    f"Group {aug_group} (original)"
                                                )

                                            st.write(
                                                f"**{group_desc}** ({len(group_result_indices)} samples):"
                                            )
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric(
                                                    "Avg Cosine Similarity",
                                                    f"{avg_cosine_sim:.4f}",
                                                )
                                            with col2:
                                                st.metric(
                                                    "Avg L2 Distance",
                                                    f"{avg_l2_dist:.4f}",
                                                )
                                            with col3:
                                                st.metric(
                                                    "Avg Similarity",
                                                    f"{avg_similarity:.4f}",
                                                )

                                            # Interpretation
                                            if avg_cosine_sim > 0.9:
                                                st.success(
                                                    "✅ High consistency within group"
                                                )
                                            elif avg_cosine_sim > 0.7:
                                                st.info(
                                                    "ℹ️ Moderate consistency within group"
                                                )
                                            elif avg_cosine_sim > 0.5:
                                                st.warning(
                                                    "⚠️ Low consistency within group"
                                                )
                                            else:
                                                st.error(
                                                    "❌ Very low consistency within group"
                                                )

                                    else:
                                        # Single sample group
                                        sample_result = group_result_indices[0][0]
                                        counterfactual_type = sample_result.get(
                                            "counterfactual_type", "original"
                                        )
                                        is_counterfactual = sample_result.get(
                                            "is_counterfactual", False
                                        )

                                        if is_counterfactual:
                                            group_desc = f"Group {aug_group} ({counterfactual_type} counterfactuals)"
                                        else:
                                            group_desc = f"Group {aug_group} (original)"

                                        st.write(
                                            f"**{group_desc}** ({len(group_result_indices)} sample - no comparison possible)"
                                        )

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Consistency Loss",
                                        f"{loss_components['rule_token_consistency']:.4f}",
                                    )
                                with col2:
                                    st.metric(
                                        "Avg Group Loss",
                                        f"{loss_components['avg_group_loss']:.4f}",
                                    )
                                with col3:
                                    st.metric(
                                        "Active Groups",
                                        f"{loss_components['active_groups']}",
                                    )
                                with col4:
                                    st.metric(
                                        "Total Pairs",
                                        f"{loss_components['total_pairs']}",
                                    )

                                # Interpretation of the loss
                                if loss_components["rule_token_consistency"] < 0.01:
                                    st.success(
                                        "✅ Very low consistency loss - rule tokens are already well-regularized"
                                    )
                                elif loss_components["rule_token_consistency"] < 0.05:
                                    st.info(
                                        "ℹ️ Low consistency loss - rule tokens are reasonably consistent"
                                    )
                                elif loss_components["rule_token_consistency"] < 0.1:
                                    st.warning(
                                        "⚠️ Moderate consistency loss - some regularization would help"
                                    )
                                else:
                                    st.error(
                                        "❌ High consistency loss - strong regularization needed"
                                    )

                                st.write("**What this means:**")
                                st.write(
                                    f"- **Current loss:** {loss_components['rule_token_consistency']:.4f} (lower = more consistent)"
                                )
                                st.write(
                                    f"- **Groups analyzed:** {loss_components['active_groups']} groups with multiple samples"
                                )
                                st.write(
                                    f"- **Pairs compared:** {loss_components['total_pairs']} pairwise comparisons"
                                )
                                st.write(
                                    "- **Training impact:** Adding this loss would encourage more consistent rule tokens"
                                )
                            else:
                                st.warning(
                                    "⚠️ Need at least 2 samples with rule tokens to calculate consistency loss"
                                )
                        except Exception as e:
                            st.warning(f"Could not calculate consistency loss: {e}")
                    else:
                        st.warning(
                            "⚠️ No compressed rule tokens available for current sample"
                        )
                else:
                    st.info(
                        "ℹ️ Only one sample found for this task - no consistency analysis possible"
                    )

                # Support loss details
                st.subheader("🔧 support loss analysis")
                st.write("**Support loss breakdown:**")
                st.write(
                    "- **Purpose:** How well the model can reconstruct support examples using rule tokens"
                )
                st.write(
                    "- **Process:** Use rule tokens to reconstruct both support examples, compare with ground truth"
                )
                st.write("- **Value:** Lower is better (0 = perfect reconstruction)")

                # Show detailed loss information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Support Loss", f"{selected_result['support_loss']:.4f}"
                    )
                with col2:
                    st.metric(
                        "Support 1 Loss", f"{selected_result['support1_loss']:.4f}"
                    )
                with col3:
                    st.metric(
                        "Support 2 Loss", f"{selected_result['support2_loss']:.4f}"
                    )
                with col4:
                    st.metric(
                        "Support Loss Weight",
                        f"{selected_result.get('support_reconstruction_weight', 0.1):.3f}",
                    )

                # Show contribution to total loss
                support_contribution = (
                    selected_result.get("support_reconstruction_weight", 0.1)
                    * selected_result["support_loss"]
                )
                st.write(
                    f"**Contribution to Total Loss:** {support_contribution:.4f} (weight × total_support_loss)"
                )

                # Show individual contributions
                support1_contribution = (
                    selected_result.get("support_reconstruction_weight", 0.1)
                    * selected_result["support1_loss"]
                )
                support2_contribution = (
                    selected_result.get("support_reconstruction_weight", 0.1)
                    * selected_result["support2_loss"]
                )
                st.write(
                    f"**Individual Contributions:** Support 1: {support1_contribution:.4f}, Support 2: {support2_contribution:.4f}"
                )

                # Show if reconstructions are perfect matches
                support1_perfect = (
                    selected_result["support1_loss"] < 1e-6
                )  # Very small threshold for "perfect"
                support2_perfect = selected_result["support2_loss"] < 1e-6
                st.write(
                    f"**Perfect Reconstructions:** Support 1: {'✅' if support1_perfect else '❌'}, Support 2: {'✅' if support2_perfect else '❌'}"
                )

                # Show support reconstruction outputs if available
                if (
                    "support_reconstruction" in selected_result
                    and selected_result["support_reconstruction"] is not None
                ):
                    st.write("**Support reconstruction outputs:**")
                    support_recon = selected_result["support_reconstruction"]
                    st.write(f"Shape: {support_recon.shape} (2 × 10 × 30 × 30)")

                    # Create a comparison visualization using matplotlib
                    try:
                        import matplotlib.pyplot as plt

                        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                        fig.suptitle("Support Reconstruction Analysis", fontsize=16)

                        # Original support examples
                        axes[0, 0].set_title("Original Support 1 Input", fontsize=12)
                        orig1_input = tensor_to_grayscale_numpy(
                            selected_result["support1_input"]
                        )
                        orig1_rgb = apply_arc_color_palette(orig1_input)
                        axes[0, 0].imshow(orig1_rgb)
                        axes[0, 0].axis("off")

                        axes[0, 1].set_title("Original Support 1 Output", fontsize=12)
                        orig1_output = tensor_to_grayscale_numpy(
                            selected_result["support1_output"]
                        )
                        orig1_out_rgb = apply_arc_color_palette(orig1_output)
                        axes[0, 1].imshow(orig1_out_rgb)
                        axes[0, 1].axis("off")

                        axes[0, 2].set_title("Original Support 2 Input", fontsize=12)
                        orig2_input = tensor_to_grayscale_numpy(
                            selected_result["support2_input"]
                        )
                        orig2_rgb = apply_arc_color_palette(orig2_input)
                        axes[0, 2].imshow(orig2_rgb)
                        axes[0, 2].axis("off")

                        axes[0, 3].set_title("Original Support 2 Output", fontsize=12)
                        orig2_output = tensor_to_grayscale_numpy(
                            selected_result["support2_output"]
                        )
                        orig2_out_rgb = apply_arc_color_palette(orig2_output)
                        axes[0, 3].imshow(orig2_out_rgb)
                        axes[0, 3].axis("off")

                        # Reconstructed support examples (these are the model's attempts to reconstruct the outputs)
                        recon1 = torch.argmax(support_recon[0], dim=0).numpy()
                        recon2 = torch.argmax(support_recon[1], dim=0).numpy()

                        axes[1, 0].set_title(
                            "Reconstructed Support 1 Output", fontsize=12
                        )
                        recon1_rgb = apply_arc_color_palette(recon1)
                        axes[1, 0].imshow(recon1_rgb)
                        axes[1, 0].axis("off")

                        axes[1, 1].set_title(
                            "Support 1 Reconstruction Error", fontsize=12
                        )
                        # Show difference between original and reconstructed
                        diff1 = np.abs(
                            orig1_output.astype(float) - recon1.astype(float)
                        )
                        axes[1, 1].imshow(diff1, cmap="hot", vmin=0, vmax=9)
                        axes[1, 1].axis("off")

                        axes[1, 2].set_title(
                            "Reconstructed Support 2 Output", fontsize=12
                        )
                        recon2_rgb = apply_arc_color_palette(recon2)
                        axes[1, 2].imshow(recon2_rgb)
                        axes[1, 2].axis("off")

                        axes[1, 3].set_title(
                            "Support 2 Reconstruction Error", fontsize=12
                        )
                        # Show difference between original and reconstructed
                        diff2 = np.abs(
                            orig2_output.astype(float) - recon2.astype(float)
                        )
                        axes[1, 3].imshow(diff2, cmap="hot", vmin=0, vmax=9)
                        axes[1, 3].axis("off")

                        # Add grid to all subplots
                        for ax in axes.flat:
                            ax.set_xticks(np.arange(-0.5, 30, 1), minor=True)
                            ax.set_yticks(np.arange(-0.5, 30, 1), minor=True)
                            ax.grid(
                                which="minor",
                                color="gray",
                                linestyle="-",
                                linewidth=0.5,
                                alpha=0.3,
                            )

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    except Exception as e:
                        st.warning(
                            f"Could not create support reconstruction visualization: {e}"
                        )
            else:
                st.info("ℹ️ intermediate outputs not available for this model type")

    else:
        st.info("👆 click 'run detailed analysis' to start analysis")


if __name__ == "__main__":
    main()
