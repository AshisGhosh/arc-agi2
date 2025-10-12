import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple

from .base import BaseARCModel


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for patch grids."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        self.d_model = d_model

        # Create 2D positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use different frequencies for row and column
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model] - patch embeddings
            grid_size: (height, width) of the patch grid
        """
        B, seq_len, d_model = x.shape
        h, w = grid_size

        # Create 2D position indices
        positions = torch.arange(seq_len).view(h, w)  # [h, w]

        # Flatten and get positional encodings
        pos_flat = positions.flatten()  # [seq_len]
        pos_encodings = self.pe[pos_flat]  # [seq_len, d_model]

        # Add to input
        return x + pos_encodings.unsqueeze(0).expand(B, -1, -1)


class TypeEmbedding(nn.Module):
    """Type embedding to distinguish input vs output patches."""

    def __init__(self, d_model: int):
        super().__init__()
        self.input_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.output_embedding = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, is_input: bool) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model] - patch embeddings
            is_input: True for input patches, False for output patches
        """
        type_emb = self.input_embedding if is_input else self.output_embedding
        return x + type_emb.expand(x.shape[0], x.shape[1], -1)


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention."""

    def __init__(
        self, d_model: int, num_heads: int = 8, d_ff: int = 512, dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class PairEncoder(nn.Module):
    """Encodes a single input-output pair into a CLS summary."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = 3
        self.grid_size = 10  # 30x30 image -> 10x10 patches

        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_size * self.patch_size, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional and type embeddings
        self.pos_encoding = PositionalEncoding2D(d_model)
        self.type_embedding = TypeEmbedding(d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_model * 4, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.
        Args:
            x: [B, H, W] or [H, W] - input image
        Returns:
            patches: [B, num_patches, patch_size^2]
        """
        # Handle both [B, H, W] and [H, W] shapes
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [H, W] -> [1, H, W]

        B, H, W = x.shape
        assert H == W == 30, f"Expected 30x30 image, got {H}x{W}"

        # Reshape to patches
        patches = x.view(
            B,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        patches = patches.view(B, -1, self.patch_size * self.patch_size)

        return patches

    def create_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Create attention mask for self-attention within the pair."""
        # All tokens can attend to each other (no masking within pair)
        return None

    def forward(
        self, input_grid: torch.Tensor, output_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single input-output pair into a CLS summary.

        Args:
            input_grid: [B, 30, 30] - input image
            output_grid: [B, 30, 30] - output image

        Returns:
            cls_summary: [B, d_model] - pair summary
        """
        # Ensure all inputs are on the same device as model parameters
        device = next(self.parameters()).device
        input_grid = input_grid.to(device)
        output_grid = output_grid.to(device)

        B = input_grid.shape[0]

        # Patchify both grids
        input_patches = self.patchify(input_grid)  # [B, 100, 9]
        output_patches = self.patchify(output_grid)  # [B, 100, 9]

        # Embed patches
        input_emb = self.patch_embedding(input_patches)  # [B, 100, d_model]
        output_emb = self.patch_embedding(output_patches)  # [B, 100, d_model]

        # Add type embeddings
        input_emb = self.type_embedding(input_emb, is_input=True)
        output_emb = self.type_embedding(output_emb, is_input=False)

        # Concatenate input and output patches
        pair_tokens = torch.cat([input_emb, output_emb], dim=1)  # [B, 200, d_model]

        # Add positional encoding
        pair_tokens = self.pos_encoding(
            pair_tokens, (self.grid_size, self.grid_size * 2)
        )

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        sequence = torch.cat([cls_tokens, pair_tokens], dim=1)  # [B, 201, d_model]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            sequence = block(sequence, self.create_attention_mask(sequence.shape[1]))

        # Extract CLS token (first token)
        cls_summary = sequence[:, 0, :]  # [B, d_model]

        # Project to final representation
        cls_summary = self.output_proj(cls_summary)  # [B, d_model]

        return cls_summary


class PMA(nn.Module):
    """Pooling by Multihead Attention to create rule tokens from pair summaries."""

    def __init__(
        self,
        num_rule_tokens: int = 8,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_rule_tokens = num_rule_tokens
        self.d_model = d_model

        # Learnable rule token seeds
        self.rule_seeds = nn.Parameter(torch.randn(num_rule_tokens, d_model))

        # Multihead attention
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, pair_summaries: torch.Tensor) -> torch.Tensor:
        """
        Create rule tokens from pair summaries using PMA.

        Args:
            pair_summaries: [B, num_pairs, d_model] - pair summaries

        Returns:
            rule_tokens: [B, num_rule_tokens, d_model] - rule tokens
        """
        B, num_pairs, d_model = pair_summaries.shape

        # Expand rule seeds for batch
        rule_seeds = self.rule_seeds.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, num_rule_tokens, d_model]

        # PMA: rule seeds attend to pair summaries
        rule_tokens, _ = self.attention(rule_seeds, pair_summaries, pair_summaries)

        # Residual connection and normalization
        rule_tokens = self.norm(rule_seeds + rule_tokens)

        # Output projection
        rule_tokens = self.output_proj(rule_tokens)

        return rule_tokens


class AlternatingCrossAttentionDecoder(nn.Module):
    """Cross-attention decoder for test input processing."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = 3
        self.grid_size = 10
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_size * self.patch_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)

        # Self-attention layers (alternating with cross-attention)
        self.self_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # RMS normalization (3x per layer: self-attn, cross-attn, ff)
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(num_layers * 3)])

        # Feed-forward networks
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        # attention monitoring
        self.attention_data = {}
        self.enable_attention_monitoring = False

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches."""
        # Handle both [B, H, W] and [H, W] shapes
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [H, W] -> [1, H, W]

        B, H, W = x.shape
        assert H == W == 30, f"Expected 30x30 image, got {H}x{W}"

        patches = x.view(
            B,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        patches = patches.view(B, -1, self.patch_size * self.patch_size)

        return patches

    def enable_attention_probes(self, enable: bool = True):
        """enable or disable attention monitoring."""
        self.enable_attention_monitoring = enable

    def clear_attention_data(self):
        """clear stored attention data."""
        self.attention_data.clear()

    def get_attention_analysis(self) -> Dict[str, Any]:
        """analyze captured attention patterns."""
        if not self.attention_data:
            return {
                "error": "no attention data captured. enable attention monitoring first."
            }

        analysis = {}

        for key, data in self.attention_data.items():
            weights = data[
                "weights"
            ]  # [batch, seq_len, seq_len] from MultiheadAttention
            layer_idx = data["layer"]
            attn_type = data["type"]

            # compute attention statistics
            if weights.dim() == 2:
                # [seq_len, seq_len] - single sample, already averaged
                seq_len = weights.shape[0]
                batch_size = 1
                num_heads = self.num_heads
            elif weights.dim() == 3:
                # [batch, seq_len, seq_len] - average across heads
                batch_size, seq_len, _ = weights.shape
                num_heads = self.num_heads
            elif weights.dim() == 4:
                # [batch, num_heads, seq_len, seq_len] - if we had per-head weights
                batch_size, num_heads, seq_len, _ = weights.shape
            else:
                print(
                    f"warning: unexpected attention weights dimension {weights.dim()} for {key}"
                )
                continue

            # average attention across batch (weights are already averaged across heads)
            if weights.dim() == 2:
                avg_weights = weights  # [seq_len, seq_len] - already averaged
            elif weights.dim() == 3:
                avg_weights = weights.mean(dim=0)  # [seq_len, seq_len]
            else:
                avg_weights = weights.mean(dim=(0, 1))  # [seq_len, seq_len]

            # safety check for empty or invalid weights
            if avg_weights.numel() == 0 or torch.isnan(avg_weights).any():
                print(f"warning: invalid attention weights for {key}")
                continue

            # attention entropy (measure of attention spread)
            # add small epsilon to avoid log(0)
            eps = 1e-8
            entropy = -(avg_weights * torch.log(avg_weights + eps)).sum(dim=-1)
            avg_entropy = entropy.mean().item()

            # safety check for entropy
            if torch.isnan(torch.tensor(avg_entropy)):
                avg_entropy = 0.0

            # attention sparsity (fraction of attention mass in top-k positions)
            # for cross-attention, use the key dimension (rule tokens), not query dimension
            if attn_type == "cross_attention":
                # cross-attention: [query_seq_len, key_seq_len] = [100, 8]
                key_seq_len = avg_weights.shape[1]  # number of rule tokens
                k = min(5, key_seq_len - 1)  # top 5 or all but one rule token
                if k > 0 and key_seq_len > 1:
                    try:
                        topk_values, _ = torch.topk(avg_weights, k, dim=-1)
                        total_attention = avg_weights.sum(dim=-1)
                        # avoid division by zero
                        if total_attention.sum() > 0:
                            sparsity = (
                                (topk_values.sum(dim=-1) / total_attention)
                                .mean()
                                .item()
                            )
                        else:
                            sparsity = 0.0
                    except Exception as e:
                        print(
                            f"warning: cross-attention sparsity calculation failed for {key}: {e}"
                        )
                        sparsity = 0.0
                else:
                    sparsity = 1.0  # if sequence is too short, consider it fully sparse
            else:
                # self-attention: [seq_len, seq_len] = [100, 100]
                k = min(5, seq_len - 1)  # top 5 or all but one position
                if k > 0 and seq_len > 1:
                    try:
                        topk_values, _ = torch.topk(avg_weights, k, dim=-1)
                        total_attention = avg_weights.sum(dim=-1)
                        # avoid division by zero
                        if total_attention.sum() > 0:
                            sparsity = (
                                (topk_values.sum(dim=-1) / total_attention)
                                .mean()
                                .item()
                            )
                        else:
                            sparsity = 0.0
                    except Exception as e:
                        print(
                            f"warning: self-attention sparsity calculation failed for {key}: {e}"
                        )
                        sparsity = 0.0
                else:
                    sparsity = 1.0  # if sequence is too short, consider it fully sparse

            # max attention value
            max_attn = avg_weights.max().item()

            # safety check for max attention
            if torch.isnan(torch.tensor(max_attn)):
                max_attn = 0.0

            # attention variance across heads (set to 0 since we don't have per-head weights)
            head_variance = 0.0

            analysis[key] = {
                "layer": layer_idx,
                "type": attn_type,
                "avg_entropy": avg_entropy,
                "sparsity": sparsity,
                "max_attention": max_attn,
                "head_variance": head_variance,
                "seq_len": seq_len,
                "num_heads": num_heads,
            }

        return analysis

    def get_attention_visualization_data(
        self, layer_idx: int = 0, head_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """get attention weights for visualization."""
        vis_data = {}

        # get self-attention
        self_key = f"self_attn_layer_{layer_idx}"
        if self_key in self.attention_data:
            weights = self.attention_data[self_key]["weights"]
            if weights.dim() == 2:
                vis_data["self_attention"] = weights  # [seq, seq]
            else:
                vis_data["self_attention"] = weights[0]  # [seq, seq]

        # get cross-attention
        cross_key = f"cross_attn_layer_{layer_idx}"
        if cross_key in self.attention_data:
            weights = self.attention_data[cross_key]["weights"]
            if weights.dim() == 2:
                vis_data["cross_attention"] = weights  # [seq, seq]
            else:
                vis_data["cross_attention"] = weights[0]  # [seq, seq]

        return vis_data

    def forward(
        self, test_input: torch.Tensor, rule_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Process test input using cross-attention with rule tokens.

        Args:
            test_input: [B, 30, 30] - test input image
            rule_tokens: [B, num_rule_tokens, d_model] - rule tokens

        Returns:
            output: [B, num_patches, d_model] - processed test patches
        """
        # Ensure all inputs are on the same device as model parameters
        device = next(self.parameters()).device
        test_input = test_input.to(device)
        rule_tokens = rule_tokens.to(device)

        # Patchify test input
        test_patches = self.patchify(test_input)  # [B, 100, 9]

        # Embed patches
        test_emb = self.patch_embedding(test_patches)  # [B, 100, d_model]

        # Add positional encoding
        test_emb = self.pos_encoding(test_emb, (self.grid_size, self.grid_size))

        # Apply alternating self-attention and cross-attention layers
        x = test_emb
        norm_idx = 0

        for i, (self_attn, cross_attn, ff) in enumerate(
            zip(self.self_attention_layers, self.cross_attention_layers, self.ffs)
        ):
            # Self-attention: test patches attend to themselves
            if self.enable_attention_monitoring:
                self_attn_out, self_attn_weights = self_attn(x, x, x, need_weights=True)
                # store attention weights for analysis
                self.attention_data[f"self_attn_layer_{i}"] = {
                    "weights": self_attn_weights.detach().cpu(),
                    "layer": i,
                    "type": "self_attention",
                }
            else:
                self_attn_out, _ = self_attn(x, x, x)
            x = self.norms[norm_idx](x + self_attn_out)
            norm_idx += 1

            # Cross-attention: test patches attend to rule tokens
            if self.enable_attention_monitoring:
                cross_attn_out, cross_attn_weights = cross_attn(
                    x, rule_tokens, rule_tokens, need_weights=True
                )
                # store attention weights for analysis
                self.attention_data[f"cross_attn_layer_{i}"] = {
                    "weights": cross_attn_weights.detach().cpu(),
                    "layer": i,
                    "type": "cross_attention",
                }
            else:
                cross_attn_out, _ = cross_attn(x, rule_tokens, rule_tokens)
            x = self.norms[norm_idx](x + cross_attn_out)
            norm_idx += 1

            # Feed-forward
            ff_out = ff(x)
            x = self.norms[norm_idx](x + ff_out)
            norm_idx += 1

        return x


class RuleBottleneck(nn.Module):
    """Optional bottleneck for compressing and expanding rule tokens."""

    def __init__(self, d_model: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Down-projection: d_model -> bottleneck_dim
        self.down_proj = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Up-projection: bottleneck_dim -> d_model
        self.up_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, rule_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply bottleneck compression and expansion to rule tokens.

        Args:
            rule_tokens: [B, num_rule_tokens, d_model] - rule tokens

        Returns:
            compressed_rule_tokens: [B, num_rule_tokens, d_model] - bottlenecked rule tokens
        """
        # Compress down
        compressed = self.down_proj(rule_tokens)  # [B, num_rule_tokens, bottleneck_dim]

        # Expand back up
        expanded = self.up_proj(compressed)  # [B, num_rule_tokens, d_model]

        return expanded


def icnr_(weight: torch.Tensor, upscale_factor: int = 3):
    # ICNR init for conv before PixelShuffle
    with torch.no_grad():
        out_c, in_c, kH, kW = weight.shape
        assert out_c % (upscale_factor**2) == 0
        nf = out_c // (upscale_factor**2)
        sub = weight.new_zeros(nf, in_c, kH, kW)
        nn.init.kaiming_normal_(sub, mode="fan_out", nonlinearity="relu")
        weight.copy_(sub.repeat_interleave(upscale_factor**2, dim=0))


class PatchOutputHead(nn.Module):
    """
    Patch → sub-pixel logits → PixelShuffle → edge-safe refinement (logits stay logits).
    """

    def __init__(self, d_model: int, num_colors: int, patch_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_colors = num_colors
        self.up = patch_size  # upscale factor (e.g., 3)

        # Predict C * r^2 per patch (sub-pixel logits)
        self.subpixel_proj = nn.Linear(d_model, num_colors * self.up * self.up)

        # Pre-shuffle 1x1 conv (ICNR init) to avoid checkerboard at init
        self.pre_shuffle = nn.Conv2d(
            num_colors * self.up * self.up,
            num_colors * self.up * self.up,
            kernel_size=1,
            bias=False,
        )

        self.pixel_shuffle = nn.PixelShuffle(self.up)

        # Edge-safe refinement on logits: depthwise 3x3 then pointwise 1x1
        self.refine = nn.Sequential(
            nn.Conv2d(
                num_colors,
                num_colors,
                kernel_size=3,
                padding=1,
                groups=num_colors,
                padding_mode="reflect",
            ),
            nn.GELU(),
            nn.Conv2d(num_colors, num_colors, kernel_size=1),
        )

        # Initialize
        icnr_(self.pre_shuffle.weight, upscale_factor=self.up)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: [B, N, model_dim], N must be H_p*W_p (square)
        returns logits: [B, num_colors, H_p*up, W_p*up]
        """
        B, N, D = patch_tokens.shape
        H_p = W_p = int(N**0.5)
        assert H_p * W_p == N, "num_patches must be a perfect square"

        # Sub-pixel logits per patch
        sp = self.subpixel_proj(patch_tokens)  # [B, N, C*r^2]
        sp = (
            sp.view(B, H_p, W_p, self.num_colors * self.up * self.up)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [B, C*r^2, H_p, W_p]

        # ICNR-friendly 1x1 then PixelShuffle to full res logits
        sp = self.pre_shuffle(sp)  # [B, C*r^2, H_p, W_p]
        logits = self.pixel_shuffle(sp)  # [B, C, H_p*up, W_p*up]

        # Light refinement on logits (no softmax here)
        logits = self.refine(logits)  # [B, C, H, W]
        return logits


class TransformerARCModel(BaseARCModel):
    """
    Transformer-based ARC model with PairEncoder, PMA, and cross-attention decoder.

    Architecture:
    1. PairEncoder: Encode support pairs into CLS summaries
    2. PMA: Create rule tokens from pair summaries
    3. AlternatingCrossAttentionDecoder: Process test input with rule tokens
    4. PatchOutputHead: Convert to pixel-level predictions
    """

    def __init__(self, config):
        super().__init__(config)

        # Model hyperparameters
        self.d_model = getattr(config, "d_model", 128)
        self.num_rule_tokens = getattr(config, "num_rule_tokens", 8)
        self.num_encoder_layers = getattr(config, "num_encoder_layers", 3)
        self.num_decoder_layers = getattr(config, "num_decoder_layers", 2)
        self.num_heads = getattr(config, "num_heads", 8)
        self.dropout = getattr(config, "dropout", 0.1)

        # Rule bottleneck parameters
        self.use_rule_bottleneck = getattr(config, "use_rule_bottleneck", False)
        self.rule_bottleneck_dim = getattr(config, "rule_bottleneck_dim", 32)

        # Components
        self.pair_encoder = PairEncoder(
            d_model=self.d_model,
            num_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        self.pma = PMA(
            num_rule_tokens=self.num_rule_tokens,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # Optional rule bottleneck
        if self.use_rule_bottleneck:
            self.rule_bottleneck = RuleBottleneck(
                d_model=self.d_model,
                bottleneck_dim=self.rule_bottleneck_dim,
                dropout=self.dropout,
            )
        else:
            self.rule_bottleneck = None

        self.alternating_cross_attention_decoder = AlternatingCrossAttentionDecoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_decoder_layers,
            dropout=self.dropout,
        )

        self.output_head = PatchOutputHead(
            d_model=self.d_model, num_colors=10, patch_size=3
        )

        # attention monitoring
        self.attention_monitoring_enabled = False

    def forward(
        self,
        example1_input: torch.Tensor,
        example1_output: torch.Tensor,
        example2_input: torch.Tensor,
        example2_output: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through complete model.

        Args:
            example1_input: First input image [B, 30, 30]
            example1_output: First output image [B, 30, 30]
            example2_input: Second input image [B, 30, 30]
            example2_output: Second output image [B, 30, 30]
            target_input: Target input image [B, 30, 30]

        Returns:
            Classification logits [B, 10, 30, 30] for each pixel and color class
        """
        # Ensure all inputs are on the same device as model parameters
        device = next(self.parameters()).device
        example1_input = example1_input.to(device)
        example1_output = example1_output.to(device)
        example2_input = example2_input.to(device)
        example2_output = example2_output.to(device)
        target_input = target_input.to(device)

        # Step 1: Encode support pairs into CLS summaries
        R_1 = self.pair_encoder(example1_input, example1_output)  # [B, d_model]
        R_2 = self.pair_encoder(example2_input, example2_output)  # [B, d_model]

        # Step 2: Stack pair summaries and create rule tokens
        pair_summaries = torch.stack([R_1, R_2], dim=1)  # [B, 2, d_model]
        rule_tokens = self.pma(pair_summaries)  # [B, num_rule_tokens, d_model]

        # Step 2.5: Apply rule bottleneck if enabled
        if self.rule_bottleneck is not None:
            rule_tokens = self.rule_bottleneck(
                rule_tokens
            )  # [B, num_rule_tokens, d_model]

        # Step 3: Process test input with cross-attention
        processed_patches = self.alternating_cross_attention_decoder(
            target_input, rule_tokens
        )  # [B, 100, d_model]

        # Step 4: Convert to pixel-level predictions
        output = self.output_head(processed_patches)  # [B, 10, 30, 30]

        return output

    def forward_with_support_batch(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        test_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with batched support inputs for efficient training.

        Args:
            support_inputs: [B, 2, 30, 30] - batch of support input pairs
            support_outputs: [B, 2, 30, 30] - batch of support output pairs
            test_inputs: [B, 30, 30] - batch of test inputs

        Returns:
            outputs: [B, 10, 30, 30] - batch of predictions
        """
        # Ensure all inputs are on the same device as model parameters
        device = next(self.parameters()).device
        support_inputs = support_inputs.to(device)
        support_outputs = support_outputs.to(device)
        test_inputs = test_inputs.to(device)

        # Get rule tokens for all pairs
        rule_tokens = self.get_rule_tokens(support_inputs, support_outputs)

        # Apply rule bottleneck if enabled
        if self.rule_bottleneck is not None:
            rule_tokens = self.rule_bottleneck(
                rule_tokens
            )  # [B, num_rule_tokens, d_model]

        # Process all test inputs with cross-attention
        processed_patches = self.alternating_cross_attention_decoder(
            test_inputs, rule_tokens
        )

        # Convert to pixel-level predictions
        outputs = self.output_head(processed_patches)

        return outputs

    def get_pair_summaries(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get pair summaries for all support pairs in a batch.

        Args:
            support_inputs: [B, 2, 30, 30] - batch of support input pairs
            support_outputs: [B, 2, 30, 30] - batch of support output pairs

        Returns:
            pair_summaries: [B, 2, d_model] - pair summaries for all pairs
        """
        B = support_inputs.shape[0]

        # Reshape to process all pairs at once
        all_inputs = support_inputs.view(B * 2, 30, 30)  # [2B, 30, 30]
        all_outputs = support_outputs.view(B * 2, 30, 30)  # [2B, 30, 30]

        # Process all pairs in one batch
        all_summaries = self.pair_encoder(all_inputs, all_outputs)  # [2B, d_model]

        # Reshape back to [B, 2, d_model]
        pair_summaries = all_summaries.view(B, 2, -1)

        return pair_summaries

    def get_rule_tokens(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get rule tokens for all support pairs in a batch.

        Args:
            support_inputs: [B, 2, 30, 30] - batch of support input pairs
            support_outputs: [B, 2, 30, 30] - batch of support output pairs

        Returns:
            rule_tokens: [B, num_rule_tokens, d_model]
        """
        # Get pair summaries
        pair_summaries = self.get_pair_summaries(support_inputs, support_outputs)

        # Create rule tokens
        rule_tokens = self.pma(pair_summaries)

        return rule_tokens

    def enable_attention_monitoring(self, enable: bool = True):
        """enable or disable attention monitoring for the decoder."""
        self.attention_monitoring_enabled = enable
        self.alternating_cross_attention_decoder.enable_attention_probes(enable)

    def clear_attention_data(self):
        """clear stored attention data."""
        if hasattr(self.alternating_cross_attention_decoder, "clear_attention_data"):
            self.alternating_cross_attention_decoder.clear_attention_data()

    def get_attention_analysis(self) -> Dict[str, Any]:
        """get attention analysis from the decoder."""
        return self.alternating_cross_attention_decoder.get_attention_analysis()

    def get_attention_visualization_data(
        self, layer_idx: int = 0, head_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """get attention weights for visualization."""
        return (
            self.alternating_cross_attention_decoder.get_attention_visualization_data(
                layer_idx, head_idx
            )
        )

    def print_attention_summary(self):
        """print a summary of attention patterns."""
        analysis = self.get_attention_analysis()

        if "error" in analysis:
            print(f"attention analysis error: {analysis['error']}")
            return

        print("\n=== attention analysis summary ===")
        for key, data in analysis.items():
            attn_type = data["type"]
            layer = data["layer"]
            entropy = data["avg_entropy"]
            sparsity = data["sparsity"]
            max_attn = data["max_attention"]
            head_var = data["head_variance"]

            print(f"{attn_type} layer {layer}:")
            print(f"  entropy: {entropy:.4f} (higher = more spread)")
            print(f"  sparsity: {sparsity:.4f} (higher = more focused)")
            print(f"  max attention: {max_attn:.4f}")
            print(f"  head variance: {head_var:.4f} (higher = more diverse heads)")
            print()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging/debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "transformer_arc",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "patch_size": 3,  # Hardcoded patch size
            "model_dim": self.d_model,
            "num_rule_tokens": self.num_rule_tokens,
            "num_encoder_layers": self.num_encoder_layers,
            "use_rule_bottleneck": self.use_rule_bottleneck,
            "rule_bottleneck_dim": self.rule_bottleneck_dim
            if self.use_rule_bottleneck
            else None,
        }
