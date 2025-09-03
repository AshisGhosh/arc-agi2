"""
HRM-style Hierarchical Reasoning Model with ACT
Recreated using original HRM components but adapted for few-shot learning
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.hrm_components import (
    trunc_normal_init_,
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CastedEmbedding,
    CastedLinear,
    CastedSparseEmbedding,
    CosSin,
)


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------
@dataclass
class HRMInnerCarry:
    """Inner carry state for H and L levels"""

    z_H: torch.Tensor  # (B, L, hidden_size)
    z_L: torch.Tensor  # (B, L, hidden_size)


@dataclass
class HRMCarry:
    """Full carry state with ACT information"""

    inner_carry: HRMInnerCarry
    steps: torch.Tensor  # (B,) current step count
    halted: torch.Tensor  # (B,) whether each sequence is halted
    current_data: Dict[str, torch.Tensor]  # current batch data


@dataclass
class FewShotBatch:
    """Few-shot learning batch"""

    support_inp: torch.Tensor  # (B, K, L) where K=2 support examples
    support_out: torch.Tensor  # (B, K, L) corresponding outputs
    query_inp: torch.Tensor  # (B, L) query input to solve
    query_out: torch.Tensor  # (B, L) query output (target)
    task_id: torch.Tensor  # (B,) task IDs


# ------------------------------------------------------------
# Transformer blocks using original HRM components
# ------------------------------------------------------------
class HRMBlock(nn.Module):
    """Single transformer block with attention + MLP"""

    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.expansion = config["expansion"]
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-5)

        # Self attention
        self.self_attn = Attention(
            hidden_size=self.hidden_size,
            head_dim=self.hidden_size // self.num_heads,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            causal=False,
        )

        # MLP
        self.mlp = SwiGLU(
            hidden_size=self.hidden_size,
            expansion=self.expansion,
        )

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm + Self Attention
        hidden_states = rms_norm(
            hidden_states
            + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.rms_norm_eps,
        )
        # Post Norm + MLP
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states), variance_epsilon=self.rms_norm_eps
        )
        return hidden_states


class HRMReasoningModule(nn.Module):
    """H or L level reasoning module with multiple transformer blocks"""

    def __init__(self, config: Dict, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([HRMBlock(config) for _ in range(num_layers)])

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# ------------------------------------------------------------
# Main HRM model
# ------------------------------------------------------------
class HRM(nn.Module):
    """Hierarchical Reasoning Model with ACT"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Model dimensions
        self.hidden_size = config["hidden_size"]
        self.num_colors = config.get("num_colors", 10)
        self.max_len = config.get("max_len", 900)
        self.num_tasks = config.get("num_tasks", 1000)

        # ACT parameters
        self.halt_max_steps = config.get("halt_max_steps", 16)
        self.halt_exploration_prob = config.get("halt_exploration_prob", 0.1)

        # H/L cycle parameters
        self.H_cycles = config.get("H_cycles", 2)
        self.L_cycles = config.get("L_cycles", 2)
        self.H_layers = config.get("H_layers", 4)
        self.L_layers = config.get("L_layers", 4)

        # Embedding scale
        self.embed_scale = math.sqrt(self.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embeddings
        self.embed_tokens = CastedEmbedding(
            self.num_colors,
            self.hidden_size,
            init_std=embed_init_std,
            cast_to=torch.bfloat16,
        )

        # Puzzle embeddings (task-specific embeddings)
        puzzle_emb_ndim = config.get("puzzle_emb_ndim", self.hidden_size)
        self.puzzle_emb_len = -(puzzle_emb_ndim // -self.hidden_size)  # ceil div
        if puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.num_tasks,
                puzzle_emb_ndim,
                batch_size=config.get("batch_size", 32),
                init_std=0,
                cast_to=torch.bfloat16,
            )

        # Positional embeddings
        pos_encodings = config.get("pos_encodings", "rope")
        if pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.hidden_size // config["num_heads"],
                max_position_embeddings=self.max_len + self.puzzle_emb_len,
                base=config.get("rope_theta", 10000.0),
            )
        elif pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.max_len + self.puzzle_emb_len,
                self.hidden_size,
                init_std=embed_init_std,
                cast_to=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {pos_encodings}")

        # Reasoning modules
        self.H_level = HRMReasoningModule(config, self.H_layers)
        self.L_level = HRMReasoningModule(config, self.L_layers)

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.hidden_size, dtype=torch.bfloat16), std=1
            ),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.hidden_size, dtype=torch.bfloat16), std=1
            ),
            persistent=True,
        )

        # Output heads
        self.lm_head = CastedLinear(self.hidden_size, self.num_colors, bias=False)
        self.q_head = CastedLinear(self.hidden_size, 2, bias=True)

        # Initialize Q head for faster learning
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Create input embeddings with token + puzzle + positional embeddings"""
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if hasattr(self, "puzzle_emb"):
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = (
                self.puzzle_emb_len * self.hidden_size - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        # Positional embeddings
        if hasattr(self, "embed_pos"):
            # Learned positional embeddings
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(torch.float32)
            )

        # Scale
        return self.embed_scale * embedding

    def empty_carry(
        self, batch_size: int, device: torch.device = None
    ) -> HRMInnerCarry:
        """Create empty carry state"""
        if device is None:
            device = next(self.parameters()).device
        return HRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.max_len + self.puzzle_emb_len,
                self.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.max_len + self.puzzle_emb_len,
                self.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            ),
        )

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: HRMInnerCarry
    ) -> HRMInnerCarry:
        """Reset carry state for halted sequences"""
        # Move initial states to the same device as the reset_flag
        H_init = self.H_init.to(reset_flag.device)
        L_init = self.L_init.to(reset_flag.device)
        return HRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
        )

    def forward(self, batch: FewShotBatch) -> Dict[str, torch.Tensor]:
        """Forward pass with few-shot learning and ACT"""
        B = batch.support_inp.shape[0]
        K = batch.support_inp.shape[1]  # should be 2
        L = batch.support_inp.shape[2]  # should be 900 (30*30)

        # Create puzzle identifiers (use task_id for now)
        puzzle_identifiers = batch.task_id

        # Encode support pairs to build context
        sup_inp_e = self._input_embeddings(
            batch.support_inp.view(B * K, L), puzzle_identifiers.repeat_interleave(K)
        )  # (B*K, L, hidden_size)
        sup_out_e = self._input_embeddings(
            batch.support_out.view(B * K, L), puzzle_identifiers.repeat_interleave(K)
        )  # (B*K, L, hidden_size)

        # Build support context (mean pool over sequence)
        sup_inp_pool = sup_inp_e.mean(dim=1)  # (B*K, hidden_size)
        sup_out_pool = sup_out_e.mean(dim=1)  # (B*K, hidden_size)

        # Reshape and combine support context
        sup_inp_pool = sup_inp_pool.view(B, K, -1)  # (B, K, hidden_size)
        sup_out_pool = sup_out_pool.view(B, K, -1)  # (B, K, hidden_size)
        support_context = torch.cat([sup_inp_pool, sup_out_pool], dim=-1).mean(
            dim=1
        )  # (B, 2*hidden_size)

        # Encode query input
        query_inp_e = self._input_embeddings(
            batch.query_inp, puzzle_identifiers
        )  # (B, L, hidden_size)

        # Project support context to hidden_size
        support_proj = CastedLinear(
            2 * self.hidden_size, self.hidden_size, bias=True
        ).to(query_inp_e.device)
        support_context_proj = support_proj(support_context)  # (B, hidden_size)

        # Initialize carry
        carry = self.empty_carry(B, device=query_inp_e.device)
        carry = self.reset_carry(
            torch.ones(B, dtype=torch.bool, device=query_inp_e.device), carry
        )

        # Prepare sequence info for attention
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Forward iterations (H/L schedule)
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.H_cycles):
                for _L_step in range(self.L_cycles):
                    if not (
                        (_H_step == self.H_cycles - 1)
                        and (_L_step == self.L_cycles - 1)
                    ):
                        # L level update
                        l_input = query_inp_e + support_context_proj.unsqueeze(1)
                        z_L = self.L_level(z_L, l_input, **seq_info)

                if not (_H_step == self.H_cycles - 1):
                    # H level update
                    h_input = z_L
                    z_H = self.H_level(z_H, h_input, **seq_info)

        # 1-step gradient update
        l_input = query_inp_e + support_context_proj.unsqueeze(1)
        z_L = self.L_level(z_L, l_input, **seq_info)
        h_input = z_L
        z_H = self.H_level(z_H, h_input, **seq_info)

        # Outputs
        logits = self.lm_head(z_H)[
            :, self.puzzle_emb_len :
        ]  # Remove puzzle embedding part
        q_logits = self.q_head(z_H[:, 0]).to(
            torch.float32
        )  # Use first token for Q-values

        return {
            "logits": logits,
            "q_halt_logits": q_logits[..., 0],
            "q_continue_logits": q_logits[..., 1],
        }


# ------------------------------------------------------------
# Loss function with ACT
# ------------------------------------------------------------
def hrm_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,  # (B, L) ints
    config: Dict = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """HRM loss with ACT Q-learning"""

    if config is None:
        config = {"act_weight": 0.1}

    # Cross-entropy loss
    logits = outputs["logits"]  # (B, L, num_colors)
    ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))

    # ACT Q-learning loss (simplified for now)
    q_halt = outputs["q_halt_logits"]  # (B,)
    q_continue = outputs["q_continue_logits"]  # (B,)

    # Simple Q-loss: encourage halt when task is solved
    pred = logits.argmax(-1)  # (B, L)
    task_solved = (pred == target).all(dim=1).float()  # (B,)

    # Target: halt if solved, continue if not
    target_halt = task_solved
    target_continue = 1.0 - task_solved

    q_loss = F.mse_loss(torch.sigmoid(q_halt), target_halt) + F.mse_loss(
        torch.sigmoid(q_continue), target_continue
    )

    # Total loss
    total_loss = ce_loss + config.get("act_weight", 0.1) * q_loss

    # Metrics
    with torch.no_grad():
        cell_acc = (pred == target).float().mean()
        perfect_acc = task_solved.mean()

    metrics = {
        "loss": float(total_loss.item()),
        "ce_loss": float(ce_loss.item()),
        "q_loss": float(q_loss.item()),
        "cell_acc": cell_acc,
        "perfect_acc": perfect_acc,
    }

    return total_loss, metrics
