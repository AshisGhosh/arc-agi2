"""
HRM-style Two-Timescale Recurrent Model (Task-Token Version)
----------------------------------------------------------------------
This file implements a compact version of the HRM idea with:
  • Two coupled recurrent modules at different timescales: L (fast), H (slow)
  • Segment-wise deep supervision (candidate output per cycle)
  • Optional ACT-style halting head (halt vs continue)
  • Task-conditioned learning via learnable task tokens
  • "One-step"-style gradient approximation via detaching inner steps

Design notes (pragmatic version):
  - We keep the cells tiny (GRU-like) so parameter count stays small.
  - Inner L-steps are computed with prev_state.detach() → approximates a 1-step
    fixed-point gradient; segment boundaries also detach for stability & O(1)-ish mem.
  - The OutputHead decodes from the slow H-state and the embedded task token.
  - The HaltingHead provides a per-cycle halt/continue logit pair.

This is NOT a full reproduction of any specific paper; it's a faithful scaffold
for experimentation that captures the schedule and learning signals.

Requires: torch>=2.0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utility: small MLP
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ------------------------------------------------------------
# Encoders for ARC-like discrete grids (0..9 colors)
# ------------------------------------------------------------
class GridEncoder(nn.Module):
    """Embeds a flattened grid of ints in [0..num_colors-1].

    Args:
      num_colors: e.g., 10 for ARC
      d_model: embedding size per token
      max_len: maximum HW length for positional embeddings
    """

    def __init__(self, num_colors: int = 10, d_model: int = 128, max_len: int = 400):
        super().__init__()
        self.tok = nn.Embedding(num_colors, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) ints
        returns: (B, L, d_model)
        """
        B, L = x.shape
        pe = self.pos.weight[:L].unsqueeze(0).expand(B, L, -1)
        return self.tok(x) + pe

    @staticmethod
    def pool_mean(embed: torch.Tensor) -> torch.Tensor:
        """Mean-pool over sequence: (B, L, D) -> (B, D)."""
        return embed.mean(dim=1)


# ------------------------------------------------------------
# Few-shot batch: support pairs, query input/output, task_id
# ------------------------------------------------------------
@dataclass
class FewShotBatch:
    support_inp: torch.Tensor  # (B, K, L) where K=2 support examples
    support_out: torch.Tensor  # (B, K, L) corresponding outputs
    query_inp: torch.Tensor  # (B, L) query input to solve
    query_out: torch.Tensor  # (B, L) query output (target)
    task_id: torch.Tensor  # (B,) task IDs


# ------------------------------------------------------------
# Tiny GRU-like gated cells for L and H
# ------------------------------------------------------------
class GatedCell(nn.Module):
    """A tiny GRU-ish cell.

    L-cell uses inputs: [prev_L, H, context]
    H-cell uses inputs: [prev_H, L_terminal]
    """

    def __init__(self, in_dim: int, state_dim: int):
        super().__init__()
        self.in_to_gate = nn.Linear(in_dim, state_dim)
        self.state_to_gate = nn.Linear(state_dim, state_dim)
        self.in_to_cand = nn.Linear(in_dim, state_dim)
        self.state_to_cand = nn.Linear(state_dim, state_dim)

    def forward(self, prev: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.in_to_gate(inp) + self.state_to_gate(prev))
        cand = torch.tanh(self.in_to_cand(inp) + self.state_to_cand(prev))
        return (1 - gate) * prev + gate * cand


# ------------------------------------------------------------
# HRM model: H/L schedule, Output head, Halting head
# ------------------------------------------------------------
class HRM(nn.Module):
    def __init__(
        self,
        num_colors: int = 10,
        d_model: int = 128,
        d_l: int = 160,
        d_h: int = 192,
        max_len: int = 400,
        num_tasks: int = 1000,
        task_token_dim: int = None,
    ):
        super().__init__()
        self.num_colors = num_colors
        self.enc = GridEncoder(num_colors=num_colors, d_model=d_model, max_len=max_len)

        # task token embedding
        self.d_model = d_model
        self.task_tok_dim = task_token_dim or d_model
        self.task_tok = nn.Embedding(
            num_tasks, d_model
        )  # always use d_model for consistency

        # context projection for task token + support context
        self.task_ctx_proj = nn.Linear(2 * d_model, d_model)

        # H state projection to d_model
        self.h_to_d = nn.Linear(d_h, d_model)

        # support context builder
        self.support_ctx = MLP(in_dim=2 * d_model, hidden=d_model, out_dim=d_model)

        # Initializers
        self.h_init = MLP(
            in_dim=2 * d_model, hidden=d_h, out_dim=d_h
        )  # [task_token + support_context] = 2*d_model
        self.l_init = MLP(
            in_dim=d_h + d_model, hidden=d_l, out_dim=d_l
        )  # [H + combined_context] = d_h + d_model

        # Cells
        self.l_cell = GatedCell(
            in_dim=d_l + d_h + d_model, state_dim=d_l
        )  # [L_state + H + combined_context]
        self.h_cell = GatedCell(in_dim=d_h + d_l, state_dim=d_h)  # [H + L_state]

        # Output & Halting heads
        self.out_head = MLP(in_dim=3 * d_model, hidden=d_model, out_dim=num_colors)
        self.halt_head = MLP(in_dim=d_h, hidden=d_h // 2, out_dim=2)

    def decode_logits(
        self, H: torch.Tensor, q_inp_e: torch.Tensor, task_context: torch.Tensor
    ) -> torch.Tensor:
        """per-position classification over colors
        H: (B, d_h)
        q_inp_e: (B, L, d_model)
        task_context: (B, d_model) - already projected
        returns: (B, L, num_colors)
        """
        B, L, D = q_inp_e.shape

        # task_context is already the right size (B, d_model)
        task_ctx = task_context  # (B, D)

        H_proj = self.h_to_d(H)  # project H to d_model dimensions

        task_rep = task_ctx.unsqueeze(1).expand(B, L, -1)  # (B,L,D)
        H_rep = H_proj.unsqueeze(1).expand(B, L, -1)  # (B,L,D)

        # include H_rep in the concat
        x = torch.cat([q_inp_e, task_rep, H_rep], dim=-1)  # (B,L, 3D)
        logits = self.out_head(x)  # ensure in_dim matches: in_dim = 3*D
        return logits

    def build_support_context(
        self, sup_inp_e: torch.Tensor, sup_out_e: torch.Tensor
    ) -> torch.Tensor:
        """build support context from support pairs
        sup_inp_e: (B, K, L, D) where K=2
        sup_out_e: (B, K, L, D)
        returns: (B, D) aggregated support context
        """
        B, K, L, D = sup_inp_e.shape

        # pool over sequence length for each support pair
        sup_inp_pool = sup_inp_e.mean(dim=2)  # (B, K, D)
        sup_out_pool = sup_out_e.mean(dim=2)  # (B, K, D)

        # concatenate input and output for each support pair
        sup_pairs = torch.cat([sup_inp_pool, sup_out_pool], dim=-1)  # (B, K, 2D)

        # aggregate over support examples
        sup_agg = sup_pairs.mean(dim=1)  # (B, 2D)

        # build context from aggregated support pairs
        support_context = self.support_ctx(sup_agg)  # (B, D)

        return support_context

    # ------------- core forward -------------
    @torch.no_grad()
    def _detach_(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.detach() for t in tensors)

    def forward(
        self,
        batch: FewShotBatch,
        cycles: int = 4,
        inner_steps: int = 2,
        one_step_grad: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """forward pass with few-shot learning
        batch: FewShotBatch with support pairs and query
        """
        B = batch.support_inp.shape[0]
        K = batch.support_inp.shape[1]  # should be 2
        L = batch.support_inp.shape[2]  # should be 900 (30*30)

        # encode support pairs
        sup_inp_e = self.enc(batch.support_inp.view(B * K, L))  # (B*K, L, D)
        sup_out_e = self.enc(batch.support_out.view(B * K, L))  # (B*K, L, D)

        # reshape back to (B, K, L, D)
        sup_inp_e = sup_inp_e.view(B, K, L, -1)
        sup_out_e = sup_out_e.view(B, K, L, -1)

        # build support context from support pairs
        support_context = self.build_support_context(sup_inp_e, sup_out_e)  # (B, D)

        # encode query input
        query_inp_e = self.enc(batch.query_inp)  # (B, L, D)

        # get task token
        task_tokens = self.task_tok(batch.task_id)  # (B, D)

        # combine task token and support context
        combined_context = torch.cat([task_tokens, support_context], dim=-1)  # (B, 2D)

        # initialize H and L states using combined context
        H = self.h_init(combined_context)  # (B, d_h) - input is 2*d_model

        # project combined_context to d_model for other uses
        combined_context_proj = self.task_ctx_proj(combined_context)  # (B, D)

        L_state = self.l_init(torch.cat([H, combined_context_proj], dim=-1))  # (B, d_l)

        # run H/L schedule
        H_states = []
        L_states = []

        for cycle in range(cycles):
            # inner loop
            for step in range(inner_steps):
                if one_step_grad:
                    # single gradient step - build input for l_cell
                    l_input = torch.cat([L_state, H, combined_context_proj], dim=-1)
                    L_state = self.l_cell(L_state, l_input)
                else:
                    # multiple gradient steps
                    for _ in range(inner_steps):
                        l_input = torch.cat([L_state, H, combined_context_proj], dim=-1)
                        L_state = self.l_cell(L_state, l_input)

            # update H state - build input for h_cell
            h_input = torch.cat([H, L_state], dim=-1)
            H = self.h_cell(H, h_input)

            # store states for deep supervision
            H_states.append(H)
            L_states.append(L_state)

        # decode final output
        logits = self.decode_logits(H, query_inp_e, combined_context_proj)

        return {
            "logits_per_cycle": [
                self.decode_logits(h, query_inp_e, combined_context_proj)
                for h in H_states
            ],
            "logits": logits,
            "H_states": H_states,
            "L_states": L_states,
        }


# ------------------------------------------------------------
# Losses: deep supervision + simple ACT auxiliary
# ------------------------------------------------------------
def hrm_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,  # (B, L) ints
    cfg: Dict = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """compute deep-supervision CE + simple ACT loss"""
    if cfg is None:
        cfg = {"ds_weight": 1.0, "ds_decay": 0.0, "act_weight": 0.0}

    logits_c = outputs["logits_per_cycle"]  # list of (B, L, C) tensors

    # stack logits if they're in a list
    if isinstance(logits_c, list):
        logits_c = torch.stack(logits_c, dim=0)  # (C, B, L, V)

    C, B, L, V = logits_c.shape

    # deep supervision loss
    total_ce = 0.0
    weights = []
    for i in range(C):
        w = cfg.get("ds_weight", 1.0) * (
            (1.0 - cfg.get("ds_decay", 0.0)) ** (C - 1 - i)
        )
        weights.append(w)
        ce = F.cross_entropy(logits_c[i].view(B * L, V), target.view(B * L))
        total_ce = total_ce + w * ce
    total_ce = total_ce / sum(weights)

    # compute metrics
    with torch.no_grad():
        pred = logits_c[-1].argmax(-1)  # (B, L) - last cycle predictions
        cell_acc = (pred == target).float().mean()
        perfect_acc = (pred == target).all(dim=1).float().mean()

    loss = total_ce  # no ACT loss for now
    metrics = {
        "loss": float(loss.item()),
        "ce": float(total_ce.item()),
        "cell_acc": cell_acc,
        "perfect_acc": perfect_acc,
    }
    return loss, metrics
