"""
HRM-style Two-Timescale Recurrent Model (Minimal, Low-Param, PyTorch)
----------------------------------------------------------------------
This file implements a compact version of the HRM idea with:
  • Two coupled recurrent modules at different timescales: L (fast), H (slow)
  • Segment-wise deep supervision (candidate output per cycle)
  • Optional ACT-style halting head (halt vs continue)
  • A simple few-shot packing for ARC-like tasks (support pairs + query)
  • "One-step"-style gradient approximation via detaching inner steps

Design notes (pragmatic version):
  - We keep the cells tiny (GRU-like) so parameter count stays small.
  - Inner L-steps are computed with prev_state.detach() → approximates a 1-step
    fixed-point gradient; segment boundaries also detach for stability & O(1)-ish mem.
  - The OutputHead decodes from the slow H-state and the embedded test input.
  - The HaltingHead provides a per-cycle halt/continue logit pair.

This is NOT a full reproduction of any specific paper; it's a faithful scaffold
for experimentation that captures the schedule and learning signals.

Requires: torch>=2.0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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

    def __init__(self, num_colors: int = 10, d_model: int = 128, max_len: int = 900):
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
# Few-shot pack: support examples and a query
# ------------------------------------------------------------
@dataclass
class FewShotBatch:
    support_inp: torch.Tensor  # (B, K, L)
    support_out: torch.Tensor  # (B, K, L)
    query_inp: torch.Tensor  # (B, L)
    query_out: Optional[torch.Tensor] = None  # (B, L), only for training


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
        max_len: int = 900,
    ):
        super().__init__()
        self.num_colors = num_colors
        self.enc = GridEncoder(num_colors=num_colors, d_model=d_model, max_len=max_len)

        # Context builders
        self.ctx_pair = MLP(in_dim=2 * d_model, hidden=d_model, out_dim=d_model)
        self.ctx_agg = MLP(in_dim=d_model, hidden=d_model, out_dim=d_model)

        # Initializers
        self.task_token = nn.Parameter(torch.randn(1, d_model))
        self.h_init = MLP(
            in_dim=d_model + d_model, hidden=d_h, out_dim=d_h
        )  # [task, agg_support]
        self.l_init = MLP(
            in_dim=d_h + d_model + d_model, hidden=d_l, out_dim=d_l
        )  # [H, agg_support, query_pool]

        # Cells
        self.l_cell = GatedCell(in_dim=d_l + d_h + d_model, state_dim=d_l)
        self.h_cell = GatedCell(in_dim=d_h + d_l, state_dim=d_h)

        # Output & Halting heads
        self.out_head = MLP(
            in_dim=d_model + d_h + d_model, hidden=d_model, out_dim=num_colors
        )
        self.halt_head = MLP(in_dim=d_h, hidden=d_h // 2, out_dim=2)

    # ------------- helpers -------------
    def build_support_context(
        self, sup_inp_e: torch.Tensor, sup_out_e: torch.Tensor
    ) -> torch.Tensor:
        """Compute an aggregated support context from K pairs.
        sup_inp_e: (B, K, L, D)
        sup_out_e: (B, K, L, D)
        returns: (B, D)
        """
        B, K, L, D = sup_inp_e.shape
        # Per-pair difference summary
        diff = (sup_out_e - sup_inp_e).mean(dim=2)  # (B, K, D)
        pair_ctx = self.ctx_pair(
            torch.cat([diff, diff], dim=-1)
        )  # simple transform (B, K, D)
        agg = pair_ctx.mean(dim=1)  # (B, D)
        return self.ctx_agg(agg)  # (B, D)

    def decode_logits(
        self, H: torch.Tensor, q_inp_e: torch.Tensor, sup_ctx: torch.Tensor
    ) -> torch.Tensor:
        """Per-position classification over colors.
        H: (B, d_h)
        q_inp_e: (B, L, d_model)
        sup_ctx: (B, d_model)
        returns: (B, L, num_colors)
        """
        B, L, D = q_inp_e.shape
        H_rep = H.unsqueeze(1).expand(B, L, -1)
        C_rep = sup_ctx.unsqueeze(1).expand(B, L, -1)
        x = torch.cat([q_inp_e, H_rep, C_rep], dim=-1)
        logits = self.out_head(x)
        return logits

    # ------------- core forward -------------
    @torch.no_grad()
    def _detach_(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.detach() for t in tensors)

    def forward(
        self,
        batch: FewShotBatch,
        *,
        cycles: int = 4,
        inner_steps: int = 3,
        one_step_grad: bool = True,
        act_infer: bool = False,
        act_min: int = 1,
        act_max: Optional[int] = None,
        act_thresh: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Run the HRM forward pass.

        Returns dict with:
          - logits_per_cycle: List[(B, L, num_colors)] stacked as (cycles, B, L, C)
          - q_per_cycle:      (cycles, B, 2) halting logits
          - H_last:           (B, d_h)
        """
        B, L = batch.query_inp.shape
        K = batch.support_inp.shape[1]

        # Encode support and query
        sup_inp_e = self.enc(batch.support_inp.view(B * K, L)).view(B, K, L, -1)
        sup_out_e = self.enc(batch.support_out.view(B * K, L)).view(B, K, L, -1)
        q_inp_e = self.enc(batch.query_inp)  # (B, L, D)

        # Aggregate contexts
        sup_ctx = self.build_support_context(sup_inp_e, sup_out_e)  # (B, D)
        q_pool = GridEncoder.pool_mean(q_inp_e)  # (B, D)

        # Initialize H from task token and support summary
        task_tok = self.task_token.expand(B, -1)
        H = self.h_init(torch.cat([task_tok, sup_ctx], dim=-1))  # (B, d_h)

        logits_list: List[torch.Tensor] = []
        q_list: List[torch.Tensor] = []

        # ACT settings
        max_cycles = act_max if (act_infer and act_max is not None) else cycles

        for n in range(max_cycles):
            # Reset/seed L from updated H and contexts
            L_state = self.l_init(torch.cat([H, sup_ctx, q_pool], dim=-1))  # (B, d_l)

            # Inner fast steps: L updates with H frozen
            for t in range(inner_steps):
                prev = L_state.detach() if one_step_grad else L_state
                # Context feature for L step = mean over query positions
                ctx = q_pool  # (B, D)
                l_in = torch.cat([prev, H, ctx], dim=-1)
                L_state = self.l_cell(prev, l_in)  # (B, d_l)

            # Slow update of H using terminal L
            h_in = torch.cat(
                [H, L_state.detach() if one_step_grad else L_state], dim=-1
            )
            H = self.h_cell(H, h_in)  # (B, d_h)

            # Candidate output + halting logits for this cycle
            logits = self.decode_logits(H, q_inp_e, sup_ctx)  # (B, L, C)
            q_logits = self.halt_head(H)  # (B, 2)

            logits_list.append(logits)
            q_list.append(q_logits)

            # Optional online halting at inference
            if act_infer and (n + 1) >= act_min:
                prob_halt = torch.softmax(q_logits, dim=-1)[..., 1]  # P(halt)
                if (prob_halt > act_thresh).all():
                    break

            # Detach between cycles (deep supervision segments)
            H = H.detach()

        return {
            "logits_per_cycle": torch.stack(
                logits_list, dim=0
            ),  # (C, B, L, num_colors)
            "q_per_cycle": torch.stack(q_list, dim=0),  # (C, B, 2)
            "H_last": H,
        }


# ------------------------------------------------------------
# Losses: deep supervision + simple ACT auxiliary
# ------------------------------------------------------------
@dataclass
class HRMLossCfg:
    ds_weight: float = 1.0  # weight per cycle (uniform)
    ds_decay: float = 0.0  # optional geometric decay across cycles
    act_weight: float = 0.1  # BCE on halting signal


def hrm_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,  # (B, L) ints
    cfg: HRMLossCfg = HRMLossCfg(),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute deep-supervision CE + simple ACT loss.

    ACT target heuristic: a cycle is "correct" if its argmax == target on all positions.
    Then we set y_halt=1 for that cycle and 0 otherwise. You can replace this with a
    proper return-based target if you implement an episodic signal.
    """
    logits_c = outputs["logits_per_cycle"]  # (C, B, L, C)
    q_c = outputs["q_per_cycle"]  # (C, B, 2)
    C, B, L, V = logits_c.shape

    total_ce = 0.0
    weights = []
    for i in range(C):
        w = cfg.ds_weight * ((1.0 - cfg.ds_decay) ** (C - 1 - i))
        weights.append(w)
        ce = F.cross_entropy(logits_c[i].view(B * L, V), target.view(B * L))
        total_ce = total_ce + w * ce
    total_ce = total_ce / sum(weights)

    # ACT target: 1 if perfect match, else 0
    with torch.no_grad():
        correct_mask = logits_c.argmax(-1) == target.unsqueeze(0)  # (C, B, L)
        perfect = correct_mask.all(dim=-1).float()  # (C, B)
    q_logits = q_c.view(C * B, 2)
    y_halt = perfect.view(C * B)
    act_ce = F.binary_cross_entropy_with_logits(q_logits[:, 1], y_halt)

    loss = total_ce + cfg.act_weight * act_ce
    metrics = {
        "loss": float(loss.item()),
        "ce": float(total_ce.item()),
        "act_bce": float(act_ce.item()),
        "perfect@last": float(perfect[-1].float().mean().item()),
    }
    return loss, metrics


# ------------------------------------------------------------
# Example training step
# ------------------------------------------------------------
class HRMSystem:
    def __init__(self, model: HRM, lr: float = 2e-3, weight_decay: float = 1e-4):
        self.model = model
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_step(
        self, batch: FewShotBatch, cfg: HRMLossCfg, *, cycles=4, inner_steps=3
    ) -> Dict[str, float]:
        self.model.train()
        out = self.model(
            batch, cycles=cycles, inner_steps=inner_steps, one_step_grad=True
        )
        assert batch.query_out is not None, "query_out required for training"
        loss, metrics = hrm_loss(out, batch.query_out, cfg)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return metrics

    @torch.inference_mode()
    def infer(
        self, batch: FewShotBatch, *, act_min=1, act_max=6, act_thresh=0.7
    ) -> torch.Tensor:
        self.model.eval()
        out = self.model(
            batch,
            act_infer=True,
            act_min=act_min,
            act_max=act_max,
            act_thresh=act_thresh,
        )
        # Use last cycle (or halted) prediction
        logits = out["logits_per_cycle"][-1]  # (B, L, V)
        return logits.argmax(-1)


# ------------------------------------------------------------
# Minimal demo with synthetic data (shapes only)
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a toy batch: recolor 1→2 task on 6×6 grids
    B, K, H, W, V = 4, 2, 6, 6, 10
    L = H * W

    def make_recolor_batch(B: int, K: int, H: int, W: int) -> FewShotBatch:
        # Inputs: random zeros with some 1s; outputs: recolor 1→2
        x_sup = torch.zeros(B, K, H, W, dtype=torch.long)
        y_sup = torch.zeros(B, K, H, W, dtype=torch.long)
        x_q = torch.zeros(B, H, W, dtype=torch.long)
        y_q = torch.zeros(B, H, W, dtype=torch.long)
        for b in range(B):
            for k in range(K):
                m = torch.rand(H, W) > 0.7
                x = torch.zeros(H, W, dtype=torch.long)
                x[m] = 1
                y = x.clone()
                y[y == 1] = 2
                x_sup[b, k] = x
                y_sup[b, k] = y
            # query
            m = torch.rand(H, W) > 0.7
            x = torch.zeros(H, W, dtype=torch.long)
            x[m] = 1
            y = x.clone()
            y[y == 1] = 2
            x_q[b] = x
            y_q[b] = y
        return FewShotBatch(
            support_inp=x_sup.view(B, K, L),
            support_out=y_sup.view(B, K, L),
            query_inp=x_q.view(B, L),
            query_out=y_q.view(B, L),
        )

    batch = make_recolor_batch(B, K, H, W)

    model = HRM(num_colors=V, d_model=64, d_l=80, d_h=96, max_len=L).to(device)
    sys = HRMSystem(model, lr=3e-3)

    # Move to device
    batch = FewShotBatch(
        support_inp=batch.support_inp.to(device),
        support_out=batch.support_out.to(device),
        query_inp=batch.query_inp.to(device),
        query_out=batch.query_out.to(device),
    )

    # Quick training loop
    cfg = HRMLossCfg(ds_weight=1.0, ds_decay=0.0, act_weight=0.05)
    for step in range(200):
        metrics = sys.train_step(batch, cfg, cycles=3, inner_steps=2)
        if (step + 1) % 50 == 0:
            print(
                f"step {step + 1}: loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} act={metrics['act_bce']:.4f} perf@last={metrics['perfect@last']:.3f}"
            )

    # Inference (ACT enabled)
    with torch.no_grad():
        pred = sys.infer(
            FewShotBatch(
                support_inp=batch.support_inp,
                support_out=batch.support_out,
                query_inp=batch.query_inp,
                query_out=None,
            ),
            act_min=1,
            act_max=5,
            act_thresh=0.6,
        )
        print("Pred correct rate:", (pred == batch.query_out).float().mean().item())
