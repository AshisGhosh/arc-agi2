"""
HRM-style Hierarchical Reasoning Model with ACT
Faithful recreation of original HRM with proper ACT mechanism
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
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
class HRMBatch:
    """Standard HRM batch with support examples"""

    inputs: torch.Tensor  # (B, L) input sequence
    labels: torch.Tensor  # (B, L) target sequence
    puzzle_identifiers: torch.Tensor  # (B,) puzzle/task IDs
    support_pairs: List[List[Dict[str, torch.Tensor]]]  # (B, num_support, {"inp": tensor, "out": tensor})


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
# Support mapping components
# ------------------------------------------------------------
class SupportPairMapper(nn.Module):
    """Learn input→output mapping per support pair with cross-attention"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Use regular nn.Linear with controlled initialization
        self.q = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        
        # Initialize attention weights with much smaller std for stability
        for layer in [self.q, self.k, self.v, self.o]:
            nn.init.normal_(layer.weight, std=0.001)  # Much smaller std for stability
        
        # lightweight attention-pooling to get a robust latent
        self.latent_query = nn.Parameter(torch.zeros(hidden_size, dtype=torch.bfloat16))
        nn.init.normal_(self.latent_query, std=0.001)  # Much smaller std for stability
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)

    def forward(self, x_in: torch.Tensor, y_out: torch.Tensor, cos_sin=None):
        # x_in, y_out: (L, H)
        B = 1; L, H = x_in.size()
        
        # Ensure inputs have correct dtype
        x_in = x_in.to(self.latent_query.dtype)
        y_out = y_out.to(self.latent_query.dtype)
        
        # Check for NaN in inputs
        if torch.isnan(x_in).any() or torch.isnan(y_out).any():
            print("Warning: NaN detected in SupportPairMapper inputs")
            return torch.zeros(self.latent_query.size(0), dtype=self.latent_query.dtype, device=x_in.device)
        
        # Check for extreme values in inputs
        if torch.abs(x_in).max() > 100 or torch.abs(y_out).max() > 100:
            print(f"Warning: Extreme values in inputs - x_in max: {torch.abs(x_in).max()}, y_out max: {torch.abs(y_out).max()}")
            # Clamp extreme values
            x_in = torch.clamp(x_in, min=-100, max=100)
            y_out = torch.clamp(y_out, min=-100, max=100)
        
        Q = self.q(x_in).view(B, L, self.num_heads, self.head_dim).transpose(1,2)   # (1,Nh,L,D)
        K = self.k(y_out).view(B, L, self.num_heads, self.head_dim).transpose(1,2)  # (1,Nh,L,D)
        V = self.v(y_out).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        # (optional) apply RoPE within each sequence only (slice to L) if cos_sin passed

        # Check for NaN in Q, K, V projections
        if torch.isnan(Q).any() or torch.isnan(K).any() or torch.isnan(V).any():
            print("Warning: NaN detected in Q, K, V projections")
            Q = torch.zeros_like(Q)
            K = torch.zeros_like(K)
            V = torch.zeros_like(V)

        # Ensure all tensors are bfloat16 for attention computation
        Q = Q.to(torch.bfloat16)
        K = K.to(torch.bfloat16)
        V = V.to(torch.bfloat16)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)      # (1,Nh,L,L)
        # Clamp scores to prevent overflow
        scores = torch.clamp(scores, min=-50, max=50)
        A = torch.softmax(scores, dim=-1)
        Z = torch.matmul(A, V)                                                      # (1,Nh,L,D)
        Z = Z.transpose(1,2).contiguous().view(B, L, H)                             # (1,L,H)
        Z = self.o(Z)                                                               # (1,L,H)
        Z = Z.to(torch.bfloat16)  # Ensure output is bfloat16
        
        # Check for NaN after attention computation
        if torch.isnan(Z).any():
            print("Warning: NaN detected after attention computation in SupportPairMapper")
            Z = torch.zeros_like(Z)

        # attention pooling with a learned query for stability over mean-only
        q = self.latent_query.view(1,1,H)                                           # (1,1,H)
        k = Z                                                                        # (1,L,H)
        # Ensure q and k are bfloat16
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (H**0.5)
        # Clamp attention scores to prevent overflow
        attn_scores = torch.clamp(attn_scores, min=-50, max=50)
        attn = torch.softmax(attn_scores, dim=-1) # (1,1,L)
        m = torch.matmul(attn, Z).squeeze(0).squeeze(0)                              # (H,)
        
        # Ensure correct dtype before normalization
        m = m.to(torch.bfloat16)
        m = self.norm(m)                                                             # (H,)
        
        # Check for NaN in pair latent
        if torch.isnan(m).any():
            print("Warning: NaN detected in pair latent, using zeros")
            m = torch.zeros_like(m)
        
        return m


class PairLatentFusion(nn.Module):
    """Fuse multiple pair latents with attention"""
    
    def __init__(self, hidden_size: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Use regular nn.Linear with controlled initialization
        self.q = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        
        # Initialize attention weights with much smaller std for stability
        for layer in [self.q, self.k, self.v, self.o]:
            nn.init.normal_(layer.weight, std=0.001)  # Much smaller std for stability
        
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)

    def forward(self, M: torch.Tensor):  # (N, H), N = num_support (≤2)
        if M.numel() == 0:
            return torch.zeros(M.size(-1), dtype=torch.bfloat16, device=M.device)
        
        # Ensure input has correct dtype
        M = M.to(torch.bfloat16)
        M = M.unsqueeze(0)                                                    # (1,N,H)
        Q = self.q(M).view(1, M.size(1), self.num_heads, self.head_dim).transpose(1,2)
        K = self.k(M).view(1, M.size(1), self.num_heads, self.head_dim).transpose(1,2)
        V = self.v(M).view(1, M.size(1), self.num_heads, self.head_dim).transpose(1,2)
        
        # Ensure all tensors are bfloat16 for attention computation
        Q = Q.to(torch.bfloat16)
        K = K.to(torch.bfloat16)
        V = V.to(torch.bfloat16)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)      # (1,Nh,N,N)
        # Clamp scores to prevent overflow
        scores = torch.clamp(scores, min=-50, max=50)
        W = torch.softmax(scores, dim=-1)
        Z = torch.matmul(W, V).transpose(1,2).contiguous().view(1, M.size(1), -1)    # (1,N,H)
        m_star = self.o(Z).mean(dim=1).squeeze(0)                                    # (H,)
        
        # Ensure correct dtype before normalization
        m_star = m_star.to(torch.bfloat16)
        m_star = self.norm(m_star)
        
        # Check for NaN in fused latent
        if torch.isnan(m_star).any():
            print("Warning: NaN detected in fused latent, using zeros")
            m_star = torch.zeros_like(m_star)
        
        return m_star


class GlobalSupportSummarizer(nn.Module):
    """Global support summarizer that captures rule-level patterns"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Use regular nn.Linear with controlled initialization
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.bfloat16)
        
        # Initialize projection weights with much smaller std for stability
        nn.init.normal_(self.proj.weight, std=0.001)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)

    def forward_pair(self, x_in: torch.Tensor, y_out: torch.Tensor):
        # x_in, y_out: (L,H)
        # Ensure inputs have correct dtype
        x_in = x_in.to(torch.bfloat16)
        y_out = y_out.to(torch.bfloat16)
        
        # simple robust [mean || max] summary
        mean_xy = (x_in.mean(dim=0) + y_out.mean(dim=0)) * 0.5
        max_xy  = torch.maximum(x_in.max(dim=0).values, y_out.max(dim=0).values)
        g = self.proj(0.5 * (mean_xy + max_xy))
        
        # Ensure correct dtype before normalization
        g = g.to(torch.bfloat16)
        return self.norm(g)  # (H,)

    def forward_batch(self, pairs: list):
        # pairs: list of (x_in, y_out) for one batch item
        if not pairs:
            return None
        G = torch.stack([self.forward_pair(x,y) for (x,y) in pairs], dim=0)  # (N,H)
        g_ctx = G.mean(dim=0)                                                 # (H,)
        
        # Check for NaN in global context
        if torch.isnan(g_ctx).any():
            print("Warning: NaN detected in global context, using zeros")
            g_ctx = torch.zeros_like(g_ctx)
        
        return g_ctx


class FiLMInjector(nn.Module):
    """Feature-wise Linear Modulation injector"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Use regular nn.Linear with controlled initialization
        self.gamma = nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.bfloat16)
        self.beta  = nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.bfloat16)
        
        # Initialize FiLM weights with much smaller std for stability
        for layer in [self.gamma, self.beta]:
            nn.init.normal_(layer.weight, std=0.001)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        self.norm  = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: (B, L', H), cond: (B, H)
        # Ensure cond has correct dtype before normalization
        cond = cond.to(x.dtype)
        cond = self.norm(cond)
        g = self.gamma(cond).unsqueeze(1)
        b = self.beta(cond).unsqueeze(1)
        return x * (1 + g) + b


# ------------------------------------------------------------
# Main HRM model
# ------------------------------------------------------------
class HRMInner(nn.Module):
    """Inner HRM model (matches original HierarchicalReasoningModel_ACTV1_Inner)"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.forward_dtype = torch.bfloat16

        # Model dimensions
        self.hidden_size = config["hidden_size"]
        self.num_heads = config.get("num_heads", 8)
        self.vocab_size = config.get("vocab_size", 10)
        self.seq_len = config.get("seq_len", 900)
        self.num_puzzle_identifiers = config.get("num_puzzle_identifiers", 1000)

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

        # Token embeddings - use regular nn.Embedding with controlled initialization
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            dtype=self.forward_dtype,
        )
        
        # Initialize token embeddings with much smaller std for stability
        with torch.no_grad():
            # Use a more robust initialization method
            # First create in float32, then convert to target dtype
            new_weights = torch.randn(
                self.embed_tokens.weight.shape, 
                dtype=torch.float32, 
                device=self.embed_tokens.weight.device
            ) * 0.001
            new_weights = new_weights.to(self.forward_dtype)
            self.embed_tokens.weight.data = new_weights
            
        assert not torch.isnan(self.embed_tokens.weight).any(), "embed_tokens weights are NaN after initialization!"

        # Puzzle embeddings (task-specific embeddings)
        puzzle_emb_ndim = config.get("puzzle_emb_ndim", 0)
        self.puzzle_emb_len = (
            -(puzzle_emb_ndim // -self.hidden_size) if puzzle_emb_ndim > 0 else 0
        )
        if puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.num_puzzle_identifiers,
                puzzle_emb_ndim,
                batch_size=config.get("batch_size", 32),
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # Positional embeddings
        pos_encodings = config.get("pos_encodings", "rope")
        if pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.hidden_size // config["num_heads"],
                max_position_embeddings=self.seq_len + self.puzzle_emb_len,
                base=config.get("rope_theta", 10000.0),
            )
        elif pos_encodings == "learned":
            self.embed_pos = nn.Embedding(
                self.seq_len + self.puzzle_emb_len,
                self.hidden_size,
                dtype=self.forward_dtype,
            )
            # Initialize positional embeddings with much smaller std for stability
            nn.init.normal_(self.embed_pos.weight, std=0.001)
        else:
            raise ValueError(f"Unknown pos_encodings: {pos_encodings}")

        # Reasoning modules
        self.H_level = HRMReasoningModule(config, self.H_layers)
        self.L_level = HRMReasoningModule(config, self.L_layers)

        # Initial states (uninitialized, must be reset/warmed before use)
        H_init = trunc_normal_init_(
            torch.empty(self.hidden_size, dtype=self.forward_dtype), std=1
        )
        L_init = trunc_normal_init_(
            torch.empty(self.hidden_size, dtype=self.forward_dtype), std=1
        )
        self.register_buffer("H_init", H_init, persistent=True)
        self.register_buffer("L_init", L_init, persistent=True)

        # Support example encoder (permutation invariant) - use regular Linear with proper dtype
        self.support_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )
        
        # Initialize support encoder weights with much smaller std for stability
        for module in self.support_encoder:
            if isinstance(module, nn.Linear):
                # Initialize weights in the correct dtype using robust method
                with torch.no_grad():
                    # Create in float32 first, then convert
                    new_weight = torch.randn(
                        module.weight.shape, 
                        dtype=torch.float32, 
                        device=module.weight.device
                    ) * 0.001
                    module.weight.data = new_weight.to(self.forward_dtype)
                    
                    if module.bias is not None:
                        module.bias.data = torch.zeros_like(module.bias.data, dtype=self.forward_dtype)
                    
                    # Assert no NaN in support encoder MLP weights
                    assert not torch.isnan(module.weight).any(), f"Support encoder MLP layer weight became NaN during initialization!"
                    if module.bias is not None:
                        assert not torch.isnan(module.bias).any(), f"Support encoder MLP layer bias became NaN during initialization!"
        
        # Test the embedding layer with random tokens
        test_tokens = torch.randint(0, min(10, self.vocab_size), (10,), dtype=torch.int32)
        try:
            test_emb = self.embed_tokens(test_tokens)
            assert not torch.isnan(test_emb).any(), "embed_tokens produces NaN even with random tokens!"
        except Exception as e:
            print(f"ERROR: Embedding test failed: {e}")
            raise
        
        assert not torch.isnan(self.embed_tokens.weight).any(), "embed_tokens weights became NaN after test!"
        
        # Test the MLP with random input to catch initialization issues
        test_input = torch.randn(10, self.hidden_size, dtype=self.forward_dtype)
        try:
            test_output = self.support_encoder(test_input)
            assert not torch.isnan(test_output).any(), "MLP produces NaN even with random input!"
        except Exception as e:
            print(f"ERROR: MLP test failed: {e}")
            raise
        
        
        # Config switches for dual-path conditioning
        self.enable_local_mapping  = config.get("enable_local_mapping", True)
        self.enable_global_summary = config.get("enable_global_summary", True)
        self.inject_mode           = config.get("inject_mode", "film")  # "film" | "add" | "prefix"
        
        # Dual-path support conditioning modules
        self.pair_mapper   = SupportPairMapper(self.hidden_size, self.num_heads)
        self.pair_fuser    = PairLatentFusion(self.hidden_size, num_heads=1)
        self.global_summarizer = GlobalSupportSummarizer(self.hidden_size)
        self.injector      = FiLMInjector(self.hidden_size)
        self.cond_fuse_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )
        
        # Initialize cond_fuse_mlp weights with much smaller std for stability
        for module in self.cond_fuse_mlp:
            if isinstance(module, nn.Linear):
                # Use robust initialization method
                with torch.no_grad():
                    new_weight = torch.randn(
                        module.weight.shape, 
                        dtype=torch.float32, 
                        device=module.weight.device
                    ) * 0.001
                    module.weight.data = new_weight.to(self.forward_dtype)
                    
                    if module.bias is not None:
                        module.bias.data = torch.zeros_like(module.bias.data, dtype=self.forward_dtype)
                    
                    # Assert no NaN in cond_fuse_mlp weights
                    assert not torch.isnan(module.weight).any(), f"cond_fuse_mlp layer weight became NaN during initialization!"
                    if module.bias is not None:
                        assert not torch.isnan(module.bias).any(), f"cond_fuse_mlp layer bias became NaN during initialization!"

        # Output heads
        self.lm_head = CastedLinear(self.hidden_size, self.vocab_size, bias=False)
        self.q_head = CastedLinear(self.hidden_size, 2, bias=True)
        
        # Check lm_head weights after initialization
        assert not torch.isnan(self.lm_head.weight).any(), "lm_head weights are NaN after initialization!"

        # Initialize Q head for faster learning
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
        
        # Final check: Assert no NaN weights anywhere in the model
        for name, param in self.named_parameters():
            assert not torch.isnan(param).any(), f"NaN found in parameter {name} at end of initialization!"

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Create input embeddings"""
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if hasattr(self, "puzzle_emb") and self.puzzle_emb_len > 0:
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
                embedding + self.embed_pos.weight.to(self.forward_dtype)
            )

        # Scale
        return self.embed_scale * embedding


    def _create_support_mapping(self, support_pairs_list: List[List[Dict[str, torch.Tensor]]], puzzle_identifiers: torch.Tensor) -> List[List[Dict[str, torch.Tensor]]]:
        """Create encoded support pairs for dual-path conditioning"""
        batch_size = puzzle_identifiers.shape[0]
        device = puzzle_identifiers.device
        
        all_encoded_pairs = []
        
        for i in range(batch_size):
            support_pairs = support_pairs_list[i]
            batch_encoded_pairs = []
            
            for support_pair in support_pairs[:2]:  # limit to 2 examples
                support_input = support_pair["inp"].to(device)
                support_output = support_pair["out"].to(device)
                
                # Create embeddings
                if hasattr(self.embed_tokens, 'weight'):
                    assert not torch.isnan(self.embed_tokens.weight).any(), "embed_tokens weights are NaN during forward pass!"
                
                input_emb = self.embed_tokens(support_input.to(torch.int32))
                output_emb = self.embed_tokens(support_output.to(torch.int32))
                
                # Ensure embeddings are in correct dtype and not NaN
                input_emb = input_emb.to(self.forward_dtype)
                output_emb = output_emb.to(self.forward_dtype)
                
                # Check for NaN in raw embeddings
                assert not (torch.isnan(input_emb).any() or torch.isnan(output_emb).any()), "NaN detected in raw support embeddings during forward pass!"
                
                # Check for extreme values in raw embeddings
                if torch.abs(input_emb).max() > 100 or torch.abs(output_emb).max() > 100:
                    print(f"Warning: Extreme values in raw embeddings - input max: {torch.abs(input_emb).max()}, output max: {torch.abs(output_emb).max()}")
                    input_emb = torch.clamp(input_emb, min=-100, max=100)
                    output_emb = torch.clamp(output_emb, min=-100, max=100)
                
                # Encode with support encoder
                try:
                    # Check MLP weights before forward pass
                    for i, layer in enumerate(self.support_encoder):
                        if hasattr(layer, 'weight'):
                            assert not torch.isnan(layer.weight).any(), f"NaN in MLP layer {i} weights during forward pass!"
                    
                    # Forward pass
                    x = input_emb
                    for i, layer in enumerate(self.support_encoder):
                        x = layer(x)
                        # Ensure output is in correct dtype
                        x = x.to(self.forward_dtype)
                        assert not torch.isnan(x).any(), f"NaN produced by MLP layer {i} during forward pass!"
                    
                    encoded_input = x
                    encoded_output = self.support_encoder(output_emb).to(self.forward_dtype)
                    
                    # Clamp gradients to prevent explosion
                    if encoded_input.requires_grad:
                        encoded_input = torch.clamp(encoded_input, min=-10, max=10)
                    if encoded_output.requires_grad:
                        encoded_output = torch.clamp(encoded_output, min=-10, max=10)
                        
                except Exception as e:
                    print(f"ERROR: Support encoder failed: {e}")
                    import traceback
                    traceback.print_exc()
                    encoded_input = torch.zeros_like(input_emb)
                    encoded_output = torch.zeros_like(output_emb)
                
                # Check for NaN after support encoder
                assert not (torch.isnan(encoded_input).any() or torch.isnan(encoded_output).any()), "NaN detected after support_encoder during forward pass!"
                
                # Check for extreme values after support encoder
                if torch.abs(encoded_input).max() > 100 or torch.abs(encoded_output).max() > 100:
                    print(f"Warning: Extreme values after support_encoder - input max: {torch.abs(encoded_input).max()}, output max: {torch.abs(encoded_output).max()}")
                    encoded_input = torch.clamp(encoded_input, min=-100, max=100)
                    encoded_output = torch.clamp(encoded_output, min=-100, max=100)
                
                batch_encoded_pairs.append({
                    "input": encoded_input,    # (L, H)
                    "output": encoded_output   # (L, H)
                })
            
            all_encoded_pairs.append(batch_encoded_pairs)
        
        return all_encoded_pairs

    def _apply_dual_path_conditioning(self, input_embeddings: torch.Tensor, encoded_support_pairs: List[List[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        """Apply dual-path conditioning: local mapping + global summary → fused conditioning"""
        batch_size, full_seq_len, hidden_size = input_embeddings.shape
        P = self.puzzle_emb_len
        
        # Split into puzzle prefix and main sequence (preserve original order)
        puzzle_part = input_embeddings[:, :P, :]      # (batch_size, P, hidden_size)
        main_sequence = input_embeddings[:, P:, :]    # (batch_size, L, hidden_size)
        
        mapped_embeddings = []
        
        for i in range(batch_size):
            support_pairs = encoded_support_pairs[i]
            
            if not support_pairs:
                # No support examples - use original sequence
                mapped_embeddings.append(main_sequence[i])
                continue
            
            # Build per-pair tensors for this batch item
            m_list = []
            g_pairs = []
            for sp in support_pairs:
                x = sp["input"]    # (L,H) already encoded by support_encoder
                y = sp["output"]   # (L,H)
                if self.enable_local_mapping:
                    m_list.append(self.pair_mapper(x, y, cos_sin=None))  # (H,)
                if self.enable_global_summary:
                    g_pairs.append((x, y))

            m_star = None
            if self.enable_local_mapping and m_list:
                M = torch.stack(m_list, dim=0)                  # (N,H)
                m_star = self.pair_fuser(M)                     # (H,)

            g_ctx = None
            if self.enable_global_summary and g_pairs:
                g_ctx = self.global_summarizer.forward_batch(g_pairs)  # (H,)

            # Fuse condition vectors (handle None cases)
            if m_star is None and g_ctx is None:
                cond = torch.zeros(self.hidden_size, dtype=main_sequence.dtype, device=main_sequence.device)
            elif m_star is None:
                cond = g_ctx
            elif g_ctx is None:
                cond = m_star
            else:
                # Ensure both latents have same dtype before concatenation
                m_star = m_star.to(main_sequence.dtype)
                g_ctx = g_ctx.to(main_sequence.dtype)
                cond = self.cond_fuse_mlp(torch.cat([m_star, g_ctx], dim=-1))  # (H,)

            # Inject into main_sequence tokens (B=1 in this loop)
            if self.inject_mode == "film":
                # Check for NaN in conditioning vector
                if torch.isnan(cond).any():
                    print(f"Warning: NaN detected in conditioning vector at batch {i}")
                    cond = torch.zeros_like(cond)
                main_seq_i = self.injector(main_sequence[i].unsqueeze(0), cond.unsqueeze(0)).squeeze(0)
            elif self.inject_mode == "add":
                bias = cond.unsqueeze(0).expand_as(main_sequence[i])
                main_seq_i = main_sequence[i] + bias
            else:
                main_seq_i = main_sequence[i]  # (prefix mode would need extra handling)

            mapped_embeddings.append(main_seq_i)
        
        mapped_main = torch.stack(mapped_embeddings, dim=0)
        
        # Concatenate back with puzzle prefix to preserve original order [P || L]
        return torch.cat([puzzle_part, mapped_main], dim=1)

    def _reproduce_support_examples(self, support_pairs_list: List[List[Dict[str, torch.Tensor]]], puzzle_identifiers: torch.Tensor, seq_info: Dict) -> Dict[str, torch.Tensor]:
        batch_size = puzzle_identifiers.shape[0]
        device = puzzle_identifiers.device
        
        all_support_inputs = []
        all_support_outputs = []
        all_support_logits = []
        
        # temporarily switch to eval mode to avoid batch size issues with puzzle_emb
        was_training = self.training
        self.eval()
        
        try:
            for i in range(batch_size):
                support_pairs = support_pairs_list[i]
                
                for support_pair in support_pairs:
                    support_input = support_pair["inp"].to(device)
                    support_output = support_pair["out"].to(device)
                    
                    support_input_emb = self._input_embeddings(support_input.unsqueeze(0), puzzle_identifiers[i:i+1])
                    
                    support_carry = self.empty_carry(1, device)
                    
                    with torch.no_grad():
                        z_H, z_L = support_carry.z_H, support_carry.z_L
                        
                        # Initialize z_H and z_L with proper values (not uninitialized garbage)
                        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(1, self.seq_len + self.puzzle_emb_len, self.hidden_size)
                        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(1, self.seq_len + self.puzzle_emb_len, self.hidden_size)
                        
                        # Check initial z_H and z_L states
                        assert not torch.isnan(z_H).any(), "NaN in initial z_H state!"
                        assert not torch.isnan(z_L).any(), "NaN in initial z_L state!"
                        
                        for _H_step in range(self.H_cycles):
                            for _L_step in range(self.L_cycles):
                                if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                                    z_L = self.L_level(z_L, z_H + support_input_emb, **seq_info)
                            
                            if not (_H_step == self.H_cycles - 1):
                                z_H = self.H_level(z_H, z_L, **seq_info)
                        
                        # Check z_H before L_level
                        assert not torch.isnan(z_H).any(), "NaN in z_H before L_level in support reproduction!"
                        
                        # Check support_input_emb
                        assert not torch.isnan(support_input_emb).any(), "NaN in support_input_emb!"
                        
                        z_L = self.L_level(z_L, z_H + support_input_emb, **seq_info)
                        
                        # Check z_L after L_level
                        assert not torch.isnan(z_L).any(), "NaN in z_L after L_level in support reproduction!"
                        
                        z_H = self.H_level(z_H, z_L, **seq_info)
                        
                        # Check z_H after H_level
                        assert not torch.isnan(z_H).any(), "NaN in z_H after H_level in support reproduction!"
                        
                        support_logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
                        
                        # Check support_logits immediately after generation
                        assert not torch.isnan(support_logits).any(), "NaN in support_logits immediately after lm_head!"
                        
                        all_support_inputs.append(support_input)
                        all_support_outputs.append(support_output)
                        all_support_logits.append(support_logits.squeeze(0))
        
        finally:
            # restore original training mode
            if was_training:
                self.train()
        
        return {
            "support_inputs": all_support_inputs,
            "support_outputs": all_support_outputs, 
            "support_logits": all_support_logits
        }

    def empty_carry(
        self, batch_size: int, device: torch.device = None
    ) -> HRMInnerCarry:
        """Create empty carry state (uninitialized, must be reset/warmed before use)"""
        if device is None:
            device = next(self.parameters()).device
        return HRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.seq_len + self.puzzle_emb_len,
                self.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.seq_len + self.puzzle_emb_len,
                self.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: HRMInnerCarry
    ) -> HRMInnerCarry:
        """Reset carry state for halted sequences"""
        # Move initial states to the same device as reset_flag
        H_init = self.H_init.to(reset_flag.device)
        L_init = self.L_init.to(reset_flag.device)
        return HRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
        )

    def forward(
        self, carry: HRMInnerCarry, batch: Dict[str, torch.Tensor], support_pairs_list: List[List[Dict[str, torch.Tensor]]]
    ) -> Tuple[HRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with support examples (support_pairs is mandatory)"""
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )

        # Support example encoding (permutation invariant) - mandatory
        # Dual-path support conditioning: local mapping + global summary → fused conditioning
        encoded_support_pairs = self._create_support_mapping(support_pairs_list, batch["puzzle_identifiers"])
        
        # Apply dual-path conditioning to input embeddings
        input_embeddings = self._apply_dual_path_conditioning(input_embeddings, encoded_support_pairs)

        # Forward iterations (matches original)
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.H_cycles):
                for _L_step in range(self.L_cycles):
                    if not (
                        (_H_step == self.H_cycles - 1)
                        and (_L_step == self.L_cycles - 1)
                    ):
                        # L level update: z_H + input_embeddings (matches original)
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.H_cycles - 1):
                    # H level update: z_L (matches original)
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step gradient update
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        
        # Assert puzzle layout invariants
        assert output.shape[1] == self.seq_len, f"Expected seq_len={self.seq_len}, got {output.shape[1]}"

        # Q head (use first main token, not puzzle prefix)
        P = self.puzzle_emb_len
        q_logits = self.q_head(z_H[:, P]).to(torch.float32)

        # Support reproduction outputs
        support_outputs = self._reproduce_support_examples(support_pairs_list, batch["puzzle_identifiers"], seq_info)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), support_outputs


class HRM(nn.Module):
    """Main HRM model with ACT wrapper (matches original)"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.inner = HRMInner(config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> HRMCarry:
        """Initialize carry state (matches original)"""
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return HRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones(
                (batch_size,), dtype=torch.bool, device=device
            ),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self, carry: HRMCarry, batch: Dict[str, torch.Tensor], support_pairs: List[List[Dict[str, torch.Tensor]]]
    ) -> Tuple[HRMCarry, Dict[str, torch.Tensor]]:
        """Forward pass with ACT and support examples (support_pairs is mandatory)"""
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Get support pairs for current batch (handle halted sequences)
        current_support_pairs = []
        for i, halted in enumerate(carry.halted):
            if not halted and i < len(support_pairs):
                current_support_pairs.append(support_pairs[i])
            else:
                current_support_pairs.append([])  # Empty support for halted sequences

        # Forward inner model with support examples
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), support_outputs = self.inner(
            new_inner_carry, new_current_data, current_support_pairs
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "support_outputs": support_outputs,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.inner.halt_max_steps

            # Default: halt if max steps reached
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.inner.halt_max_steps > 1):
                # Halt signal
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.inner.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.inner.halt_max_steps + 1
                )

                # Force continue for exploration steps
                exploration_continue = (new_steps < min_halt_steps)
                halted = halted & ~exploration_continue

                # Compute target Q (bootstrapping)
                _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(
                    new_inner_carry, new_current_data, current_support_pairs
                )

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return HRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs


# ------------------------------------------------------------
# Loss function with ACT
# ------------------------------------------------------------
def hrm_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,  # (B, L) ints
    carry: HRMCarry,
    config: Dict = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """HRM loss with ACT Q-learning"""

    if config is None:
        config = {"act_weight": 0.5}

    # Correctness computation
    with torch.no_grad():
        mask = labels != -100  # ignore label
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs

        is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
        seq_is_correct = is_correct.sum(-1) == loss_counts

        # Content vs background token analysis
        content_mask = (labels != 0) & (labels != -100)  # Non-background, non-padding
        background_mask = (labels == 0) & (labels != -100)  # Background, non-padding
        
        # Calculate separate accuracies
        content_correct = content_mask & is_correct
        background_correct = background_mask & is_correct
        
        content_counts = content_mask.sum(-1)
        background_counts = background_mask.sum(-1)
        
        # Content-weighted accuracy (80% content, 20% background)
        content_accuracy = torch.where(
            content_counts > 0, 
            content_correct.sum(-1).float() / content_counts.float(), 
            torch.zeros_like(content_counts, dtype=torch.float32)
        )
        background_accuracy = torch.where(
            background_counts > 0,
            background_correct.sum(-1).float() / background_counts.float(),
            torch.zeros_like(background_counts, dtype=torch.float32)
        )
        
        # Weighted accuracy to fix background dominance
        weighted_accuracy = 0.8 * content_accuracy + 0.2 * background_accuracy
        
        # Metrics (all sequences with valid labels)
        valid_metrics = loss_counts > 0
        
        # Overall accuracy (original)
        overall_accuracy = torch.where(
            valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0
        )
        metrics = {
            "count": valid_metrics.sum(),
            "accuracy": weighted_accuracy.sum(),  # Use weighted accuracy instead
            "overall_accuracy": overall_accuracy.sum(),  # Keep original for comparison
            "content_accuracy": torch.where(valid_metrics, content_accuracy, 0).sum(),
            "background_accuracy": torch.where(valid_metrics, background_accuracy, 0).sum(),
            "content_coverage": torch.where(valid_metrics, content_counts.float() / loss_counts.float(), 0).sum(),
            "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
            "q_halt_accuracy": (
                valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
            ).sum(),
            "steps": torch.where(valid_metrics, carry.steps, 0).sum(),
        }

    # Losses
    # Language modeling loss (run in float32 for numerical stability)
    logits = outputs["logits"].to(torch.float32)
    lm_loss = (
        F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape)
        / loss_divisor
    ).sum()
    
    # Check for NaN in lm_loss
    assert not torch.isnan(lm_loss), "NaN detected in lm_loss!"

    # Q halt loss (binary cross-entropy)
    q_halt_loss = F.binary_cross_entropy_with_logits(
        outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype)
    )
    
    # Check for NaN in q_halt_loss
    assert not torch.isnan(q_halt_loss), "NaN detected in q_halt_loss!"

    # Q continue loss (bootstrapping target loss)
    q_continue_loss = 0
    if "target_q_continue" in outputs:
        q_continue_loss = F.binary_cross_entropy_with_logits(
            outputs["q_continue_logits"], outputs["target_q_continue"]
        )

    # Support reproduction loss
    support_reproduction_loss = 0
    if "support_outputs" in outputs and outputs["support_outputs"]["support_logits"]:
        support_logits = outputs["support_outputs"]["support_logits"]
        support_outputs = outputs["support_outputs"]["support_outputs"]
        
        # Check support outputs for NaN
        for i, support_logit in enumerate(support_logits):
            assert not torch.isnan(support_logit).any(), f"NaN detected in support_logit {i}!"
        
        # Compute loss for each support example (run in float32 for numerical stability)
        for i, (support_logit, support_output) in enumerate(zip(support_logits, support_outputs)):
            support_logit_f32 = support_logit.to(torch.float32)
            support_loss = F.cross_entropy(
                support_logit_f32.view(-1, support_logit_f32.shape[-1]),
                support_output.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            
            # Check individual support loss
            assert not torch.isnan(support_loss), f"NaN detected in support_loss {i}!"
            
            support_reproduction_loss += support_loss
        
        # Average across support examples
        if len(support_logits) > 0:
            support_reproduction_loss = support_reproduction_loss / len(support_logits)
            
        # Check final support reproduction loss
        assert not torch.isnan(support_reproduction_loss), "NaN detected in final support_reproduction_loss!"

    # Total loss (matches original weighting + support reproduction)
    support_weight = config.get("support_weight", 0.1)  # Weight for support reproduction
    total_loss = lm_loss + config.get("act_weight", 0.5) * (
        q_halt_loss + q_continue_loss
    ) + support_weight * support_reproduction_loss
    
    # Check final total loss
    assert not torch.isnan(total_loss), "NaN detected in total_loss!"

    metrics.update(
        {
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "q_continue_loss": q_continue_loss.detach()
            if q_continue_loss != 0
            else torch.tensor(0.0),
            "support_reproduction_loss": support_reproduction_loss.detach()
            if support_reproduction_loss != 0
            else torch.tensor(0.0),
        }
    )

    return total_loss, metrics
