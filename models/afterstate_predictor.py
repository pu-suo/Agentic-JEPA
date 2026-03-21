"""
Afterstate Predictor with Adaptive Computation Time (ACT).

Takes current state h_t and action embedding a_t, produces afterstate as_t.
Uses ACT to dynamically determine how many latent reasoning steps to take.

CRITICAL: The ACT mechanism is NOT just a ponder cost penalty.
It is a full halting loop with:
1. Per-step halting probability p_t = σ(w · h + b)
2. Cumulative halting score
3. Weighted mean-field output
4. Guaranteed termination at K_max steps
"""
import torch
import torch.nn as nn
from utils.math_utils import l2_normalize


class AfterstatePredictor(nn.Module):
    def __init__(self, d_model: int = 768, n_layers: int = 4, n_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1,
                 act_max_steps: int = 8, act_halt_bias: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.act_max_steps = act_max_steps

        # Action embedding projection
        self.action_proj = nn.Linear(d_model, d_model)

        # State fusion: combine h_t and a_t into a single representation
        self.state_action_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Single Transformer block (applied recurrently for ACT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.reasoning_block = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # === ACT Halting Unit ===
        # Produces a scalar halting probability at each step
        self.halt_linear = nn.Linear(d_model, 1)
        # IMPORTANT: Initialize bias to encourage more computation initially
        nn.init.constant_(self.halt_linear.bias, act_halt_bias)

    def forward(self, h_t: torch.Tensor, action_embed: torch.Tensor):
        """
        Perform adaptive-depth latent reasoning.

        Args:
            h_t: (batch, d_model) current state on S^(d-1)
            action_embed: (batch, d_model) L2-normalized action embedding

        Returns:
            as_t: (batch, d_model) afterstate on S^(d-1)
            n_steps: (batch,) number of ponder steps taken per sample
            halt_probs: list of (batch, 1) halting probabilities per step
            remainders: (batch, 1) ACT remainder term R = 1 - Σ_{k<N} p_k
            still_running_list: list of (batch, 1) binary masks — 1.0 if sample hadn't halted yet at that step
        """
        batch_size = h_t.shape[0]
        device = h_t.device

        # Fuse state and action
        fused = self.state_action_fusion(torch.cat([h_t, action_embed], dim=-1))
        state = l2_normalize(fused)  # Start on manifold

        # === ACT Loop ===
        cumulative_halt = torch.zeros(batch_size, 1, device=device)
        remainders = torch.zeros(batch_size, 1, device=device)
        n_updates = torch.zeros(batch_size, 1, device=device)
        weighted_state = torch.zeros(batch_size, self.d_model, device=device)
        halt_probs_list = []
        still_running_list = []

        for step in range(self.act_max_steps):
            # One reasoning step (Transformer pass)
            # Reshape for transformer: (batch, 1, d_model)
            state_seq = state.unsqueeze(1)
            state_seq = self.reasoning_block(state_seq)
            state = l2_normalize(state_seq.squeeze(1))  # Stay on manifold

            # Compute halting probability
            p = torch.sigmoid(self.halt_linear(state))  # (batch, 1)
            halt_probs_list.append(p)

            # Determine which samples are still running
            still_running = (cumulative_halt < 1.0).float()
            still_running_list.append(still_running.detach())  # mask only, no grad flow

            # Which samples halt at this step
            new_halted = ((cumulative_halt + p) >= 1.0).float() * still_running

            # Update cumulative halt score
            cumulative_halt = cumulative_halt + p * still_running

            # Compute remainder for samples that just halted
            remainders = remainders + new_halted * (1.0 - (cumulative_halt - p))

            # Weight for this step's state
            weight = p * still_running + new_halted * (1.0 - (cumulative_halt - p))
            weight = weight.clamp(min=0.0)  # Safety

            # Accumulate weighted state
            weighted_state = weighted_state + weight * state

            # Track number of updates
            n_updates = n_updates + still_running

            # Early exit if all samples have halted
            if (cumulative_halt >= 1.0).all():
                break

        # Final afterstate: L2-normalized weighted mean-field output
        as_t = l2_normalize(weighted_state)
        n_steps = n_updates.squeeze(-1)

        return as_t, n_steps, halt_probs_list, remainders, still_running_list
