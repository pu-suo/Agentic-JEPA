"""
Gated SLERP Fusion Module.

Fuses stochastic observations into the afterstate on S^(d-1).

CRITICAL: Gate bias must be initialized to config.slerp_gate_bias_init (-3.0)
so that α_t ≈ 0.05 at the start of training. Without this, α_t starts at 0.5
and fuses 50% random garbage into the manifold, destroying it.
"""
import torch
import torch.nn as nn
from utils.math_utils import safe_slerp


class GatedSLERPFusion(nn.Module):
    def __init__(self, d_model: int = 768, gate_hidden: int = 256,
                 gate_bias_init: float = -3.0):
        super().__init__()

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        # CRITICAL: Initialize final bias for near-zero fusion
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias_init)

    def forward(self, afterstate: torch.Tensor, obs_embed: torch.Tensor):
        """
        Fuse observation into afterstate via Gated SLERP.

        Args:
            afterstate: (batch, d_model) afterstate on S^(d-1)
            obs_embed: (batch, d_model) observation embedding on S^(d-1)

        Returns:
            h_fused: (batch, d_model) fused state on S^(d-1)
            alpha: (batch, 1) fusion weight (for logging/visualization)
        """
        gate_input = torch.cat([afterstate, obs_embed], dim=-1)
        alpha = torch.sigmoid(self.gate_mlp(gate_input))  # (batch, 1)

        h_fused = safe_slerp(afterstate, obs_embed, alpha)

        return h_fused, alpha
