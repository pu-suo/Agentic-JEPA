"""
Latent Value Head.

Maps a post-fusion latent state h_{t+1} to a scalar expected value.

CRITICAL: This must be trained on h_{t+1} (POST-SLERP fused state),
NOT on as_t (the afterstate). Reason: the backtracking controller
computes Δ_t = V(h_{t+1}) - V(as_t). If V was only trained on
afterstates, then V(h_{t+1}) is out-of-distribution at inference.

During MCTS planning (before observation), V evaluates afterstates as a
planning heuristic. This is acceptable — but during TRAINING, the MSE
loss must be computed on the post-fusion state.
"""
import torch
import torch.nn as nn


class LatentValueHead(nn.Module):
    def __init__(self, d_model: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, d_model) latent state (afterstate OR fused state)
        Returns:
            value: (batch,) scalar expected value
        """
        return self.mlp(h).squeeze(-1)
