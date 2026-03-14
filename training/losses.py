"""
All loss functions for Agentic JEPA.

The unified loss is:
    L = τ(a_t) · λ_JEPA · L_JEPA + λ_V · L_V + λ_p · N_steps

Where:
- L_JEPA = cosine distance between predicted afterstate and EMA target
           of the ACTION TEXT
- L_V = MSE between V(h_{t+1}) and empirical return R_t
         where h_{t+1} is the POST-SLERP fused state
- N_steps = ponder step count from ACT (scalar per sample)
"""
import torch
import torch.nn.functional as F
from utils.math_utils import cosine_distance


def jepa_loss(predicted_afterstate: torch.Tensor,
              ema_target: torch.Tensor,
              tau: torch.Tensor) -> torch.Tensor:
    """
    Action-weighted JEPA cosine distance loss.

    Args:
        predicted_afterstate: (batch, d) predictor output, L2-normalized
        ema_target: (batch, d) stop-gradient EMA target, L2-normalized
                    THIS IS THE ENCODING OF THE ACTION TEXT via the EMA encoder.
                    The predictor learns: given state h_t, predict EMA_Enc(action_text).
                    Using action_text (not context_after_action) prevents degeneracy
                    where CLS(long_context_before) ≈ CLS(long_context_before + action).
        tau: (batch,) action-type weight. ~1.0 for internal, ~0.1 for external

    Returns:
        scalar loss (mean over batch)
    """
    cos_dist = cosine_distance(predicted_afterstate, ema_target.detach())
    weighted = tau * cos_dist
    return weighted.mean()


def value_loss(v_fused: torch.Tensor, v_afterstate: torch.Tensor,
               target_return: torch.Tensor) -> torch.Tensor:
    """
    Dual-evaluation MSE loss for the value head.

    The value head must be calibrated on BOTH post-SLERP fused states
    (for the backtracking controller's surprise metric at inference) AND
    afterstates (for MCTS planning before observations arrive).

    If trained only on fused states, V(as_t) during planning is
    out-of-distribution. If trained only on afterstates, V(h_{t+1})
    during backtracking is out-of-distribution.

    The fused-state loss is primary (weight 1.0) because backtracking
    correctness is safety-critical. The afterstate loss is auxiliary
    (weight 0.5) for planning calibration.

    Args:
        v_fused: (batch,) V_θ(h_{t+1}) — value of POST-SLERP fused state
        v_afterstate: (batch,) V_θ(as_t) — value of afterstate (pre-observation)
        target_return: (batch,) empirical return R_t (binary for Phase 1)

    Returns:
        scalar loss
    """
    loss_fused = F.mse_loss(v_fused, target_return)
    loss_as = F.mse_loss(v_afterstate, target_return)
    return loss_fused + 0.5 * loss_as


def compute_total_loss(l_jepa: torch.Tensor, l_value: torch.Tensor,
                       ponder_cost: torch.Tensor, lambda_jepa: float,
                       lambda_v: float, lambda_ponder: float) -> torch.Tensor:
    """
    Compute weighted total loss.

    ponder_cost must be a differentiable scalar using the Graves (2016) ACT
    formulation: mean(N.detach() + R), where N is the discrete step count and
    R is the remainder term (differentiable). Do NOT pass the raw discrete
    n_steps count here; it is not differentiable.
    """
    total = lambda_jepa * l_jepa + lambda_v * l_value + lambda_ponder * ponder_cost
    return total
