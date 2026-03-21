"""
All loss functions for Agentic JEPA.

The unified loss is:
    L = τ(a_t) · λ_JEPA · L_JEPA + λ_V · L_V + λ_p · Σ(halt_probs) + 0.01 · L_var

Where:
- L_JEPA = cosine distance between predicted afterstate and EMA target
           of the CONTEXT_AFTER_ACTION (cumulative history including the action)
- L_V = MSE between V(h_{t+1}) and empirical return R_t
         where h_{t+1} is the POST-SLERP fused state
- Σ(halt_probs) = sum of per-step mean halting probabilities (fully differentiable)
- L_var = VICReg-style variance regularization (collapse prevention, only when use_lora=True)
"""
import torch
import torch.nn.functional as F
from utils.math_utils import cosine_distance


def variance_regularization(embeddings: torch.Tensor,
                             gamma: float = 1.0,
                             eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg-style variance regularization (Bardes et al., 2022).

    Penalizes dimensions whose standard deviation falls below gamma.
    Prevents representational collapse after LoRA enables gradient flow
    into the encoder. Applied to L2-normalized encoder outputs on S^(d-1).

    Args:
        embeddings: (batch, d) L2-normalized encoder outputs
        gamma: target minimum std per dimension
        eps: numerical stability term inside sqrt

    Returns:
        scalar loss (zero when all dims have std >= gamma)
    """
    std = torch.sqrt(embeddings.var(dim=0) + eps)   # (d,)
    return torch.mean(torch.clamp(gamma - std, min=0.0))


def jepa_loss(predicted_afterstate: torch.Tensor,
              ema_target: torch.Tensor,
              tau: torch.Tensor) -> torch.Tensor:
    """
    Action-weighted JEPA cosine distance loss.

    Args:
        predicted_afterstate: (batch, d) predictor output, L2-normalized
        ema_target: (batch, d) stop-gradient EMA target, L2-normalized
                    This is EMA_Enc(context_after_action): the cumulative context
                    including the current action but excluding the observation.
                    The predictor learns: given h_t, predict where the world-state
                    lands on S^(d-1) after this action is applied.
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


def obs_utility_loss(v_fused: torch.Tensor, v_afterstate: torch.Tensor,
                     target_return: torch.Tensor) -> torch.Tensor:
    """
    Direct gradient signal to the SLERP gate. Only rewards, never penalizes.

    Measures whether mixing in the observation improves value prediction vs the
    afterstate baseline. Stop-gradient on baseline so gradient flows only through
    v_fused → h_fused → SLERP → α → gate_mlp.

    improvement = MSE(v_afterstate.detach(), r) - MSE(v_fused, r)
      > 0 when fused predicts reward better → returns negative loss (reward)
      < 0 when fused adds noise → clamped to 0 (no signal, no penalty)

    The clamp is critical: Stage 1 data frequently contains observations that add
    noise to an already-good afterstate prediction. Without the clamp, the positive
    penalty from those cases would suppress α back to near-zero — reproducing the
    retreat we observed. The existing value_loss already penalizes bad v_fused
    predictions; this function adds only reward for the cases where the gate helps.
    """
    loss_fused = F.mse_loss(v_fused, target_return)
    loss_as = F.mse_loss(v_afterstate.detach(), target_return)  # stop-grad baseline
    improvement = loss_as - loss_fused
    return -improvement.clamp(min=0)


def compute_total_loss(l_jepa: torch.Tensor, l_value: torch.Tensor,
                       ponder_cost: torch.Tensor, lambda_jepa: float,
                       lambda_v: float, lambda_ponder: float) -> torch.Tensor:
    """
    Compute weighted total loss.

    ponder_cost must be a fully differentiable scalar computed as:
        sum(p.mean() for p in halt_probs)
    where halt_probs is the list of per-step (batch,1) halting probability
    tensors from the ACT loop. This formulation has strong gradients to all
    halting units and drives variable-depth computation.

    Do NOT pass the Graves (2016) N.detach() + R formulation — the detached N
    provides no gradient and the remainder R alone is too weak, causing ACT
    to collapse to K_max on every input.
    """
    total = lambda_jepa * l_jepa + lambda_v * l_value + lambda_ponder * ponder_cost
    return total
