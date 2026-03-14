"""
Three-Stage Curriculum Controller.

Manages loss weight scheduling and stage transitions based on
measurable convergence criteria.

Stage 0 (Pure JEPA): Only JEPA loss active. No observations, no value head.
    Transition when: JEPA val loss plateaus (< ε improvement for N evals)

Stage 1 (Mild Observations): JEPA + low-weight Value + SLERP on deterministic obs.
    Transition when: mean α_t > 0.1 AND JEPA loss within 5% of Stage 0 plateau

Stage 2 (Full Stochasticity): All losses active including ponder cost.
    No transition — train until convergence or max epochs.
"""
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CurriculumState:
    current_stage: int = 0
    stage0_plateau_value: float = float('inf')
    stage0_eval_history: List[float] = field(default_factory=list)
    stage1_alpha_history: List[float] = field(default_factory=list)
    stage1_jepa_history: List[float] = field(default_factory=list)
    stage1_eval_count: int = 0


class CurriculumController:
    def __init__(self, config):
        self.config = config
        self.state = CurriculumState()

    def get_loss_weights(self):
        """Return (lambda_jepa, lambda_v, lambda_ponder) for current stage."""
        if self.state.current_stage == 0:
            return 1.0, 0.0, 0.0
        elif self.state.current_stage == 1:
            return 1.0, 0.05, 0.0
        else:  # Stage 2
            return 1.0, 0.1, 0.01

    def get_data_filter(self):
        """Return which trajectory steps to include in training."""
        if self.state.current_stage == 0:
            return "internal_only"   # Only steps with τ(a_t) ≈ 1.0
        elif self.state.current_stage == 1:
            return "successful_obs"  # Include obs, but only successful ones (exit code 0)
        else:
            return "all"             # Full stochastic distribution

    def check_transition(self, eval_metrics: dict) -> bool:
        """
        Check if we should transition to the next stage.
        Call this after each validation evaluation.

        Args:
            eval_metrics: dict with keys:
                - 'jepa_loss': float (validation JEPA loss)
                - 'mean_alpha': float (mean SLERP gate value, Stage 1+)

        Returns:
            True if stage advanced
        """
        if self.state.current_stage == 0:
            return self._check_stage0_transition(eval_metrics)
        elif self.state.current_stage == 1:
            return self._check_stage1_transition(eval_metrics)
        return False  # Stage 2 has no transition

    def _check_stage0_transition(self, metrics: dict) -> bool:
        jepa_loss = metrics['jepa_loss']
        self.state.stage0_eval_history.append(jepa_loss)

        if len(self.state.stage0_eval_history) < self.config.stage0_plateau_patience + 1:
            return False

        recent = self.state.stage0_eval_history[-self.config.stage0_plateau_patience:]
        older = self.state.stage0_eval_history[-(self.config.stage0_plateau_patience + 1)]

        improvements = [older - r for r in recent]
        if all(imp < self.config.stage0_plateau_epsilon for imp in improvements):
            self.state.stage0_plateau_value = min(recent)
            self.state.current_stage = 1
            logger.info(
                f"=== ADVANCING TO STAGE 1 === "
                f"(plateau at {self.state.stage0_plateau_value:.4f})"
            )
            return True
        return False

    def _check_stage1_transition(self, metrics: dict) -> bool:
        mean_alpha = metrics.get('mean_alpha', 0.0)
        jepa_loss = metrics['jepa_loss']

        self.state.stage1_alpha_history.append(mean_alpha)
        self.state.stage1_jepa_history.append(jepa_loss)

        self.state.stage1_eval_count += 1

        alpha_ok = mean_alpha > self.config.stage1_alpha_threshold
        jepa_ok = jepa_loss < self.state.stage0_plateau_value * (
            1 + self.config.stage1_jepa_tolerance
        )
        time_ok = self.state.stage1_eval_count >= self.config.stage1_max_evals

        if (alpha_ok and jepa_ok) or time_ok:
            self.state.current_stage = 2
            reason = "α threshold met" if alpha_ok else f"max evals ({self.config.stage1_max_evals}) reached"
            logger.info(
                f"=== ADVANCING TO STAGE 2 === "
                f"(α={mean_alpha:.3f}, JEPA={jepa_loss:.4f}, reason: {reason})"
            )
            return True
        return False
