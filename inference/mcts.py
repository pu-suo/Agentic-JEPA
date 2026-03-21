"""
Discrete MCTS Planner for Agentic JEPA.

Action Proposal Strategy (Prototype Heuristic):
- Branch 1: Ground-truth action from dataset (for evaluation / ranking test)
- Branches 2..k: Random actions sampled from training action vocabulary

This isolates the test of the Latent Value Head's ranking ability without
requiring a separate policy network. Replace with a learned policy head or
LLM-based proposer in later phases.

Inference loop per decision step:
  1. Propose k candidate actions
  2. For each, run Afterstate Predictor → as_t^(k)
  3. Evaluate with Value Head → V(as_t^(k))
  4. Select highest-value action
  5. Execute best_action DIRECTLY (Talker bypassed — see NON-NEGOTIABLE note below)
  6. Receive observation from environment
  7. Run SLERP Fusion → h_{t+1}
  8. Compute surprise Δ_t = V(h_{t+1}) - V(as_t)
  9. If Δ_t < delta_fatal → backtrack

NON-NEGOTIABLE: The selected action string is executed directly without Talker
re-generation. The action string is already known (from the vocabulary or
gt_action); routing it through the Talker is lossy — the Talker may produce a
different or syntactically invalid string, triggering false backtracks even when
the planning was correct. The Talker is retained for standalone generation use
(future: generating actions from pure latent states) but MUST NOT be used in
the plan_and_act loop.
"""
import logging
import random
from typing import List, Optional, Tuple

import torch

from models.afterstate_predictor import AfterstatePredictor
from models.encoders import TextEncoder
from models.slerp_fusion import GatedSLERPFusion
from models.talker import Talker
from models.value_head import LatentValueHead
from inference.backtracking import BacktrackingController
from utils.math_utils import l2_normalize

logger = logging.getLogger(__name__)


class MCTSPlanner:
    """
    Discrete MCTS planner operating in latent space.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        predictor: AfterstatePredictor,
        slerp_fusion: GatedSLERPFusion,
        value_head: LatentValueHead,
        talker: Talker,
        tokenizer,
        action_vocabulary: List[str],
        device: torch.device,
        n_branches: int = 5,
        delta_fatal: float = -0.5,
        talker_retries: int = 3,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.slerp_fusion = slerp_fusion
        self.value_head = value_head
        self.talker = talker
        self.tokenizer = tokenizer
        self.action_vocabulary = action_vocabulary
        self.device = device
        self.n_branches = n_branches
        self.backtracking = BacktrackingController(delta_fatal=delta_fatal)
        self.talker_retries = talker_retries

    def _propose_actions(
        self, gt_action: Optional[str] = None
    ) -> List[str]:
        """
        Propose k candidate action strings.

        Branch 1: Ground-truth action (if provided; used for evaluation).
        Branches 2..k: Random samples from the action vocabulary.
        """
        candidates = []
        if gt_action is not None:
            candidates.append(gt_action)

        n_random = self.n_branches - len(candidates)
        if n_random > 0 and self.action_vocabulary:
            sampled = random.sample(
                self.action_vocabulary,
                min(n_random, len(self.action_vocabulary))
            )
            candidates.extend(sampled)

        # Pad to n_branches if vocabulary is small
        while len(candidates) < self.n_branches and candidates:
            candidates.append(random.choice(candidates))

        return candidates[:self.n_branches]

    @torch.no_grad()
    def plan_and_act(
        self,
        h_t: torch.Tensor,
        context_text: str,
        observation_fn,
        gt_action: Optional[str] = None,
        masked_actions: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Execute one full decision step: plan → decode → execute → fuse.

        Args:
            h_t: (1, d_model) current latent state
            context_text: current context string (for logging)
            observation_fn: callable(action_text) → observation_text
                            Simulates or calls the environment.
            gt_action: Ground-truth action (used as Branch 1 for eval)
            masked_actions: Actions to exclude (previously failed)

        Returns:
            h_next: (1, d_model) next latent state after fusion
            trace: dict with all intermediate values for logging
        """
        masked_actions = masked_actions or []
        trace = {
            "candidates": [],
            "values": [],
            "selected_action": None,
            "selected_as": None,
            "generated_code": None,
            "syntax_valid": False,
            "observation": None,
            "alpha": None,
            "h_fused": None,
            "delta": None,
            "backtracked": False,
        }

        max_retries = len(self._propose_actions(gt_action)) + 1

        for attempt in range(max_retries):
            # === 1. Propose candidate actions ===
            candidates = [
                a for a in self._propose_actions(gt_action)
                if a not in masked_actions
            ]
            if not candidates:
                logger.warning("No candidate actions available; aborting.")
                return h_t, trace

            trace["candidates"] = candidates

            # === 2–3. Evaluate each candidate with value head ===
            best_action = None
            best_value = float('-inf')
            best_as = None
            candidate_values = []

            for action_text in candidates:
                action_embed = self.encoder([action_text])  # (1, d)
                as_k, _, _, _, _ = self.predictor(h_t, action_embed)  # unpack all 5 returns
                v_k = self.value_head(as_k).item()
                candidate_values.append(v_k)

                if v_k > best_value:
                    best_value = v_k
                    best_action = action_text
                    best_as = as_k

            trace["values"] = candidate_values
            trace["selected_action"] = best_action
            trace["selected_as"] = best_as

            # === 4. Cache state for potential backtrack ===
            self.backtracking.cache_state(
                h_t=h_t,
                afterstate=best_as,
                value_before=best_value,
                available_actions=[a for a in candidates if a != best_action],
            )

            # === 5. Execute best_action directly (Talker bypassed) ===
            # best_action is a known string from the action vocabulary or gt_action.
            # It was already used to compute the latent value; execute it as-is.
            # The Talker is NOT called here — re-generating from the latent afterstate
            # is lossy and introduces AST-validation failures even when the plan is
            # correct, causing false backtracks on valid strategies.
            # (self.talker retained for standalone generation path; not used here.)
            trace["generated_code"] = best_action
            trace["syntax_valid"] = True

            # === 6. Execute in environment ===
            observation_text = observation_fn(best_action)
            trace["observation"] = observation_text

            # === 8. SLERP Fusion ===
            obs_embed = self.encoder([observation_text])
            h_fused, alpha = self.slerp_fusion(best_as, obs_embed)

            trace["alpha"] = alpha.item()
            trace["h_fused"] = h_fused

            # === 9–10. Surprise check ===
            v_fused = self.value_head(h_fused).item()
            delta = v_fused - best_value
            trace["delta"] = delta

            logger.debug(
                f"Δ_t = {delta:.4f} | V(as)={best_value:.4f} | "
                f"V(h_fused)={v_fused:.4f}"
            )

            decision = self.backtracking.check_surprise(v_fused)

            if decision == "backtrack":
                logger.info(
                    f"Surprise Δ_t={delta:.4f} < delta_fatal. Backtracking."
                )
                trace["backtracked"] = True
                cached = self.backtracking.backtrack()
                if cached is None:
                    logger.warning("Backtrack cache empty; returning current state.")
                    return h_fused, trace

                masked_actions.append(best_action)
                h_t = cached['h_t']
                self.backtracking.pop_state()
                continue
            elif decision == "warn":
                logger.warning(f"Soft surprise warning: Δ_t={delta:.4f}")

            # === Success: return fused state ===
            return h_fused, trace

        # Exhausted all attempts
        logger.warning("Exhausted all planning attempts. Returning current h_t.")
        return h_t, trace

    def run_trajectory(
        self,
        initial_context: str,
        observation_fn,
        gt_actions: Optional[List[str]] = None,
        max_steps: int = 10,
    ) -> List[dict]:
        """
        Run a full inference trajectory.

        Args:
            initial_context: Initial context string
            observation_fn: callable(action_text) → observation_text
            gt_actions: Optional list of ground-truth actions (for eval)
            max_steps: Maximum number of steps

        Returns:
            List of per-step trace dicts
        """
        # Encode initial context
        h_t = self.encoder([initial_context])  # (1, d_model)
        traces = []

        for step in range(max_steps):
            gt_action = gt_actions[step] if gt_actions and step < len(gt_actions) else None

            h_t, trace = self.plan_and_act(
                h_t=h_t,
                context_text=initial_context,
                observation_fn=observation_fn,
                gt_action=gt_action,
            )
            trace["step"] = step
            traces.append(trace)

            action_preview = trace['selected_action'][:50] if trace['selected_action'] else 'None'
            alpha_str = f"{trace['alpha']:.3f}" if trace['alpha'] is not None else 'N/A'
            delta_str = f"{trace['delta']:.4f}" if trace['delta'] is not None else 'N/A'
            logger.info(
                f"Step {step}: action={action_preview!r}... "
                f"| α={alpha_str} "
                f"| Δ={delta_str}"
            )

        return traces
