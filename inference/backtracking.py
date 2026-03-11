"""
Backtracking Controller.

Inference-time heuristic — NOT a trained module.
Caches afterstates at decision nodes and reverts when surprise metric
indicates a catastrophic environmental response.
"""
import torch
from typing import List, Optional


class BacktrackingController:
    def __init__(self, delta_fatal: float = -0.5):
        self.delta_fatal = delta_fatal
        self.state_cache: List[dict] = []

    def cache_state(self, h_t: torch.Tensor, afterstate: torch.Tensor,
                    value_before: float, available_actions: list):
        """Cache state at a decision node before acting."""
        self.state_cache.append({
            'h_t': h_t.clone(),
            'as_t': afterstate.clone(),
            'v_as': value_before,
            'remaining_actions': list(available_actions),
        })

    def check_surprise(self, value_after: float) -> str:
        """
        Compute surprise and decide action.

        Returns: 'continue', 'warn', or 'backtrack'
        """
        if not self.state_cache:
            return 'continue'

        v_as = self.state_cache[-1]['v_as']
        delta = value_after - v_as

        if delta < self.delta_fatal:
            return 'backtrack'
        elif delta < self.delta_fatal * 0.5:  # Soft warning zone
            return 'warn'
        return 'continue'

    def backtrack(self) -> Optional[dict]:
        """
        Revert to the most recent cached state.
        Returns the cached state dict, or None if cache is empty.
        The caller should remove the failed action from remaining_actions.
        """
        if not self.state_cache:
            return None
        return self.state_cache[-1]  # Don't pop — may need to backtrack again

    def pop_state(self):
        """Remove the most recent cached state (all branches exhausted)."""
        if self.state_cache:
            self.state_cache.pop()
