"""
CodeActInstruct Data Loader for Agentic JEPA.

Downloads and parses the xingyaoww/code-act dataset from HuggingFace.

CRITICAL: Real dataset schema differences from initial assumptions:
  - Environment observations come as "user" role (NOT "environment" role)
  - Actions are inside <execute>...</execute> tags within "assistant" messages
  - There is NO reward field — reward must be inferred from trajectory outcomes
  - Roles observed: "system", "user", "assistant" only

CRITICAL: The afterstate target boundary.
==================================================
Two context fields are produced per step:

  context_before_action: system + c_0 + a_1 + o_1 + ... + a_{t-1} + o_{t-1}
                         (EXCLUDES current action a_t)
                         → used as h_t input to the Context Encoder

  context_after_action:  system + c_0 + a_1 + o_1 + ... + a_{t-1} + o_{t-1} + a_t
                         (INCLUDES current action, EXCLUDES current observation o_t)
                         → used as the EMA Target Encoder input for the afterstate target

If this boundary is wrong, the JEPA target degenerates to a trivial
identity mapping and the entire factorization is invalidated.
==================================================
"""
import re
import logging
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from data.action_classifier import classify_action

logger = logging.getLogger(__name__)

EXECUTE_RE = re.compile(r'<execute>(.*?)</execute>', re.DOTALL)
SOLUTION_RE = re.compile(r'<solution>(.*?)</solution>', re.DOTALL)


@dataclass
class TrajectoryStep:
    """A single (action, observation) step."""
    action_text: str       # Full assistant message (may contain <execute> and <solution> tags)
    action_code: str       # Extracted code from <execute> tags (for classification)
    observation_text: str  # The subsequent user message (environment output)
    tau: float             # Action-type weight from classify_action()


@dataclass
class Trajectory:
    """A complete trajectory."""
    id: str
    context: str           # Initial task (first user message after system)
    system_prompt: str     # System message content (if present, else "")
    steps: List[TrajectoryStep]
    reward: float          # Inferred binary reward (1.0 = success, 0.0 = failure)


def extract_code_blocks(assistant_content: str) -> str:
    """
    Extract code from <execute> tags.
    Returns concatenated code blocks, or the full content if no tags found.
    """
    blocks = EXECUTE_RE.findall(assistant_content)
    if blocks:
        return "\n".join(b.strip() for b in blocks)
    return assistant_content


def infer_reward(conversations: list) -> float:
    """
    Infer binary reward from trajectory conversations.

    Success (1.0): Trajectory contains a <solution> tag AND the final user
                   message does not indicate "your answer is wrong".
    Failure (0.0): Otherwise.
    """
    has_solution = False
    last_user_content = ""

    for msg in conversations:
        if msg["role"] == "assistant" and SOLUTION_RE.search(msg["content"]):
            has_solution = True
        if msg["role"] == "user":
            last_user_content = msg["content"]

    if not has_solution:
        return 0.0
    if "your answer is wrong" in last_user_content.lower():
        return 0.0
    return 1.0


def parse_codeact_trajectory(row: dict) -> Optional[Trajectory]:
    """
    Parse a single CodeActInstruct row into a Trajectory.

    Handles the real dataset schema:
      - Roles: "system", "user", "assistant"
      - First "user" after optional "system" = initial task context c_0
      - All subsequent "user" messages = environment observations o_t
      - "assistant" messages may contain <execute>code</execute> blocks

    Args:
        row: dict with 'id' and 'conversations' keys (HuggingFace row format)
             OR legacy dict with 'conversations' and 'reward' keys (synthetic/JSONL)

    Returns:
        Trajectory or None if unparseable
    """
    traj_id = row.get("id", "unknown")
    convos = row.get("conversations", [])

    if len(convos) < 2:
        return None

    # Extract system prompt if present
    system_prompt = ""
    start_idx = 0
    if convos[0]["role"] == "system":
        system_prompt = convos[0]["content"]
        start_idx = 1

    # First user message = initial task context
    if start_idx >= len(convos) or convos[start_idx]["role"] != "user":
        return None

    initial_context = convos[start_idx]["content"]
    start_idx += 1

    # Parse alternating assistant / user turns
    steps = []
    i = start_idx
    while i < len(convos):
        if convos[i]["role"] != "assistant":
            i += 1
            continue

        action_full = convos[i]["content"]
        action_code = extract_code_blocks(action_full)

        # Next user message is the environment observation (if present)
        observation = ""
        if i + 1 < len(convos) and convos[i + 1]["role"] == "user":
            observation = convos[i + 1]["content"]
            i += 2
        else:
            i += 1  # Terminal assistant message with no following observation

        tau = classify_action(action_code)
        steps.append(TrajectoryStep(
            action_text=action_full,
            action_code=action_code,
            observation_text=observation,
            tau=tau,
        ))

    if not steps:
        return None

    # Reward: prefer explicit field (synthetic data), otherwise infer
    if "reward" in row:
        reward = float(row["reward"])
    else:
        reward = infer_reward(convos)

    return Trajectory(
        id=traj_id,
        context=initial_context,
        system_prompt=system_prompt,
        steps=steps,
        reward=reward,
    )


def load_from_huggingface(max_count: int = 1000) -> List[Trajectory]:
    """
    Load CodeActInstruct trajectories from HuggingFace (xingyaoww/code-act).

    Uses the "codeact" split (7,139 agentic trajectories).
    The "general" split contains non-agentic conversations and is NOT used.

    Requires: pip install datasets
    """
    from datasets import load_dataset

    logger.info("Loading xingyaoww/code-act dataset from HuggingFace...")
    dataset = load_dataset("xingyaoww/code-act", split="codeact")
    logger.info(f"Dataset has {len(dataset)} raw trajectories")

    trajectories = []
    for row in dataset:
        if len(trajectories) >= max_count:
            break
        traj = parse_codeact_trajectory(dict(row))
        if traj is not None:
            trajectories.append(traj)

    logger.info(
        f"Parsed {len(trajectories)} valid trajectories "
        f"(success rate: {sum(t.reward for t in trajectories)/max(len(trajectories),1):.1%})"
    )
    return trajectories


class AgenticJEPADataset(Dataset):
    """
    Dataset yielding trajectory steps with correct afterstate target boundaries.

    Each item yields:
      context_before_action  → h_t input (Context Encoder)
      context_after_action   → EMA afterstate target (includes action, excludes obs)
      action_text            → Full assistant message (for action embedding)
      observation_text       → Subsequent user/env message (for SLERP fusion)
      tau                    → Action-type weight
      reward                 → Binary trajectory success
      is_terminal            → Whether this is the last step
    """

    def __init__(self, trajectories: List[Trajectory], max_len: int = 1024):
        self.samples = []
        # max_len in chars (approx): 4 chars per token is a conservative estimate
        self._max_chars = max_len * 4

        for traj in trajectories:
            # Build the cumulative context string, starting with system + initial task
            base = ""
            if traj.system_prompt:
                base += f"[System] {traj.system_prompt}\n"
            base += f"[User] {traj.context}\n"

            cumulative_before = base  # context BEFORE the first action

            for step_idx, step in enumerate(traj.steps):
                # context_after_action: includes current action, excludes observation
                cumulative_after = cumulative_before + f"[Assistant] {step.action_text}\n"

                self.samples.append({
                    # Core JEPA fields — truncated with prompt preservation
                    "context_before_action": self._truncate_context(cumulative_before, base),
                    "context_after_action": self._truncate_context(cumulative_after, base),
                    # Action fields
                    "action_text": step.action_text,
                    "action_code": step.action_code,
                    # Observation for SLERP fusion
                    "observation_text": step.observation_text or "(no observation)",
                    # Labels
                    "tau": step.tau,
                    "reward": traj.reward,
                    "is_terminal": (step_idx == len(traj.steps) - 1),
                    "step_idx": step_idx,
                    "total_steps": len(traj.steps),
                })

                # Advance cumulative context to include observation for next step
                if step.observation_text:
                    cumulative_before = cumulative_after + f"[Observation] {step.observation_text}\n"
                else:
                    cumulative_before = cumulative_after

    def _truncate_context(self, text: str, base_prefix: str) -> str:
        """
        Truncate context to _max_chars while preserving the system+task prefix.

        If the full text fits, return unchanged. Otherwise, keep up to 25% of
        the budget for the system prompt + task (the start of the string) and
        fill the remaining 75% from the most recent context (the right side).
        This ensures the task description is never silently dropped.
        """
        if len(text) <= self._max_chars:
            return text
        prefix_len = min(len(base_prefix), self._max_chars // 4)
        suffix_len = self._max_chars - prefix_len
        return text[:prefix_len] + " [...] " + text[-suffix_len:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "context_before_action": s["context_before_action"],
            "context_after_action": s["context_after_action"],
            "action_text": s["action_text"],
            "observation_text": s["observation_text"],
            "tau": torch.tensor(s["tau"], dtype=torch.float32),
            "reward": torch.tensor(s["reward"], dtype=torch.float32),
            "is_terminal": s["is_terminal"],
            "step_idx": s["step_idx"],
            "total_steps": s["total_steps"],
        }


def collate_fn(batch: list) -> dict:
    """Custom collation: text fields stay as lists, tensors get stacked."""
    return {
        "context_before_action": [b["context_before_action"] for b in batch],
        "context_after_action": [b["context_after_action"] for b in batch],
        "action_text": [b["action_text"] for b in batch],
        "observation_text": [b["observation_text"] for b in batch],
        "tau": torch.stack([b["tau"] for b in batch]),
        "reward": torch.stack([b["reward"] for b in batch]),
        "is_terminal": [b["is_terminal"] for b in batch],
        "step_idx": [b["step_idx"] for b in batch],
        "total_steps": [b["total_steps"] for b in batch],
    }


def create_dataloaders(
    trajectories: List[Trajectory],
    val_split: float = 0.1,
    batch_size: int = 16,
    max_len: int = 1024,
):
    """Shuffle, split, and create train/val DataLoaders."""
    random.shuffle(trajectories)
    split_idx = max(1, int(len(trajectories) * (1 - val_split)))
    train_trajs = trajectories[:split_idx]
    val_trajs = trajectories[split_idx:]

    train_ds = AgenticJEPADataset(train_trajs, max_len=max_len)
    val_ds = AgenticJEPADataset(val_trajs, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
