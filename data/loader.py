"""
CodeActInstruct Data Loader for Agentic JEPA.

Each trajectory is parsed into a sequence of (context, action, observation, reward) tuples.

CRITICAL: The afterstate target boundary.
==================================================
The JEPA loss trains the predictor to match the EMA encoding of the
PRE-OBSERVATION state — the state AFTER the agent's action but BEFORE
the environment responds.

In CodeActInstruct, turns alternate:
  - Assistant message (code action a_t)
  - Environment message (terminal output o_t)

The afterstate target for step t encodes:
  context = concat(c_0, a_1, o_1, ..., a_{t-1}, o_{t-1}, a_t)
                                                           ^^^^
                                        INCLUDES the current action

  NOT:
  context = concat(c_0, a_1, o_1, ..., a_t, o_t)
                                             ^^^^
                              EXCLUDES the current observation

If this boundary is wrong, the entire afterstate factorization is
invalidated and the Fréchet mean gradient pathology returns.
==================================================
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from data.action_classifier import classify_action


class TrajectoryStep:
    """A single (action, observation) step in a trajectory."""
    def __init__(self, action_text: str, observation_text: str,
                 action_type_weight: float):
        self.action_text = action_text
        self.observation_text = observation_text
        self.tau = action_type_weight  # τ(a_t) for JEPA loss weighting


class Trajectory:
    """A complete trajectory: initial context, sequence of steps, final reward."""
    def __init__(self, context: str, steps: List[TrajectoryStep], reward: float):
        self.context = context
        self.steps = steps
        self.reward = reward  # Binary: 1.0 = task solved, 0.0 = failed


def parse_codeact_trajectory(raw: dict) -> Optional[Trajectory]:
    """
    Parse a single CodeActInstruct trajectory JSON into our format.

    Expected structure (adapt to actual dataset format):
    {
        "conversations": [
            {"role": "user", "content": "..."},        # c_0
            {"role": "assistant", "content": "..."},    # a_1 (Python code)
            {"role": "environment", "content": "..."},  # o_1 (terminal output)
            {"role": "assistant", "content": "..."},    # a_2
            {"role": "environment", "content": "..."},  # o_2
            ...
        ],
        "reward": 1  # or 0
    }

    Adapt the parsing logic to match the actual CodeActInstruct schema.
    """
    convos = raw.get("conversations", [])
    if len(convos) < 3:
        return None

    context = convos[0]["content"]  # Initial user prompt
    steps = []

    i = 1
    while i + 1 < len(convos):
        action_msg = convos[i]
        obs_msg = convos[i + 1]

        if action_msg.get("role") != "assistant":
            i += 1
            continue

        action_text = action_msg["content"]
        obs_text = obs_msg["content"] if obs_msg.get("role") in ("environment", "tool") else ""

        tau = classify_action(action_text)
        steps.append(TrajectoryStep(action_text, obs_text, tau))
        i += 2

    if not steps:
        return None

    reward = float(raw.get("reward", 0))
    return Trajectory(context, steps, reward)


def load_trajectories_from_jsonl(filepath: str, max_count: int = None) -> List[Trajectory]:
    """Load trajectories from a JSONL file."""
    trajectories = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            traj = parse_codeact_trajectory(raw)
            if traj is not None:
                trajectories.append(traj)
                if max_count and len(trajectories) >= max_count:
                    break
    return trajectories


class AgenticJEPADataset(Dataset):
    """
    Dataset that yields trajectory steps with all required fields.

    Each item returns:
    - pre_action_text: The full text up to and including action a_t
                       (for afterstate target encoding)
    - action_text: The action a_t text (for action embedding)
    - observation_text: The observation o_t text (for SLERP fusion)
    - post_observation_text: Full text including o_t (for value head target context)
    - tau: Action-type weight for JEPA loss
    - reward: Binary task success (for value head)
    - is_terminal: Whether this is the last step
    """

    def __init__(self, trajectories: List[Trajectory], tokenizer, max_len: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for traj in trajectories:
            cumulative_context = traj.context
            for step_idx, step in enumerate(traj.steps):
                # === AFTERSTATE TARGET BOUNDARY ===
                # pre_action_text: everything up to AND INCLUDING the current action
                pre_action_text = cumulative_context + "\n" + step.action_text

                # post_observation_text: everything INCLUDING the observation
                post_obs_text = pre_action_text + "\n" + step.observation_text

                self.samples.append({
                    "pre_action_text": pre_action_text,
                    "action_text": step.action_text,
                    "observation_text": step.observation_text,
                    "post_observation_text": post_obs_text,
                    "tau": step.tau,
                    "reward": traj.reward,
                    "is_terminal": (step_idx == len(traj.steps) - 1),
                    "step_idx": step_idx,
                    "total_steps": len(traj.steps),
                })

                # Update cumulative context for next step
                cumulative_context = post_obs_text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Return raw text and metadata; encoding happens in model forward pass
        return {
            "pre_action_text": sample["pre_action_text"],
            "action_text": sample["action_text"],
            "observation_text": sample["observation_text"],
            "post_observation_text": sample["post_observation_text"],
            "tau": torch.tensor(sample["tau"], dtype=torch.float32),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "is_terminal": sample["is_terminal"],
            "step_idx": sample["step_idx"],
            "total_steps": sample["total_steps"],
        }


def collate_fn(batch):
    """Custom collate to handle variable-length text fields."""
    return {
        "pre_action_text": [item["pre_action_text"] for item in batch],
        "action_text": [item["action_text"] for item in batch],
        "observation_text": [item["observation_text"] for item in batch],
        "post_observation_text": [item["post_observation_text"] for item in batch],
        "tau": torch.stack([item["tau"] for item in batch]),
        "reward": torch.stack([item["reward"] for item in batch]),
        "is_terminal": [item["is_terminal"] for item in batch],
        "step_idx": [item["step_idx"] for item in batch],
        "total_steps": [item["total_steps"] for item in batch],
    }


def create_dataloaders(trajectories: List[Trajectory], tokenizer,
                       val_split: float = 0.1, batch_size: int = 16,
                       max_len: int = 1024):
    """Split trajectories into train/val and create DataLoaders."""
    n_val = max(1, int(len(trajectories) * val_split))
    val_trajs = trajectories[-n_val:]
    train_trajs = trajectories[:-n_val]

    train_ds = AgenticJEPADataset(train_trajs, tokenizer, max_len)
    val_ds = AgenticJEPADataset(val_trajs, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader
