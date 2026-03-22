"""
K-Way Accuracy Evaluation for Agentic-JEPA.

Tests whether the Latent Value Head can rank the ground-truth action
above K-1 random distractor actions from the training vocabulary.

Random baseline: 1/K (20% for K=5, 10% for K=10, 5% for K=20).
"""
import random
import logging
from typing import List, Dict

import torch
import numpy as np

logger = logging.getLogger(__name__)


@torch.no_grad()
def k_way_accuracy(
    encoder,
    predictor,
    value_head,
    val_loader,
    action_vocabulary: List[str],
    device: torch.device,
    k_values: List[int] = [5, 10, 20],
    n_trials: int = 5,  # average over multiple random samplings
) -> Dict:
    """
    Run K-way accuracy evaluation.

    For each step in the validation set, the ground-truth action is placed
    among K-1 random distractors. The value head scores all K afterstates.
    We check if ground truth ranks #1.

    Args:
        encoder: trained TextEncoder
        predictor: trained AfterstatePredictor
        value_head: trained LatentValueHead
        val_loader: validation DataLoader
        action_vocabulary: list of all unique action strings from training set
        device: torch device
        k_values: list of K values to test (e.g., [5, 10, 20])
        n_trials: number of random distractor samplings to average over

    Returns:
        dict with keys like 'top1_acc_k5', 'top1_acc_k10', 'mrr_k5', etc.
    """
    encoder.eval()
    predictor.eval()
    value_head.eval()

    results = {}

    for K in k_values:
        random_baseline = 1.0 / K
        all_top1 = []
        all_top3 = []
        all_mrr = []

        for trial in range(n_trials):
            trial_top1 = []
            trial_top3 = []
            trial_mrr = []

            for batch in val_loader:
                h_t = encoder(batch['context_before_action'])  # (B, d)
                gt_actions = batch['action_text']  # list of strings

                for i in range(len(gt_actions)):
                    gt_action = gt_actions[i]
                    h_t_i = h_t[i:i+1]  # (1, d)

                    # Sample K-1 distractors (excluding ground truth)
                    distractors = []
                    while len(distractors) < K - 1:
                        sample = random.choice(action_vocabulary)
                        if sample != gt_action and sample not in distractors:
                            distractors.append(sample)

                    # Build candidate list with ground truth at random position
                    candidates = distractors + [gt_action]
                    random.shuffle(candidates)
                    gt_idx = candidates.index(gt_action)

                    # Score all K candidates
                    values = []
                    for action_text in candidates:
                        a_embed = encoder([action_text])  # (1, d)
                        as_k, _, _, _, _ = predictor(h_t_i, a_embed)
                        v_k = value_head(as_k).item()
                        values.append(v_k)

                    # Rank
                    ranked_indices = sorted(range(K), key=lambda j: values[j], reverse=True)
                    gt_rank = ranked_indices.index(gt_idx) + 1  # 1-indexed rank

                    trial_top1.append(1.0 if gt_rank == 1 else 0.0)
                    trial_top3.append(1.0 if gt_rank <= 3 else 0.0)
                    trial_mrr.append(1.0 / gt_rank)

            all_top1.append(np.mean(trial_top1))
            all_top3.append(np.mean(trial_top3))
            all_mrr.append(np.mean(trial_mrr))

        results[f'top1_acc_k{K}'] = np.mean(all_top1)
        results[f'top1_std_k{K}'] = np.std(all_top1)
        results[f'top3_acc_k{K}'] = np.mean(all_top3)
        results[f'mrr_k{K}'] = np.mean(all_mrr)
        results[f'random_baseline_k{K}'] = random_baseline

        logger.info(
            f"K={K}: Top-1={results[f'top1_acc_k{K}']:.3f}\u00b1{results[f'top1_std_k{K}']:.3f} "
            f"(random={random_baseline:.3f}) | "
            f"Top-3={results[f'top3_acc_k{K}']:.3f} | "
            f"MRR={results[f'mrr_k{K}']:.3f}"
        )

    return results
