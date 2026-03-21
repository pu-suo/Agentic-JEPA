"""
Main training loop for Agentic JEPA.

Per-step training flow:
  1. Encode context_before_action with Context Encoder → h_t
  2. Encode action_text with Context Encoder → action_embed
  3. Run Afterstate Predictor (ACT) → as_t, n_steps, halt_probs, remainders
  4. Encode context_after_action with EMA Target Encoder → ema_target (stop-gradient)
     context_after_action = context_before_action + action (pre-observation afterstate target)
  5. Compute JEPA loss: cosine_distance(as_t, sg[ema_target]) weighted by tau
     Gradients flow: L_JEPA → as_t → P_θ → h_t → E_θ (LoRA layers)
  6. Run Value Head on as_t → v_afterstate
  7. Encode observation_text with Context Encoder → obs_embed
  8. Run Gated SLERP: h_{t+1}, alpha = SLERP(as_t, obs_embed)
  9. Run Value Head on h_{t+1} → v_fused
  10. Compute Value loss: dual MSE on (v_fused, v_afterstate, reward)
  10b. Compute obs_utility_loss (Stage 1+): clamped info-gain reward for SLERP gate
  11. Compute variance regularization on h_t (collapse prevention, active when use_lora=True)
  12. Compute total loss with curriculum weights
  13. Backward, clip gradients, optimizer step
  14. EMA update (LoRA weights only)
"""
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import AgenticJEPAConfig
from models.encoders import TextEncoder, create_ema_encoder
from models.afterstate_predictor import AfterstatePredictor
from models.slerp_fusion import GatedSLERPFusion
from models.value_head import LatentValueHead
from training.losses import jepa_loss, value_loss, compute_total_loss, variance_regularization, obs_utility_loss
from training.curriculum import CurriculumController
from utils.ema import update_ema

logger = logging.getLogger(__name__)


class AgenticJEPATrainer:
    def __init__(self, config: AgenticJEPAConfig, device: torch.device):
        self.config = config
        self.device = device

        # --- Build models ---
        self.encoder = TextEncoder(
            model_name=config.encoder_name,
            d_model=config.d_model,
            freeze=config.freeze_encoder,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            lora_bias=config.lora_bias,
        ).to(device)

        self.ema_encoder = create_ema_encoder(self.encoder).to(device)

        self.predictor = AfterstatePredictor(
            d_model=config.d_model,
            n_layers=config.predictor_layers,
            n_heads=config.predictor_heads,
            ff_dim=config.predictor_ff_dim,
            dropout=config.predictor_dropout,
            act_max_steps=config.act_max_steps,
            act_halt_bias=config.act_halt_bias_init,
        ).to(device)

        self.slerp_fusion = GatedSLERPFusion(
            d_model=config.d_model,
            gate_hidden=config.slerp_gate_hidden,
            gate_bias_init=config.slerp_gate_bias_init,
        ).to(device)

        self.value_head = LatentValueHead(
            d_model=config.d_model,
            hidden_dim=config.value_hidden,
        ).to(device)

        # --- Optimizer (only trainable params) ---
        trainable_params = (
            list(self.predictor.parameters())
            + list(self.slerp_fusion.parameters())
            + list(self.value_head.parameters())
            + list(self.encoder.projection.parameters())  # always trainable
        )
        if config.use_lora:
            # Add only the LoRA adapter weights from the encoder backbone
            trainable_params += [
                p for n, p in self.encoder.named_parameters()
                if 'lora_' in n and p.requires_grad
            ]
        elif not config.freeze_encoder:
            # Full fine-tuning fallback (ablation: use_lora=False, freeze_encoder=False)
            trainable_params += [
                p for p in self.encoder.backbone.parameters() if p.requires_grad
            ]

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.learning_rate
        )

        # --- Curriculum ---
        self.curriculum = CurriculumController(config)

        # --- Logging state ---
        self.global_step = 0
        self.metrics_history: List[Dict] = []

    def _apply_data_filter(self, batch: dict, data_filter: str) -> Optional[dict]:
        """
        Filter a batch according to curriculum data filter.
        Returns filtered batch, or None if no samples remain.
        """
        if data_filter == "all":
            return batch

        tau = batch["tau"]  # (batch,)

        if data_filter == "internal_only":
            mask = tau >= 0.9  # Keep only internal actions (τ ≈ 1.0)
        elif data_filter == "successful_obs":
            # Include all steps; external steps included only if observation non-empty
            # For simplicity, include all (Stage 1 allows observations)
            mask = torch.ones(tau.shape[0], dtype=torch.bool)
        else:
            mask = torch.ones(tau.shape[0], dtype=torch.bool)

        if not mask.any():
            return None

        return {
            "context_before_action": [t for t, m in zip(batch["context_before_action"], mask) if m],
            "context_after_action": [t for t, m in zip(batch["context_after_action"], mask) if m],
            "action_text": [t for t, m in zip(batch["action_text"], mask) if m],
            "observation_text": [t for t, m in zip(batch["observation_text"], mask) if m],
            "tau": batch["tau"][mask],
            "reward": batch["reward"][mask],
            "is_terminal": [t for t, m in zip(batch["is_terminal"], mask) if m],
        }

    def train_step(self, batch: dict) -> Optional[Dict]:
        """
        Execute a single training step.
        Returns dict of metrics, or None if batch was filtered out.
        """
        lambda_jepa, lambda_v, lambda_ponder = self.curriculum.get_loss_weights()
        data_filter = self.curriculum.get_data_filter()

        # Apply curriculum data filter
        batch = self._apply_data_filter(batch, data_filter)
        if batch is None:
            return None

        tau = batch["tau"].to(self.device)
        reward = batch["reward"].to(self.device)

        self.optimizer.zero_grad()

        # === Step 1-2: Encode context and action ===
        h_t = self.encoder(batch["context_before_action"])    # (B, d) — state BEFORE action
        action_embed = self.encoder(batch["action_text"])      # (B, d)

        # === Step 3: Afterstate Predictor (ACT) ===
        as_t, n_steps, halt_probs, remainders, still_running_list = self.predictor(h_t, action_embed)
        # Penalize continuation probability for active steps only.
        # Minimizing (1-p_k)*s_k pushes p_k → 1 (halt early) for active steps.
        # Phantom steps (after a sample has already halted) are masked out by
        # still_running, preventing spurious gradients on dead steps.
        # Previous sum(p.mean()) was backwards: minimizing p pushes toward never halting.
        ponder_cost = sum(((1.0 - p) * s).mean() for p, s in zip(halt_probs, still_running_list))

        # === Step 4: EMA target — encode CONTEXT_AFTER_ACTION (stop-gradient) ===
        # JEPA target = EMA encoding of the pre-observation afterstate:
        # context_after_action = context_before_action + action text (excludes observation).
        # The predictor must learn: given h_t (state), predict where the world-state
        # lands on S^(d-1) after action a_t is applied.
        # Requires LoRA-adapted encoder to be sensitive to the appended action text.
        with torch.no_grad():
            ema_target = self.ema_encoder(batch["context_after_action"])  # (B, d)

        # === Step 5: JEPA loss ===
        l_jepa = jepa_loss(as_t, ema_target, tau)

        # === Step 6: Value of afterstate (for MCTS calibration) ===
        v_afterstate = self.value_head(as_t)

        # === Step 7: Encode observation ===
        obs_embed = self.encoder(batch["observation_text"])    # (B, d)

        # === Step 8: Gated SLERP fusion ===
        # Detach as_t so the JEPA gradient (λ=1.0) doesn't flow into the SLERP path
        # and dominate the value gradient (λ=0.5). This gives the gate an independent
        # gradient channel from the value loss.
        h_fused, alpha = self.slerp_fusion(as_t.detach(), obs_embed)

        # === Step 9: Value of fused state (for backtracking calibration) ===
        v_fused = self.value_head(h_fused)

        # === Step 10: Value loss ===
        # Stage 2: fused-only loss forces ALL reward-prediction gradients through the gate.
        # The 0.5*MSE(V(as_t), reward) term in value_loss trains V to bypass the gate via
        # the afterstate; removing it in Stage 2 eliminates that shortcut.
        # Stages 0–1: dual loss keeps afterstate value calibrated for MCTS planning.
        if self.curriculum.state.current_stage >= 2:
            l_value = F.mse_loss(v_fused, reward)
        else:
            l_value = value_loss(v_fused, v_afterstate, reward)

        # === Step 10b: Observation utility loss (Stage 1+) ===
        # Rewards the SLERP gate when incorporating the observation improves value prediction.
        # Uses clamped information-gain formulation so the gate is never penalized for exploring —
        # only rewarded when it helps. Penalty for unhelpful observations is already covered by
        # the value_loss fused component above.
        l_obs = obs_utility_loss(v_fused, v_afterstate, reward)

        # === Step 11: Variance regularization (collapse prevention) ===
        # Applied to h_t (live encoder output) so gradients flow into LoRA layers.
        # Fixed weight of 0.01 across all curriculum stages when use_lora=True.
        if self.config.use_lora:
            l_var = variance_regularization(h_t)
        else:
            l_var = torch.zeros(1, device=self.device)

        # === Step 12: Total loss ===
        total = compute_total_loss(
            l_jepa, l_value, ponder_cost,
            lambda_jepa, lambda_v, lambda_ponder
        )
        if self.config.use_lora:
            total = total + 0.01 * l_var
        if self.curriculum.state.current_stage >= 1:
            total = total + self.config.lambda_obs_utility * l_obs

        # === Step 12: Backward + gradient clip + optimizer step ===
        total.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None],
            self.config.gradient_clip
        )
        self.optimizer.step()

        # === Step 13: EMA update ===
        update_ema(self.encoder, self.ema_encoder, self.config.ema_decay)

        self.global_step += 1

        metrics = {
            "step": self.global_step,
            "stage": self.curriculum.state.current_stage,
            "loss_total": total.item(),
            "loss_jepa": l_jepa.item(),
            "loss_value": l_value.item(),
            "loss_var": l_var.item(),
            "loss_obs": l_obs.item(),
            "ponder_steps": n_steps.mean().item(),
            "mean_alpha": alpha.mean().item(),
            "lambda_jepa": lambda_jepa,
            "lambda_v": lambda_v,
            "lambda_ponder": lambda_ponder,
        }
        self.metrics_history.append(metrics)
        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Run validation and return aggregated metrics."""
        self.encoder.eval()
        self.predictor.eval()
        self.slerp_fusion.eval()
        self.value_head.eval()

        total_jepa = 0.0
        total_alpha = 0.0
        total_ponder = 0.0
        n_batches = 0

        for batch in val_loader:
            tau = batch["tau"].to(self.device)

            h_t = self.encoder(batch["context_before_action"])
            action_embed = self.encoder(batch["action_text"])
            as_t, n_steps, _, _, _ = self.predictor(h_t, action_embed)

            ema_target = self.ema_encoder(batch["context_after_action"])
            l_jepa = jepa_loss(as_t, ema_target, tau)

            obs_embed = self.encoder(batch["observation_text"])
            _, alpha = self.slerp_fusion(as_t, obs_embed)

            total_jepa += l_jepa.item()
            total_alpha += alpha.mean().item()
            total_ponder += n_steps.mean().item()
            n_batches += 1

        self.encoder.train()
        self.predictor.train()
        self.slerp_fusion.train()
        self.value_head.train()

        return {
            "jepa_loss": total_jepa / max(n_batches, 1),
            "mean_alpha": total_alpha / max(n_batches, 1),
            "mean_ponder": total_ponder / max(n_batches, 1),
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              max_epochs: int = None) -> List[Dict]:
        """
        Full training loop across all curriculum stages.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            max_epochs: Override config max_epochs_per_stage

        Returns:
            List of per-step metric dicts
        """
        max_epochs = max_epochs or self.config.max_epochs_per_stage
        all_metrics = []

        for epoch in range(max_epochs * 3):  # 3 stages max
            stage_at_epoch_start = self.curriculum.state.current_stage
            logger.info(
                f"Epoch {epoch} | Stage {stage_at_epoch_start} | "
                f"Step {self.global_step}"
            )

            for batch in train_loader:
                step_metrics = self.train_step(batch)
                if step_metrics is not None:
                    all_metrics.append(step_metrics)

                    if self.global_step % self.config.eval_every_n_steps == 0:
                        val_metrics = self.evaluate(val_loader)
                        logger.info(
                            f"[Val] Step {self.global_step} | "
                            f"JEPA={val_metrics['jepa_loss']:.4f} | "
                            f"Alpha={val_metrics['mean_alpha']:.3f} | "
                            f"Ponder={val_metrics['mean_ponder']:.2f}"
                        )
                        advanced = self.curriculum.check_transition(val_metrics)
                        if advanced:
                            new_stage = self.curriculum.state.current_stage
                            logger.info(f"Stage advanced to {new_stage}")
                            if new_stage == 1:
                                # Manifold is stable after Stage 0; reset gate bias from -3.0
                                # to -1.5 (α ≈ 0.047 → 0.18) to escape the vanishing-gradient
                                # trap. dα/d(logit) = α(1−α): 0.045 → 0.148 (3× stronger).
                                nn.init.constant_(
                                    self.slerp_fusion.gate_mlp[-1].bias, -1.5
                                )
                                logger.info(
                                    "SLERP gate bias reset to -1.5 (α ≈ 0.18) for Stage 1 onset"
                                )

            # Check if we've completed all 3 stages
            if self.curriculum.state.current_stage == 2 and epoch > 0:
                epochs_in_stage2 = epoch - (max_epochs * 2)
                if epochs_in_stage2 >= max_epochs:
                    logger.info("Stage 2 complete. Training finished.")
                    break

        return all_metrics

    def train_talker(self, talker, train_loader: DataLoader,
                     tokenizer, max_epochs: int = 10) -> List[float]:
        """
        Train the Talker module with all other components frozen.

        The Talker is trained after the Predictor is fully trained.
        Uses cross-entropy loss to reconstruct action text from latent states.
        """
        # Freeze everything except talker
        for model in [self.encoder, self.predictor, self.slerp_fusion, self.value_head]:
            for p in model.parameters():
                p.requires_grad = False

        talker = talker.to(self.device)
        talker_optimizer = torch.optim.AdamW(
            talker.parameters(), lr=self.config.learning_rate
        )
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or 0)

        epoch_losses = []

        for epoch in range(max_epochs):
            talker.train()
            epoch_loss = 0.0
            n_steps = 0

            for batch in train_loader:
                # Get latent state for each action
                with torch.no_grad():
                    h_t = self.encoder(batch["context_before_action"])
                    action_embed = self.encoder(batch["action_text"])
                    as_t, _, _, _, _ = self.predictor(h_t, action_embed)

                # Tokenize target actions
                target_enc = tokenizer(
                    batch["action_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.talker_max_tokens,
                ).to(self.device)
                target_ids = target_enc["input_ids"]  # (B, seq_len)

                # Forward pass through talker (teacher-forced)
                # Input: tokens[:-1], Target: tokens[1:]
                input_ids = target_ids[:, :-1]
                label_ids = target_ids[:, 1:]

                logits = talker(as_t.detach(), input_ids)  # (B, seq_len-1, vocab)

                # Compute cross-entropy loss
                B, S, V = logits.shape
                loss = ce_loss_fn(logits.reshape(B * S, V), label_ids.reshape(B * S))

                talker_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(talker.parameters(), self.config.gradient_clip)
                talker_optimizer.step()

                epoch_loss += loss.item()
                n_steps += 1

            avg_loss = epoch_loss / max(n_steps, 1)
            epoch_losses.append(avg_loss)
            logger.info(f"Talker epoch {epoch} | CE Loss: {avg_loss:.4f}")

        # Restore trainable parameters that were frozen for Talker training.
        # Without this, any further training calls after train_talker() would
        # silently produce zero gradients for the predictor, slerp_fusion, and value_head.
        for p in self.predictor.parameters():
            p.requires_grad = True
        for p in self.slerp_fusion.parameters():
            p.requires_grad = True
        for p in self.value_head.parameters():
            p.requires_grad = True
        for p in self.encoder.projection.parameters():
            p.requires_grad = True
        if self.config.use_lora:
            for n, p in self.encoder.named_parameters():
                if 'lora_' in n:
                    p.requires_grad = True
        elif not self.config.freeze_encoder:
            for p in self.encoder.backbone.parameters():
                p.requires_grad = True

        return epoch_losses

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts for each module."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "encoder_total": count(self.encoder),
            "encoder_trainable": count_trainable(self.encoder),
            "predictor": count(self.predictor),
            "slerp_fusion": count(self.slerp_fusion),
            "value_head": count(self.value_head),
            "total_trainable": (
                count_trainable(self.predictor)
                + count_trainable(self.slerp_fusion)
                + count_trainable(self.value_head)
                + count_trainable(self.encoder)
            ),
        }
