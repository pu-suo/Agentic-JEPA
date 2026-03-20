"""
Encoder modules: Context Encoder, EMA Target Encoder, Observation Encoder.

All encoders output L2-normalized vectors on S^(d-1).
The Context Encoder uses LoRA adapters on CodeBERT for trainable representation learning.
The EMA Target Encoder is a deepcopy whose LoRA weights are tracked via EMA (utils/ema.py).
"""
import copy
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from utils.math_utils import l2_normalize


class TextEncoder(nn.Module):
    """
    CodeBERT encoder with optional LoRA adapters that maps text -> S^(d-1).

    Base backbone weights are always frozen. With use_lora=True, ~1M LoRA adapter
    weights (query/value projections) are trainable and receive JEPA gradients,
    enabling true JEPA representation learning rather than COCONUT-style static embeddings.
    """
    def __init__(self, model_name: str = "microsoft/codebert-base",
                 d_model: int = 768, freeze: bool = True,
                 use_lora: bool = False,
                 lora_r: int = 8, lora_alpha: int = 16,
                 lora_target_modules: tuple = ("query", "value"),
                 lora_dropout: float = 0.05, lora_bias: str = "none"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Projection head: backbone_dim -> d_model (if different)
        backbone_dim = self.backbone.config.hidden_size
        if backbone_dim != d_model:
            self.projection = nn.Linear(backbone_dim, d_model)
        else:
            self.projection = nn.Identity()

        # Always freeze base CodeBERT weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        if use_lora:
            # Inject trainable LoRA adapters into query/value attention projections.
            # After get_peft_model(), only lora_A/lora_B weights have requires_grad=True.
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                lora_dropout=lora_dropout,
                bias=lora_bias,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)

    def forward(self, text_list: list) -> torch.Tensor:
        """
        Encode a list of strings into L2-normalized vectors.
        Returns: (batch_size, d_model) tensor on S^(d-1)
        """
        device = next(self.backbone.parameters()).device
        tokens = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # CodeBERT (RoBERTa-base) max_position_embeddings=514 → 512 usable
        ).to(device)

        backbone_requires_grad = any(p.requires_grad for p in self.backbone.parameters())
        if backbone_requires_grad:
            outputs = self.backbone(**tokens)
        else:
            with torch.no_grad():
                outputs = self.backbone(**tokens)

        # Use CLS token representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_hidden)
        return l2_normalize(projected)


def create_ema_encoder(context_encoder: TextEncoder) -> TextEncoder:
    """
    Create an EMA copy of the context encoder. All params set requires_grad=False.

    With LoRA: base CodeBERT weights are identical in both encoders (both frozen).
    Only the LoRA adapter weights differ and evolve — update_ema() in utils/ema.py
    tracks them by name ('lora_' in param_name), leaving frozen base weights untouched.
    """
    ema = copy.deepcopy(context_encoder)
    for param in ema.parameters():
        param.requires_grad = False
    return ema
