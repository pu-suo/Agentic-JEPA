"""
Encoder modules: Context Encoder, EMA Target Encoder, Observation Encoder.

All encoders output L2-normalized vectors on S^(d-1).
The Target Encoder is an EMA copy of the Context Encoder.
"""
import copy
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from utils.math_utils import l2_normalize


class TextEncoder(nn.Module):
    """
    Frozen pre-trained encoder that maps text -> S^(d-1).
    Used as the Context Encoder and Observation Encoder.
    """
    def __init__(self, model_name: str = "microsoft/codebert-base",
                 d_model: int = 768, freeze: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Projection head: backbone_dim -> d_model (if different)
        backbone_dim = self.backbone.config.hidden_size
        if backbone_dim != d_model:
            self.projection = nn.Linear(backbone_dim, d_model)
        else:
            self.projection = nn.Identity()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
    """Create an EMA copy of the context encoder. All params frozen."""
    ema = copy.deepcopy(context_encoder)
    for param in ema.parameters():
        param.requires_grad = False
    return ema
