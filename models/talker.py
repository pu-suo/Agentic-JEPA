"""
Talker Module — converts latent states into discrete tokens.

Trained AFTER the Predictor is fully trained and frozen (Stage 3 / post-curriculum).
Uses standard cross-entropy loss to reconstruct action text from latent states.

At inference, includes an AST-based syntax validator:
- If the generated code fails ast.parse(), retry with higher temperature
- After N retries, trigger a latent-level backtrack (not a token-level fix)
"""
import ast
import torch
import torch.nn as nn
from transformers import AutoTokenizer


class Talker(nn.Module):
    def __init__(self, d_model: int = 768, vocab_size: int = 50265,
                 n_layers: int = 2, n_heads: int = 4, max_tokens: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens

        # Project latent state to initial decoder hidden state
        self.latent_proj = nn.Linear(d_model, d_model)

        # Lightweight autoregressive decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Token embedding and output head
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, latent_state: torch.Tensor, target_tokens: torch.Tensor):
        """
        Training forward pass.

        Args:
            latent_state: (batch, d_model) frozen latent representation
            target_tokens: (batch, seq_len) ground-truth token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Create memory from latent state
        memory = self.latent_proj(latent_state).unsqueeze(1)  # (batch, 1, d_model)

        # Embed target tokens (teacher-forced)
        tgt_embed = self.token_embed(target_tokens)  # (batch, seq_len, d_model)

        # Causal mask
        seq_len = target_tokens.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        causal_mask = causal_mask.to(latent_state.device)

        # Decode
        decoded = self.decoder(tgt_embed, memory, tgt_mask=causal_mask)
        logits = self.output_head(decoded)

        return logits

    @torch.no_grad()
    def generate(self, latent_state: torch.Tensor, tokenizer,
                 temperature: float = 1.0, max_tokens: int = None) -> str:
        """
        Autoregressive generation from a latent state.

        Args:
            latent_state: (1, d_model) latent representation (single sample)
            tokenizer: HuggingFace tokenizer
            temperature: sampling temperature
            max_tokens: override max token count

        Returns:
            generated_text: decoded string
        """
        max_len = max_tokens or self.max_tokens
        device = latent_state.device

        memory = self.latent_proj(latent_state).unsqueeze(1)  # (1, 1, d_model)

        # Start with BOS token
        bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
        generated = [bos_id]

        for _ in range(max_len):
            input_ids = torch.tensor([generated], device=device)
            tgt_embed = self.token_embed(input_ids)

            seq_len = input_ids.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            decoded = self.decoder(tgt_embed, memory, tgt_mask=causal_mask)
            logits = self.output_head(decoded[:, -1, :])  # last token logits

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == eos_id:
                break

        return tokenizer.decode(generated[1:], skip_special_tokens=True)


def validate_syntax(code_string: str) -> bool:
    """Check if generated code is valid Python via AST parsing."""
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        return False
