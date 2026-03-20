import torch


@torch.no_grad()
def update_ema(online_model, ema_model, decay: float = 0.998):
    """
    Update EMA model parameters: ema = decay * ema + (1 - decay) * online.

    LoRA-aware: when LoRA adapters are present, only updates parameters whose
    names contain 'lora_' (the trainable adapter weights). Base backbone weights
    are identical and frozen in both models — updating them is a no-op but the
    zip() approach would silently include them. Name-based filtering is explicit
    and correct.

    Falls back to updating all parameters when no LoRA params are found
    (e.g., use_lora=False ablation runs).
    """
    online_named = dict(online_model.named_parameters())
    ema_named    = dict(ema_model.named_parameters())
    lora_names   = [n for n in online_named if 'lora_' in n]

    if lora_names:
        for name in lora_names:
            ema_named[name].data.mul_(decay).add_(
                online_named[name].data, alpha=1.0 - decay
            )
    else:
        # Fallback for non-LoRA runs (use_lora=False)
        for ep, op in zip(ema_model.parameters(), online_model.parameters()):
            ep.data.mul_(decay).add_(op.data, alpha=1.0 - decay)
