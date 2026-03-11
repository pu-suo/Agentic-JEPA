import torch


@torch.no_grad()
def update_ema(online_model, ema_model, decay: float = 0.998):
    """Update EMA model parameters: ema = decay * ema + (1 - decay) * online."""
    for ema_param, online_param in zip(ema_model.parameters(), online_model.parameters()):
        ema_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)
