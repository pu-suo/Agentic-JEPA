from dataclasses import dataclass


@dataclass
class AgenticJEPAConfig:
    # === Embedding ===
    d_model: int = 768              # Latent dimension (must match encoder output)

    # === Encoder ===
    encoder_name: str = "microsoft/codebert-base"  # HuggingFace model ID
    freeze_encoder: bool = True     # Freeze base CodeBERT weights; LoRA adapters remain trainable
    ema_decay: float = 0.998        # EMA decay rate for target encoder (LoRA weights only)

    # === LoRA (Low-Rank Adaptation for CodeBERT) ===
    use_lora: bool = True           # Inject trainable LoRA adapters into frozen backbone
    lora_r: int = 8                 # LoRA rank (~1M trainable params on CodeBERT)
    lora_alpha: int = 16            # LoRA scaling factor (effective scale = alpha/r = 2.0)
    lora_target_modules: tuple = ("query", "value")  # RoBERTa attention projections
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    # === Afterstate Predictor ===
    predictor_layers: int = 4       # Number of Transformer decoder layers
    predictor_heads: int = 8        # Attention heads
    predictor_ff_dim: int = 2048    # Feed-forward intermediate dimension
    predictor_dropout: float = 0.1

    # === ACT (Adaptive Computation Time) ===
    act_max_steps: int = 8          # K_max: maximum latent reasoning steps
    act_halt_bias_init: float = 1.0 # Initial bias for halting unit (encourages more compute initially)

    # === SLERP Gate ===
    slerp_gate_hidden: int = 256    # Hidden dim of gate MLP
    slerp_gate_bias_init: float = -3.0  # CRITICAL: σ(-3.0) ≈ 0.047, near-zero fusion at start

    # === Value Head ===
    value_hidden: int = 256

    # === Loss Weights (vary by curriculum stage) ===
    # Stage 0: lambda_jepa=1.0, lambda_v=0.0, lambda_ponder=0.0
    # Stage 1: lambda_jepa=1.0, lambda_v=0.1,  lambda_ponder=0.0
    # Stage 2: lambda_jepa=1.0, lambda_v=0.5,  lambda_ponder=0.05
    lambda_jepa: float = 1.0
    lambda_v: float = 0.0
    lambda_ponder: float = 0.0
    lambda_obs_utility: float = 0.2   # Weight for obs-utility loss (Stage 1+). Amplifies gate gradient.

    # === Curriculum Transitions ===
    stage0_plateau_patience: int = 3    # Consecutive evals with < epsilon improvement
    stage0_plateau_epsilon: float = 0.005
    stage1_alpha_threshold: float = 0.08  # Peak α over recent window must exceed this (natural peak ~0.092)
    stage1_alpha_window: int = 5          # Number of recent evals to look back for peak α
    stage1_jepa_tolerance: float = 0.05  # JEPA loss must be within 5% of Stage 0 plateau
    stage1_max_evals: int = 200           # Time-based fallback: advance to Stage 2 after this many evals

    # === Training ===
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_epochs_per_stage: int = 50
    eval_every_n_steps: int = 100
    gradient_clip: float = 1.0

    # === Talker ===
    talker_layers: int = 4
    talker_heads: int = 4
    talker_max_tokens: int = 512

    # === Inference ===
    mcts_branches: int = 5          # k: number of discrete actions to consider
    delta_fatal: float = -0.5       # Default; calibrate on validation set after training
    talker_retries: int = 3         # Syntax validation retries before latent backtrack

    # === Data ===
    num_trajectories: int = 1000
    max_seq_len: int = 512
    val_split: float = 0.1
