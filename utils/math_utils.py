import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project onto unit hypersphere S^(d-1)."""
    return F.normalize(x, p=2, dim=dim)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity. Both inputs assumed L2-normalized."""
    return 1.0 - (a * b).sum(dim=-1)


def safe_slerp(v0: torch.Tensor, v1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Spherical Linear Interpolation on S^(d-1) with numerical safety.

    v0, v1: (..., d) tensors, assumed L2-normalized
    alpha: (..., 1) tensor in [0, 1]

    Returns: L2-normalized interpolated vector on S^(d-1)

    CRITICAL: This function must handle three edge cases:
    1. v0 ≈ v1 (parallel, Ω → 0): sin(Ω) → 0, division by zero
    2. v0 ≈ -v1 (antipodal, Ω → π): sin(Ω) → 0, division by zero
    3. Numerical drift from fp16/bf16 training

    Falls back to normalized linear interpolation when sin(Ω) is too small.
    """
    # Clamp dot product to avoid acos domain errors
    dot = torch.clamp((v0 * v1).sum(dim=-1, keepdim=True), -0.9999, 0.9999)
    omega = torch.acos(dot)  # angle between vectors
    sin_omega = torch.sin(omega)

    # SLERP coefficients
    coeff0 = torch.sin((1.0 - alpha) * omega) / sin_omega
    coeff1 = torch.sin(alpha * omega) / sin_omega
    slerp_result = coeff0 * v0 + coeff1 * v1

    # Linear interpolation fallback (for near-parallel or near-antipodal)
    lerp_result = (1.0 - alpha) * v0 + alpha * v1
    lerp_result = l2_normalize(lerp_result)

    # Use SLERP when sin_omega is large enough, fallback otherwise
    use_slerp = (sin_omega.abs() > 1e-6).float()
    result = use_slerp * slerp_result + (1.0 - use_slerp) * lerp_result

    return l2_normalize(result)  # safety re-normalization
