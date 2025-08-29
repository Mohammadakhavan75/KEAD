import torch
import torch.nn.functional as F

def variance_floor(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg-style variance floor: encourages each embedding dimension to have
    standard deviation at least gamma, preventing collapse of the upright cloud.

    Args:
        z: Tensor of shape (N, D) -- raw embeddings (will be normalized internally).
        gamma: target minimum std per dimension (after normalization, 1.0 is typical).
        eps: small constant added inside sqrt for numerical stability.

    Returns:
        scalar loss: mean over dimensions of max(0, gamma - std_d)^2
    """
    # Normalize each vector to unit length so that target gamma=1.0 makes sense.
    # z = F.normalize(z, dim=1)                     # (N, D)
    # Per-dimension standard deviation across the batch
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)  # (D,)
    # Penalize only if std < gamma
    loss = torch.mean(torch.relu(gamma - std) ** 2)
    return loss, std  # return std for logging/diagnostics