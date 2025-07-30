import torch

def compute_per_dim_std(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute per-dimension standard deviation of the given embeddings.

    Args:
        embeddings: Tensor of shape (N, D) where N = # upright samples, D = embedding dim.
        eps: small value to avoid sqrt of zero.

    Returns:
        std: Tensor of shape (D,) containing σ_d for each embedding dimension.
    """
    # Use the population variance (unbiased=False) for stability
    var = embeddings.var(dim=0, unbiased=False)  # shape (D,)
    std = torch.sqrt(var + eps)
    return std