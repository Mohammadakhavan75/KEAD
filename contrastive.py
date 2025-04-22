import numpy as np
import torch
import warnings
import torch.nn.functional as F
def norm(x):
    return torch.linalg.vector_norm(x)


def similarity(x, x_prime):
    return x * x_prime / (norm(x) * norm(x_prime))


def cosine_similarity(feature_map1, feature_map2):
    # Flatten the feature maps to treat them as vectors
    feature_map1_flat = feature_map1.flatten()
    feature_map2_flat = feature_map2.flatten()

    # Calculate the dot product and norms
    # dot_product = np.sum(feature_map1_flat * feature_map2_flat)
    dot_product = torch.dot(feature_map1_flat, feature_map2_flat)
    norm1 = torch.linalg.norm(feature_map1_flat)
    norm2 = torch.linalg.norm(feature_map2_flat)

    # Prevent division by zero
    if norm1 == 0 or norm2 == 0:
        return torch.tensor(0).to(feature_map1.device)

    # Cosine similarity
    cosine_similarity_map = dot_product / (norm1 * norm2)
    return cosine_similarity_map



def contrastive(input, positive, negative, temperature=0.5, epsilon = 1e-12): # epsilon for non getting devided by zero error
    
    sim_n = torch.zeros(negative.shape[0]).to(negative.device)
    sim_p = torch.zeros(positive.shape[0]).to(positive.device)
    if negative.shape[0] != input.shape[0]:
        for j, feature in enumerate(negative):
            sim_n[j] = cosine_similarity(input, feature)
    else:
        # sim_n = similarity(input, negative)
        sim_n = cosine_similarity(input, negative)
        
    if positive.shape[0] != input.shape[0]:
        for j, feature in enumerate(positive):
            sim_p[j] = cosine_similarity(input, feature)
    else:
        # sim_p = similarity(input, positive)
        sim_p = cosine_similarity(input, positive)

    denom = torch.exp(sim_n/temperature) + torch.exp(sim_p/temperature)

    if positive.shape[0] != input.shape[0]:
        card = len(positive)
    else:
        card = 1
    
    return (- 1/card) * torch.log(torch.sum(torch.exp(sim_p/temperature), dim=0)/(torch.sum(denom, dim=0) + epsilon)), sim_p, sim_n # epsilon for non getting devided by zero error




import torch
import warnings
from typing import Optional

def contrastive_matrix(
    anchors: torch.Tensor,          # (B, D)
    positives: torch.Tensor,        # (B·Kₚ, D)  or (B, D) if Kₚ=1
    negatives: torch.Tensor,        # (B·Kₙ, D)  or arbitrary (M, D) if Kₙ=None
    temperature: float = 0.5,
    epsilon: float = 1e-12,
    positives_per_anchor: Optional[int] = None,     # Kₚ, or None → infer
    negatives_per_anchor: Optional[int] = None,     # Kₙ, or None → treat all M as negatives
):
    """
    General NT‑Xent with Kₚ positives and Kₙ negatives per anchor.

    If negatives_per_anchor is None, every entry in `negatives`
    is used for every anchor (SimCLR‑like).
    """

    B = anchors.shape[0]
    D = anchors.shape[1]
    device = anchors.device

    # ---------------- l2 norms ----------------
    norm_a = torch.norm(anchors,   p=2, dim=1, keepdim=True)    # (B,1)
    norm_p = torch.norm(positives, p=2, dim=1, keepdim=True)    # (B·Kₚ,1) or (B,1)
    norm_n = torch.norm(negatives, p=2, dim=1, keepdim=True)    # (B·Kₙ,1) or (M,1)

    if (norm_a == 0).any() or (norm_p == 0).any() or (norm_n == 0).any():
        warnings.warn("Zero‑norm row(s) detected; cosine similarity undefined there.")

        # Clamp norms to avoid division by zero
        norm_a = torch.clamp(norm_a, min=epsilon)
        norm_p = torch.clamp(norm_p, min=epsilon)
        norm_n = torch.clamp(norm_n, min=epsilon)

    # ---------------- cosine similarities ----------------
    sim_p_full = anchors @ positives.t() / (norm_a * norm_p.t() + epsilon)   # (B, B·Kₚ) or (B, B)
    sim_n_full = anchors @ negatives.t() / (norm_a * norm_n.t() + epsilon)   # (B, B·Kₙ) or (B, M)

    # ----------- extract Kₚ positives for each anchor ------------
    if positives_per_anchor is None:
        positives_per_anchor = positives.shape[0] // B                       # infers 1 or Kₚ
    Kp = positives_per_anchor

    if Kp > 1:
        row_offset = (torch.arange(B, device=device) * Kp).unsqueeze(1)      # (B,1)
        col_index  = torch.arange(Kp, device=device).unsqueeze(0)            # (1,Kₚ)
        idx_p      = row_offset + col_index                                  # (B,Kₚ)
        sim_p = sim_p_full.gather(1, idx_p)                                  # (B,Kₚ)
    else:                                                                    # Kₚ = 1
        sim_p = sim_p_full.diag().unsqueeze(1)                               # (B,1)

    # ----------- extract Kₙ negatives for each anchor ------------
    if negatives_per_anchor is None:                                         # use ALL negatives
        # sim_n = sim_n_full                                                   # (B,M)
        # Kn    = sim_n.shape[1]                                               # M
        # exclude the “self‐negative” n_i for anchor x_i
        mask = ~torch.eye(B, dtype=torch.bool, device=device)   # (B,B)
        sim_n = sim_n_full[mask].view(B, B-1)                   # (B, B−1)
        Kn    = B - 1
    else:
        Kn = negatives_per_anchor
        row_offset = (torch.arange(B, device=device) * Kn).unsqueeze(1)      # (B,1)
        col_index  = torch.arange(Kn, device=device).unsqueeze(0)            # (1,Kₙ)
        idx_n      = row_offset + col_index                                  # (B,Kₙ)
        sim_n      = sim_n_full.gather(1, idx_n)                             # (B,Kₙ)

    # ---------------- denominator ----------------
    denom = (
        torch.sum(torch.exp(sim_p / temperature), dim=1, keepdim=True) +
        torch.sum(torch.exp(sim_n / temperature), dim=1, keepdim=True)
    )                                                                        # (B,1)

    # ---------------- loss ----------------
    loss_per_positive = (
        -torch.log(torch.exp(sim_p / temperature) / (denom + epsilon))
    ) / Kp                                                                    # (B,Kₚ) or (B,1)

    loss = loss_per_positive.mean()

    return loss, sim_p, sim_n, norm_a, norm_n, norm_p
