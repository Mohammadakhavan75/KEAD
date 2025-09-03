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


def contrastive_matrix(
    rep_a: torch.Tensor,
    rep_n: torch.Tensor,
    temperature: float,
    eps: float = 1e-8,
):
    """
    In-batch positives (requested): rep_p is None
    - Positives = *all other anchors* (off-diagonals in rep_a @ rep_a^T).
    - Negatives = all rows in rep_n.

    Returns:
        con_loss, sim_p_mean, sim_n_mean, norm_a, norm_n
    """
    device = rep_a.device
    T = temperature

    # L2-normalize
    rep_a = F.normalize(rep_a, p=2, dim=1)
    norm_a = rep_a.norm(p=2, dim=1).mean().detach()

    
    rep_n = F.normalize(rep_n, p=2, dim=1)
    norm_n = rep_n.norm(p=2, dim=1).mean().detach()
    
    B = rep_a.size(0)
    
    # anchor-anchor sims
    S_aa = (rep_a @ rep_a.t()) / T        # (B, B)
    # mask out self as positive (we want off-diagonals)
    offdiag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
    pos_logits = S_aa[offdiag_mask].view(B, B - 1)   # (B, B-1)

    # anchor-negative sims
    S_an = (rep_a @ rep_n.t()) / T               # (B, M)
    all_logits = torch.cat([pos_logits, S_an], dim=1)  # (B, (B-1)+M)

    # log-sum-exp numerator/denominator (stable)
    max_logits, _ = all_logits.max(dim=1, keepdim=True)
    all_logits_stable = all_logits - max_logits

    log_num = torch.logsumexp(all_logits_stable[:, :pos_logits.size(1)], dim=1)  # positives only
    log_den = torch.logsumexp(all_logits_stable, dim=1)                          # pos + neg

    con_loss = -(log_num - log_den).mean()

    # diagnostics
    sim_p_mean = pos_logits.mean().detach()
    sim_n_mean = S_an.mean().detach() if S_an is not None and S_an.numel() > 0 \
                    else torch.tensor(float("nan"), device=device)

    return con_loss, sim_p_mean, sim_n_mean, norm_a, norm_n