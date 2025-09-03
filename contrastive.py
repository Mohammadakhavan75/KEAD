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


def nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
    eps: float = 1e-8,
):
    """
    NT-Xent loss (SimCLR/CSI): given two augmented views per sample.

    Args:
        z1: tensor of shape (B, D) - projection head outputs for view 1
        z2: tensor of shape (B, D) - projection head outputs for view 2
        temperature: temperature scalar (tau)
        eps: small constant for numerical stability

    Returns:
        loss: scalar NT-Xent loss averaged over 2B anchors
        sim_p_mean: mean cosine similarity over positive pairs
        sim_n_mean: mean cosine similarity over all negatives
        raw_norm_1: mean L2 norm of z1 before normalization (diagnostic)
        raw_norm_2: mean L2 norm of z2 before normalization (diagnostic)
    """
    device = z1.device
    B = z1.size(0)

    # Keep raw norms for diagnostics
    raw_norm_1 = z1.norm(p=2, dim=1).mean().detach()
    raw_norm_2 = z2.norm(p=2, dim=1).mean().detach()

    # L2-normalize
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Cosine similarity matrix across 2B vectors
    z = torch.cat([z1, z2], dim=0)   # (2B, D)
    sim = (z @ z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarities for denominator
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)

    # Stabilize
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_stable = sim - sim_max

    # Denominator: sum over all except self
    exp_sim = torch.exp(sim_stable) * (~self_mask).float()
    log_den = torch.log(exp_sim.sum(dim=1) + eps)  # (2B,)

    # Numerator: positive pair for each anchor
    # For i in [0..B-1], pos is (i, i+B); for i in [B..2B-1], pos is (i, i-B)
    pos_logits = torch.cat([
        torch.diag(sim_stable, B),
        torch.diag(sim_stable, -B)
    ], dim=0)  # (2B,)

    loss = -(pos_logits - log_den).mean()

    # Diagnostics: positive and negative cosine similarity means
    # Using unit-normalized z => cosine similarity = dot product
    pos_sims = (z1 * z2).sum(dim=1)  # (B,)
    sim_p_mean = pos_sims.mean().detach()

    with torch.no_grad():
        sim_full = z @ z.t()
        neg_mask = (~self_mask).clone()
        # remove positive-pair positions from negatives for mean calc
        pos_mask_upper = torch.eye(B, device=device, dtype=torch.bool)
        pos_mask = torch.zeros_like(neg_mask)
        pos_mask[:B, B:] = pos_mask_upper
        pos_mask[B:, :B] = pos_mask_upper
        neg_mask = neg_mask & (~pos_mask)
        sim_n_mean = sim_full[neg_mask].mean() if neg_mask.any() else torch.tensor(float('nan'), device=device)
        sim_n_mean = sim_n_mean.detach()

    return loss, sim_p_mean, sim_n_mean, raw_norm_1, raw_norm_2
