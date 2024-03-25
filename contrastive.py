import numpy as np
import torch

def norm(x):
    return torch.linalg.vector_norm(x)

def similarity(x, x_prime):
    return x * x_prime / (norm(x) * norm(x_prime))

def cosine_similarity(feat_map1, feat_map2):
  # Flatten the feature maps into vectors
  feat_map1_vec = feat_map1.flatten()
  feat_map2_vec = feat_map2.flatten()

  # Calculate dot product and norms
  dot_product = torch.dot(feat_map1_vec, feat_map2_vec)
  norm1 = torch.linalg.norm(feat_map1_vec, ord=1)
  norm2 = torch.linalg.norm(feat_map2_vec, ord=1)

  # Prevent division by zero
  if norm1 == 0 or norm2 == 0:
    return 0

  cosine_sim = dot_product / (norm1 * norm2)
  return cosine_sim


def contrastive(input, positive, negative, temperature=0.5, epsilon = 1e-9): # epsilon for non getting devided by zero error
    
    sim_n = torch.zeros(negative.shape).to(negative.device)
    sim_p = torch.zeros(positive.shape).to(positive.device)
    if negative.shape[0] != input.shape[0]:
        for j, feature in enumerate(negative):
            sim_n[j] = similarity(input, feature)
    else:
        # sim_n = similarity(input, negative)
        sim_n = cosine_similarity(input, negative)
        
    if positive.shape[0] != input.shape[0]:
        for j, feature in enumerate(positive):
            sim_p[j] = similarity(input, feature)
    else:
        # sim_p = similarity(input, positive)
        sim_p = cosine_similarity(input, positive)
        
    denom = sim_n + sim_p#).to(negative.device) + epsilon # for non getting devided by zero error

    card = positive.shape[0] * positive.shape[1] * positive.shape[2]
    return (- 1/card) * torch.log(torch.sum(torch.exp(sim_p)/temperature, dim=0)/torch.sum(torch.exp(denom)/temperature, dim=0))
