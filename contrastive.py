import numpy as np
import torch

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



def contrastive(input, positive, negative, temperature=0.5, epsilon = 1e-9): # epsilon for non getting devided by zero error
    
    sim_n = torch.zeros(negative.shape).to(negative.device)
    sim_p = torch.zeros(positive.shape).to(positive.device)
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
        
    denom = torch.exp(sim_n/temperature) + torch.exp(sim_p/temperature) + epsilon # for non getting devided by zero error

    if positive.shape[0] != input.shape[0]:
        card = len(positive)
    else:
        card = 1
    
    return (- 1/card) * torch.log(torch.sum(torch.exp(sim_p/temperature), dim=0)/torch.sum(denom, dim=0))
