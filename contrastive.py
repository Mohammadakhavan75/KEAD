import numpy as np
import torch
import warnings

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




def contrastive_matrix(data, positive, negative, temperature=0.5, epsilon = 1e-12): # epsilon for non getting devided by zero error
    """
    Compute the contrastive loss for a batch of inputs.

    Args:
        data (Tensor): The input tensor.
        positive (Tensor): The positive tensor (soft augmentation).
        negative (Tensor): The negative tensor (hard augmentation).
        temperature (float, optional): The temperature parameter. Defaults to 0.5.
        epsilon (float, optional): The epsilon value for avoiding division by zero. Defaults to 1e-12.

    Returns:
        Tensor: The contrastive loss.
        Tensor: The similarity between the data and positive tensors.
        Tensor: The similarity between the data and negative tensors.
    """
       
    data_norm = torch.norm(data, p=2, dim=1, keepdim=True)
    negative_norms = torch.norm(negative, p=2, dim=1, keepdim=True)
    positive_norms = torch.norm(positive, p=2, dim=1, keepdim=True)

    # Check for zero norms
    if torch.any(data_norm == 0) or torch.any(positive_norms == 0) or torch.any(negative_norms == 0):
        warnings.warn("Zero norm encountered")

    sim_n = torch.matmul(data, negative.t()) / (data_norm * negative_norms.t() + epsilon)
    sim_p = torch.matmul(data, positive.t()) / (data_norm * positive_norms.t() + epsilon)
   
    sim_p = sim_p.diag() 
    # sim_n = sim_n.diag() 
    # denom = torch.exp(sim_n/temperature) + torch.exp(sim_p/temperature)
    denom = torch.sum(torch.exp(sim_n/temperature), dim=1) + torch.exp(sim_p/temperature)
   
    # card = len(positive[0])
    card = 1
    loss = (-1 / card) * torch.log(torch.exp(sim_p / temperature) / (denom + epsilon))

    return torch.mean(loss), sim_p, sim_n, data_norm, negative_norms, positive_norms # epsilon for non getting devided by zero error
