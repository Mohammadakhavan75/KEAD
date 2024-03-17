import numpy as np
import torch

def norm(x):
    return torch.linalg.vector_norm(x)

def similarity(x, x_prime):
    return x * x_prime / (norm(x) * norm(x_prime))

def contrastive(input, positive, negative):
    sim_n = torch.zeros(negative.shape).to(negative.device)
    sim_p = torch.zeros(positive.shape).to(positive.device)
    for j, feature in enumerate(negative):
        sim_n[j] = similarity(input, feature)
    for j, feature in enumerate(positive):
        sim_p[j] = similarity(input, feature)
    
    denom = torch.cat([sim_n, sim_p]).to(negative.device)

    return (- 1/torch.abs(positive)) * torch.log(torch.sum(torch.exp(sim_p), dim=0)/torch.sum(torch.exp(denom), dim=0))
