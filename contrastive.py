import numpy as np
import torch

def norm(x):
    return torch.linalg.vector_norm(x)

def similarity(x, x_prime):
    return x * x_prime / (norm(x) * norm(x_prime))

def contrastive(input, positive, negative):
    epsilon = 1e-6 # for non getting devided by zero error
    sim_n = torch.zeros(negative.shape).to(negative.device)
    sim_p = torch.zeros(positive.shape).to(positive.device)
    if negative.shape[0] != input.shape[0]:
        for j, feature in enumerate(negative):
            sim_n[j] = similarity(input, feature)
    else:
        sim_n = similarity(input, negative)
    if positive.shape[0] != input.shape[0]:
        for j, feature in enumerate(positive):
            sim_p[j] = similarity(input, feature)
    else:
        sim_p = similarity(input, positive)
        
    denom = torch.cat([sim_n, sim_p]).to(negative.device) + epsilon # for non getting devided by zero error

    # for non getting devided by zero error
    return (- 1/torch.abs(positive + epsilon)) * torch.log(torch.sum(torch.exp(sim_p), dim=0)/torch.sum(torch.exp(denom), dim=0))
