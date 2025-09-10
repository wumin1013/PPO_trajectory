import torch
import numpy as np

def compute_advantage(gamma, lmbda, td_delta):
    """
    Compute advantage using GAE (Generalized Advantage Estimation)
    """
    advantage = torch.zeros_like(td_delta)
    advantage_last = 0
    
    # Backward computation
    for t in reversed(range(len(td_delta))):
        advantage[t] = td_delta[t] + gamma * lmbda * advantage_last
        advantage_last = advantage[t]
    
    return advantage