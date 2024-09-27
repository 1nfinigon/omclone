#!/usr/bin/env python3

import torch

def device():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device('mps')
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device('cuda')
    print("Using CPU")
    return torch.device('cpu')
