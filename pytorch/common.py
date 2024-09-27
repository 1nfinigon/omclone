#!/usr/bin/env python3

import torch

def device():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device('mps')
    if torch.cuda.is_available():
        print("Using CUDA")
        min_d, min_w = None, None
        for i in range(torch.cuda.device_count()):
            d = torch.cuda.device(i)
            w = torch.cuda.power_draw(d)
            if min_d is None or w < min_w:
                min_d, min_d = d, w
        print("Selected device ", min_d)
        return min_d
    print("Using CPU")
    return torch.device('cpu')
