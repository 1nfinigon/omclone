#!/usr/bin/env python3

import torch
import sys

if len(sys.argv) != 4:
    print("usage: {} MODEL_WITH_CODE MODEL_WITH_WEIGHTS OUTPUT".format(sys.argv[0]))
    sys.exit(1)

model1 = torch.jit.load(sys.argv[1])
model2 = torch.jit.load(sys.argv[2])
assert([p.numel() for p in model1.parameters()] == [p.numel() for p in model2.parameters()])
model1.load_state_dict(model2.state_dict())
model1.save(sys.argv[3])
