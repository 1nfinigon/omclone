#!/usr/bin/env python3

import model
import numpy as np
import torch
import os

model.run_tests()

torch.manual_seed(0)

the_model = model.model_v1_params_v1()

model_parameters = filter(lambda p: p.requires_grad, the_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Model has {} trainable parameters".format(params))

# construct a sample input, for tracing

dtype = torch.float
spatial = torch.rand((2, SPATIAL_FEATURES, 1, 1), dtype=dtype)
spatiotemporal = torch.rand((2, SPATIOTEMPORAL_FEATURES, N_HISTORY_CYCLES, 1, 1), dtype=dtype)
temporal = torch.rand((2, TEMPORAL_FEATURES, 1), dtype=dtype)
policy_softmax_temperature = torch.Tensor([1])

the_model.eval()
traced_model = torch.jit.trace(the_model, (spatial, spatiotemporal, temporal, policy_softmax_temperature))

filename = "test/net/init.pt"
print("Saved model to {}".format(filename))
traced_model.save(filename)
