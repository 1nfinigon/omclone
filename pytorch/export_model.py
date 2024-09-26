#!/usr/bin/env python3

import model
import numpy as np
import torch
import os

model.run_tests()

# constants from nn.rs
SPATIAL_FEATURES = 141
SPATIOTEMPORAL_FEATURES = 72
TEMPORAL_FEATURES = 2
N_HISTORY_CYCLES = 4

# tunable constants
CHANNELS = 96
POOL_CHANNELS = 32
HEAD_CHANNELS = 96
VALUE_CHANNELS = 48
LAYERS = [
    "res",
    "respool",
    "res",
    "convtime",
    "res",
    "respool",
    "res"
]

torch.manual_seed(0)

the_model = model.ModelV1(
    spatial_features = SPATIAL_FEATURES,
    spatiotemporal_features = SPATIOTEMPORAL_FEATURES,
    temporal_features = TEMPORAL_FEATURES,
    time_size = N_HISTORY_CYCLES,
    trunk_channels = CHANNELS,
    pool_channels = POOL_CHANNELS,
    trunk_layers = LAYERS,
    policy_head_channels = HEAD_CHANNELS,
    value_head_channels = HEAD_CHANNELS,
    value_channels = VALUE_CHANNELS)

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

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pt')
print("Saved model to {}".format(filename))
traced_model.save(filename)
