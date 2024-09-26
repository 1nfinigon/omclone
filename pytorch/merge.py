#!/usr/bin/env python3

import torch
import os
import sys

basedir = os.path.dirname(os.path.realpath(__file__))

model = torch.jit.load(os.path.join(basedir, 'model.pt'))
state = torch.load(sys.argv[1], weights_only=True)
model.load_state_dict(state)

model.save(os.path.join(basedir, 'merged.pt'))
