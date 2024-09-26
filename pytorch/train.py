#!/usr/bin/env python3

import torch
from torch.utils import tensorboard
import numpy as np
from pathlib import Path
import os
import datetime

import model
from common import *

BATCH_SIZE = 8
EPOCHS = 5

class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.files = list(Path(path).glob('*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        items = np.load(str(self.files[item]))
        torch_items = dict()
        for key, value in items.items():
            torch_items[key] = torch.from_numpy(value).to(self.device)
        return torch_items

device = device()

full_set = NPZDataset('test/next-training', device=device)
training_set, validation_set = torch.utils.data.random_split(full_set, [0.75, 0.25])
training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE)

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pt')
model = torch.jit.load(filename, map_location=device)
model.to(device)

def loss_fn(model, pos, model_value_outputs, model_policy_outputs, value_outputs, policy_outputs):
    value_error = torch.nn.functional.cross_entropy(model_value_outputs, value_outputs)
    B = model_policy_outputs.size()[0]
    model_policy_outputs = torch.stack([model_policy_outputs[b, :, pos[b, 0], pos[b, 1]] for b in range(B)])
    policy_error = torch.nn.functional.cross_entropy(model_policy_outputs, policy_outputs)
    l2_penalty = sum([torch.linalg.norm(p) for p in model.parameters() if p.requires_grad])
    return 1.5 * value_error + policy_error + 3e-5 * l2_penalty

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print('Model has {} trainable parameters'.format(sum(p.numel() for p in model.parameters())))

policy_softmax_temperature = torch.tensor([1.0], device=device)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    n_iterations_since_stats_printed = 0

    for i, data in enumerate(training_loader):
        spatial_inputs        = data['spatial_input']
        spatiotemporal_inputs = data['spatiotemporal_input']
        temporal_inputs       = data['temporal_input']
        value_outputs         = data['value_output']
        policy_outputs        = data['policy_output']
        pos                   = data['pos']

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        model_policy_outputs, model_value_outputs = model(spatial_inputs, spatiotemporal_inputs, temporal_inputs, policy_softmax_temperature)

        # Compute the loss and its gradients
        loss = loss_fn(model, pos, model_value_outputs, model_policy_outputs,
                       value_outputs, policy_outputs)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        n_iterations_since_stats_printed += 1
        if i % 10 == 0:
            last_loss = running_loss / n_iterations_since_stats_printed # loss per batch
            print('  batch {}/{} loss: {}'.format(i + 1, len(training_loader), last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            n_iterations_since_stats_printed = 0

    return last_loss

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = tensorboard.SummaryWriter('runs/{}'.format(timestamp))
epoch_number = 0

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            spatial_inputs        = data['spatial_input']
            spatiotemporal_inputs = data['spatiotemporal_input']
            temporal_inputs       = data['temporal_input']
            value_outputs         = data['value_output']
            policy_outputs        = data['policy_output']
            pos                   = data['pos']

            model_policy_outputs, model_value_outputs = model(spatial_inputs, spatiotemporal_inputs, temporal_inputs, policy_softmax_temperature)
            loss = loss_fn(model, pos, model_value_outputs, model_policy_outputs,
                        value_outputs, policy_outputs)
            running_vloss += loss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
