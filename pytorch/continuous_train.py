#!/usr/bin/env python3

import torch
from torch.utils import tensorboard
import numpy as np
from pathlib import Path
import os
import sys
import datetime
import multiprocessing

import model
from common import *
from collections import namedtuple

BATCH_SIZE = 128

NPZData = namedtuple('NPZData', 'spatial_inputs spatiotemporal_inputs temporal_inputs value_outputs policy_outputs pos loss_weights')

class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, n_files, device):
        self.device = device
        dirs = [(int(x.stem), x) for x in Path("test/training_data").glob("*")]
        dirs.sort(key=lambda x: x[0])
        self.files = []
        while dirs and len(self.files) < n_files:
            remaining = n_files - len(self.files)
            (_, dir) = dirs.pop()
            this_files = [(int(x.stem), x) for x in Path(dir).glob("*")]
            this_files.sort(key=lambda x: x[0], reverse=True)
            self.files += [x[1] for x in this_files[:remaining]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        npz_data = np.load(str(self.files[item]))
        def parse_sparse(prefix):
            indices = torch.from_numpy(npz_data['{}_indices'.format(prefix)])
            values = torch.from_numpy(npz_data['{}_values'.format(prefix)])
            size = torch.Size(npz_data['{}_size'.format(prefix)])
            return torch.sparse_coo_tensor(indices, values, size, is_coalesced=True).to_dense()
        data = NPZData(
            spatial_inputs        = parse_sparse('spatial_input'),
            spatiotemporal_inputs = parse_sparse('spatiotemporal_input'),
            temporal_inputs       = parse_sparse('temporal_input'),
            value_outputs         = torch.from_numpy(npz_data['value_output']),
            policy_outputs        = torch.from_numpy(npz_data['policy_output']),
            pos                   = torch.from_numpy(npz_data['pos']),
            loss_weights          = torch.from_numpy(npz_data['loss_weights']),
        )
        return data

if __name__ == "__main__":
    device = device()

    torch.manual_seed(0)

    (model_number, model_path) = max(((int(p.stem), p) for p in Path("test/net").glob("*")), key=lambda x: x[0])
    print("Using model name: {}".format(model_number))

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    training_run_name = 'mainline'
    print("Training run name: {}".format(training_run_name))

    full_set = NPZDataset(250000, device=device)
    training_set, validation_set = torch.utils.data.random_split(full_set, [0.75, 0.25], generator=torch.Generator().manual_seed(42))
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

    model = torch.jit.load(model_path, map_location=device)
    model.to(device)

    def loss_fn(model, pos, model_value_outputs, model_policy_outputs, value_outputs, policy_outputs, loss_weights):
        value_error = 1.5 * (value_outputs * model_value_outputs.clamp(min=1e-3).log()).sum(dim=1)
        B = model_policy_outputs.size()[0]
        model_policy_outputs = torch.stack([model_policy_outputs[b, :, pos[b, 0], pos[b, 1]] for b in range(B)])
        policy_error = 1.0 * (policy_outputs * model_policy_outputs.clamp(min=1e-3).log()).sum(dim=1)
        l2_penalty = 3e-5 * torch.stack([torch.linalg.norm(p) for p in model.parameters() if p.requires_grad]).sum().expand(B)
        return torch.stack([value_error, policy_error, l2_penalty], dim=1) * loss_weights

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    print('Model has {} trainable parameters'.format(sum(p.numel() for p in model.parameters())))

    policy_softmax_temperature = torch.tensor([1.0], device=device)

    def train_one_epoch(tb_writer):
        running_losses = torch.zeros([3], device=device)
        last_losses = None
        n_iterations_since_stats_printed = 0

        for i, data in enumerate(training_loader):
            # Zero your gradients for every batch!
            optimizer.zero_grad(set_to_none=True)

            # Make predictions for this batch
            model_policy_outputs, model_value_outputs = model(
                data.spatial_inputs.to(device, non_blocking=True),
                data.spatiotemporal_inputs.to(device, non_blocking=True),
                data.temporal_inputs.to(device, non_blocking=True),
                policy_softmax_temperature)

            # Compute the loss and its gradients
            losses = loss_fn(model,
                             data.pos.to(device, non_blocking=True),
                             model_value_outputs,
                             model_policy_outputs,
                             data.value_outputs.to(device, non_blocking=True),
                             data.policy_outputs.to(device, non_blocking=True),
                             data.loss_weights.to(device, non_blocking=True)).mean(dim=0)
            loss = losses.sum()
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_losses += losses
            n_iterations_since_stats_printed += 1
            if i % 10 == 0:
                last_losses = (running_losses / n_iterations_since_stats_printed).tolist()
                last_value_loss, last_policy_loss, last_l2_loss = last_losses

                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
                grads_mean = grads.mean().item()
                grads_std = grads.std().item()
                grads_absmax = grads.abs().max().item()
                print('  batch {}/{}   loss: {:.3f} = {:.3f}+{:.3f}+{:.3f}  grad mu={:.2e} sd={:.2e} max={:.2e}'.format(
                    i + 1, len(training_loader),
                    sum(last_losses), last_value_loss, last_policy_loss, last_l2_loss,
                    grads_mean, grads_std, grads_absmax,
                ))
                tb_x = model_number + (i+1) * BATCH_SIZE
                tb_writer.add_scalar('Gradient/Mean/train', grads_mean, tb_x)
                tb_writer.add_scalar('Gradient/Std/train', grads_std, tb_x)
                tb_writer.add_scalar('Gradient/Max/train', grads_absmax, tb_x)
                tb_writer.add_scalar('Loss/Value/train', last_value_loss, tb_x)
                tb_writer.add_scalar('Loss/Policy/train', last_policy_loss, tb_x)
                tb_writer.add_scalar('Loss/L2 penalty/train', last_l2_loss, tb_x)
                tb_writer.add_scalar('Loss/Total/train', sum(last_losses), tb_x)
                running_losses *= 0.
                n_iterations_since_stats_printed = 0

        return sum(last_losses)

    writer = tensorboard.SummaryWriter('test/tensorboard/{}'.format(training_run_name))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            model_policy_outputs, model_value_outputs = model(
                data.spatial_inputs.to(device, non_blocking=True),
                data.spatiotemporal_inputs.to(device, non_blocking=True),
                data.temporal_inputs.to(device, non_blocking=True),
                policy_softmax_temperature)

            losses = loss_fn(model,
                                data.pos.to(device, non_blocking=True),
                                model_value_outputs,
                                model_policy_outputs,
                                data.value_outputs.to(device, non_blocking=True),
                                data.policy_outputs.to(device, non_blocking=True),
                                data.loss_weights.to(device, non_blocking=True)).mean(dim=0)

            loss = losses.sum()
            running_vloss += loss

    avg_vloss = running_vloss / len(validation_loader)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    model_number)
    writer.flush()

    # Save the model's state
    model_path = 'test/net/{}.pt'.format(model_number + len(training_set))
    print("saving to {}".format(model_path))
    model.save(model_path)
