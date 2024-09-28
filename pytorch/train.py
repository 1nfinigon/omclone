#!/usr/bin/env python3

import torch
from torch.utils import tensorboard
import numpy as np
from pathlib import Path
import os
import sys
import datetime
import multiprocessing
import zipfile

import model
from common import *
from collections import namedtuple

BATCH_SIZE = 128
EPOCHS = 50

NPZData = namedtuple('NPZData', 'spatial_inputs spatiotemporal_inputs temporal_inputs value_outputs policy_outputs pos loss_weights')

class NPZDataset(torch.utils.data.Dataset):
    def _zip_load(self):
        self.zipfile = zipfile.ZipFile(self.zip_path)
        self.zipfp = open(self.zip_path, 'rb')

    def _zip_unload(self):
        self.zipfile = None
        self.zipfp = None

    def _npz_get(self, inner_path):
        """
        A faster alternative to using NpzFile, using mmapping
        See: https://github.com/numpy/numpy/issues/5976
        """

        info = self.zipfile.getinfo(inner_path + '.npy')
        assert info.compress_type == 0
        with self.zipfile.open(info) as f:
            offset = f._orig_compress_start

        self.zipfp.seek(offset)
        version = np.lib.format.read_magic(self.zipfp)
        assert version in [(1,0), (2,0)]
        if version == (1,0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(self.zipfp)
        elif version == (2,0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(self.zipfp)
        data_offset = self.zipfp.tell()
        return np.memmap(self.zip_path, dtype=dtype, shape=shape,
                        order='F' if fortran_order else 'C', mode='c',
                        offset=data_offset)

    def __init__(self, zip_path, device):
        self.zip_path = zip_path
        self.device = device
        self._zip_load()
        self.sample_names = list(set((s.split('/')[0] for s in self.zipfile.namelist())))
        # unload for 2 reasons:
        # 1) pickle doesn't support sending open files;
        # 2) save memory in the main process, which doesn't need the full zipfile object
        self._zip_unload()

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, item):
        if self.zipfile is None:
            self._zip_load()

        sample_name = self.sample_names[item]
        def parse_sparse(prefix):
            indices = torch.from_numpy(self._npz_get('{}/{}/indices'.format(sample_name, prefix)))
            values = torch.from_numpy(self._npz_get('{}/{}/values'.format(sample_name, prefix)))
            size = torch.Size(self._npz_get('{}/{}/size'.format(sample_name, prefix)))
            return torch.sparse_coo_tensor(indices, values, size, is_coalesced=True).to_dense()
        def parse_tensor(name):
            return torch.from_numpy(self._npz_get('{}/{}'.format(sample_name, name)))
        data = NPZData(
            spatial_inputs        = parse_sparse('spatial_input'),
            spatiotemporal_inputs = parse_sparse('spatiotemporal_input'),
            temporal_inputs       = parse_sparse('temporal_input'),
            value_outputs         = parse_tensor('value_output'),
            policy_outputs        = parse_tensor('policy_output'),
            pos                   = parse_tensor('pos'),
            loss_weights          = parse_tensor('loss_weights'),
        )
        return data

if __name__ == "__main__":
    device = device()

    torch.manual_seed(0)

    model_name = sys.argv[1]
    print("Using model name {}".format(model_name))

    full_set = NPZDataset('test/next-training.npz', device=device)
    training_set, validation_set = torch.utils.data.random_split(full_set, [0.75, 0.25], generator=torch.Generator().manual_seed(42))
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '{}.pt'.format(model_name))
    model = torch.jit.load(filename, map_location=device)
    model.to(device)

    def loss_fn(model, pos, model_value_outputs, model_policy_outputs, value_outputs, policy_outputs, loss_weights):
        value_error = 1.5 * torch.nn.functional.cross_entropy(model_value_outputs, value_outputs, reduction='none')
        B = model_policy_outputs.size()[0]
        model_policy_outputs = torch.stack([model_policy_outputs[b, :, pos[b, 0], pos[b, 1]] for b in range(B)])
        policy_error = 1.0 * torch.nn.functional.cross_entropy(model_policy_outputs, policy_outputs, reduction='none')
        l2_penalty = 3e-5 * torch.stack([torch.linalg.norm(p) for p in model.parameters() if p.requires_grad]).sum().expand(B)
        return torch.stack([value_error, policy_error, l2_penalty], dim=1) * loss_weights

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    print('Model has {} trainable parameters'.format(sum(p.numel() for p in model.parameters())))

    policy_softmax_temperature = torch.tensor([1.0], device=device)

    def train_one_epoch(epoch_index, tb_writer):
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
                last_losses = running_losses / n_iterations_since_stats_printed # loss per batch
                print('  batch {}/{}   loss: {:.6f} = {:.6f} + {:.6f} + {:.6f}'.format(i + 1, len(training_loader), last_losses.sum().item(), *last_losses.tolist()))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Value loss/train', last_losses[0].item(), tb_x)
                tb_writer.add_scalar('Policy loss/train', last_losses[1].item(), tb_x)
                tb_writer.add_scalar('L2 penalty/train', last_losses[2].item(), tb_x)
                tb_writer.add_scalar('Total loss/train', last_losses.sum().item(), tb_x)
                running_losses *= 0.
                n_iterations_since_stats_printed = 0

        return last_losses.sum().item()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = tensorboard.SummaryWriter('runs/{}_{}'.format(model_name, timestamp))
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
            model_path = '{}_{}_{}'.format(model_name, timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
