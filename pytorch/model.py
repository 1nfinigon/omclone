#!/usr/bin/env python3

# https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/

import torch

N_INSTRUCTIONS = 11

class Conv1dTime2dHex(torch.nn.Conv3d):
    """
    Input:  BCTHW
    Output: BCTHW

    Torchscript: Trace generalizes across shape
    """

    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        assert(len(kernel_size) == 3)
        super().__init__(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.hex_mask = torch.ones_like(self.weight)
        min_hex_kernel_size_dim_ = min(kernel_size[-1], kernel_size[-2])
        assert(min_hex_kernel_size_dim_ % 2 == 1)
        for i in range(min_hex_kernel_size_dim_ // 2 + 1):
            for j in range(min_hex_kernel_size_dim_ // 2 - i):
                self.hex_mask[:, :, i, -1-j] = 0.
                self.hex_mask[:, :, -1-i, j] = 0.

    def forward(self, input):
        return torch.nn.functional.conv2d(
            input, self.weight * self.hex_mask, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

class GlobalPool1dTime2d(torch.nn.Module):
    """
    Input:  BCTHW
    Output: BXT11 (X=2*C)

    Torchscript: Trace generalizes but not across shape
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = 2 * self.in_channels

    def forward(self, input):
        B = input.shape[0]
        C = input.shape[1]
        T = input.shape[2]
        assert(C == self.in_channels)
        max = torch.max(input.view(B, C, T, -1), dim=3).view(B, C, T, 1, 1)
        mean = torch.mean(input, dim=(3, 4), keepdim=True)
        return torch.cat((max, mean), dim=1)

class GlobalPoolBiasStructure1dTime2d(torch.nn.Module):
    """
    Input:       BCTHW
    Pool input:  BPTHW
    Output:      BCTHW
    """

    def __init__(self, in_channels, pool_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.pool_channels = pool_channels

        self.bn_pool = torch.nn.BatchNorm3d(pool_channels)
        self.pool = GlobalPool1DTime2d(pool_channels)
        self.fc_pool = torch.nn.Linear(self.pool.out_channels, in_channels)

    def forward(self, input, pool_input):
        assert(input.shape[1] == self.in_channels)
        assert(pool_input.shape[1] == self.pool_channels)
        bias = self.bn_pool(pool_input)
        bias = self.pool(bias)
        bias = bias.view(-1, self.fc_pool.in_features)
        bias = self.fc_pool(bias)
        bias = bias.view(-1, self.fc_pool.out_features, 1, 1, 1)
        input += bias
        return input

class PreActivationResBlock(torch.nn.Module):
    """
    Input:       BCTHW
    Output:      BCTHW
    """

    def __init__(self, channels, pool_channels=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.bn1 = torch.nn.BatchNorm3d(channels)
        self.relu1 = torch.nn.ReLU()
        self.conv1 = Conv1dTime2dHex(channels, channels, (1, 3, 3))

        if pool_channels is not None:
            self.pool_channels = pool_channels
            self.bias_structure = GlobalPoolBiasStructure1dTime2d(channels - pool_channels, pool_channels)

        self.bn2 = torch.nn.BatchNorm3d(channels)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = Conv1dTime2dHex(channels, channels, (1, 3, 3))

    def forward(self, input):
        input = self.bn1(input)
        input = self.relu1(input)
        input = self.conv1(input)
        if self.bias_structure is not None:
            pool_input, input = torch.split(input, [self.pool_channels, self.channels - self.pool_channels], dim=1)
            input = self.bias_structure(input, pool_input)
        input = self.bn2(input)
        input = self.relu2(input)
        input = self.conv2(input)
        return input

class Trunk(torch.nn.Module):
    def __init__(self, input_features, global_features, layers, pool_layers, channels, pool_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.pool_channels = pool_channels

        self.input_conv1 = torch.nn.Conv3d(input_features, channels, (1, 1, 1))
        self.input_conv7 = Conv1dTime2dHex(channels, channels, (1, 7, 7))
        self.global_fc = torch.nn.Linear(global_features, channels)

        self.resblocks = torch.nn.Sequential()
        is_pool_layer = [False for i in range(layers)]
        for pl in range(pool_layers):
            is_pool_layer[int(float(pl + 1) * float(layers) / float(pool_layers + 1))] = True
        assert(sum(is_pool_layer) == pool_layers)
        for is_pool_layer in is_pool_layer:
            if is_pool_layer:
                layer = PreActivationResBlock(channels, pool_channels)
            else:
                layer = PreActivationResBlock(channels)
            self.resblocks.append(layer)

    def forward(self, input_features, global_features):
        input_features = self.input_conv1(input_features)
        input_features = self.input_conv7(input_features)
        global_features = self.global_fc(global_features)
        global_features = global_features.view(-1, self.channels, 1, 1)
        input = input_features + global_features
        input = self.resblocks(input)
        return input

class PolicyHead(torch.nn.Module):
    def __init__(self, trunk_channels, head_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trunk_channels = trunk_channels
        self.head_channels = head_channels

        self.conv_in = torch.nn.Conv2d(trunk_channels, head_channels, 1)
        self.conv_pool = torch.nn.Conv2d(trunk_channels, head_channels, 1)
        self.bias_structure = GlobalPoolBiasStructure2d(head_channels, head_channels)

        self.bn_out = torch.nn.BatchNorm2d(self.head_channels)
        self.relu_out = torch.nn.ReLU()
        self.conv_out = torch.nn.Conv2d(head_channels, 1, N_INSTRUCTIONS)

    def forward(self, input):
        pool_input = self.conv_pool(input)
        input = self.conv_in(input)
        input = self.bias_structure(input, pool_input)
        input = self.bn_out(input)
        input = self.relu_out(input)
        input = self.conv_out(input)
        return input

class ValueHead(torch.nn.Module):
    def __init__(self, trunk_channels, head_channels, value_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trunk_channels = trunk_channels
        self.head_channels = head_channels
        self.value_channels = value_channels

        self.conv_in = torch.nn.Conv2d(trunk_channels, head_channels, 1)
        self.global_pool = GlobalPool2d(head_channels)
        self.fc_pre = torch.nn.Linear(self.global_pool.out_channels, value_channels)
        self.relu = torch.nn.ReLU()
        self.fc_post = torch.nn.Linear(value_channels, 3)

    def forward(self, input):
        input = self.conv_in(input)
        input = self.global_pool(input)
        input = input.squeeze(dims=(2, 3))
        input = self.fc_pre(input)
        input = self.relu(input)
        input = self.fc_post(input)
        return input
