#!/usr/bin/env python3

# https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/

import torch
import numpy as np

N_INSTR_TYPES = 11

class Conv2dHex(torch.nn.Module):
    """
    Input:  BIHW or BITHW
    Output: BOHW or BOTHW

    Torchscript: Trace generalizes across B, H, W
    """

    def __init__(self, in_channels, out_channels, kernel_size, has_time_dimension, stride=1, padding=0, dilation=1, groups=1, *args, **kwargs):
        assert(type(has_time_dimension) == bool)
        super().__init__(*args, **kwargs)

        self.has_time_dimension = has_time_dimension

        if has_time_dimension:
            assert(len(kernel_size) == 3)
            m = torch.nn.Conv3d
        else:
            assert(len(kernel_size) == 2)
            m = torch.nn.Conv2d
        self.conv = m(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, *args, **kwargs)

        self.hex_mask = torch.ones_like(self.conv.weight, requires_grad=False)
        min_hex_kernel_size_dim_ = min(kernel_size[-1], kernel_size[-2])
        assert(min_hex_kernel_size_dim_ % 2 == 1)
        for i in range(min_hex_kernel_size_dim_ // 2 + 1):
            for j in range(min_hex_kernel_size_dim_ // 2 - i):
                if has_time_dimension:
                    self.hex_mask[:, :, :, i, -1-j] = 0.
                    self.hex_mask[:, :, :, -1-i, j] = 0.
                else:
                    self.hex_mask[:, :, i, -1-j] = 0.
                    self.hex_mask[:, :, -1-i, j] = 0.

    def forward(self, input):
        if self.has_time_dimension:
            f = torch.nn.functional.conv3d
        else:
            f = torch.nn.functional.conv2d
        return f(
            input, self.conv.weight * self.hex_mask, self.conv.bias, self.conv.stride,
            self.conv.padding, self.conv.dilation, self.conv.groups)

if __name__ == "__main__":
    # test trace generalizability
    C = 2
    input1 = torch.rand(1,C,4,5)
    input2 = torch.rand(6,C,9,10)
    model = Conv2dHex(C, C, (3, 3), has_time_dimension=False)
    assert torch.allclose(torch.jit.trace(model, input1)(input2), model(input2))

    C = 2
    input1 = torch.rand(1,C,3,4,5)
    input2 = torch.rand(6,C,8,9,10)
    model = Conv2dHex(C, C, (1, 3, 3), has_time_dimension=True)
    assert torch.allclose(torch.jit.trace(model, input1)(input2), model(input2))


if __name__ == "__main__":
    # test gradient correctness
    C = 1
    input = torch.ones(1, C, 3, 3, requires_grad=True)
    model = Conv2dHex(C, C, (3, 3), has_time_dimension=False, padding='same')
    output = model(input)
    assert(output.size() == (1, 1, 3, 3))
    output_centre = output[0, 0, 1, 1]
    output_centre.backward(inputs=[input])
    assert(input.grad.count_nonzero() == 7)

    C = 1
    input = torch.ones(1, C, 1, 3, 3, requires_grad=True)
    model = Conv2dHex(C, C, (1, 3, 3), has_time_dimension=True, padding='same')
    output = model(input)
    assert(output.size() == (1, 1, 1, 3, 3))
    output_centre = output[0, 0, 0, 1, 1]
    output_centre.backward(inputs=[input])
    assert(input.grad.count_nonzero() == 7)

class GlobalPool2d(torch.nn.Module):
    """
    Input:  BCHW or BCTHW
    Output: BX11 or BXT11 (X=2*C)

    Torchscript: Trace generalizes across H, W
    """

    def __init__(self, in_channels, has_time_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_time_dimension = has_time_dimension
        self.in_channels = in_channels
        self.out_channels = 2 * self.in_channels

    def forward(self, input):
        B = input.shape[0]
        C = input.shape[1]
        if not torch.jit.is_tracing():
            assert(C == self.in_channels)
        if self.has_time_dimension:
            T = input.shape[2]
            max = torch.max(input.view(B, C, T, -1), dim=-1).values.view(B, C, T, 1, 1)
        else:
            max = torch.max(input.view(B, C, -1), dim=-1).values.view(B, C, 1, 1)
        mean = torch.mean(input, dim=(-1, -2), keepdim=True)
        return torch.cat((max, mean), dim=1)

if __name__ == "__main__":
    # test trace generalizability
    C = 2
    input1 = torch.rand(1,C,3,4,5)
    input2 = torch.rand(1,C,3,14,15)
    model = GlobalPool2d(C, has_time_dimension=True)
    output = model(input2)
    assert(output.size() == (1,2*C,3,1,1))
    traced_output = torch.jit.trace(model, input1)(input2)
    assert torch.allclose(traced_output, output)

    C = 2
    input1 = torch.rand(1,C,4,5)
    input2 = torch.rand(1,C,14,15)
    model = GlobalPool2d(C, has_time_dimension=False)
    output = model(input2)
    assert(output.size() == (1,2*C,1,1))
    traced_output = torch.jit.trace(model, input1)(input2)
    assert torch.allclose(traced_output, output)

if __name__ == "__main__":
    # test correctness
    def gen_wh(n):
        return [[n, n-1], [n-2, n-3]]
    c0t0 = gen_wh(4)
    c0t1 = gen_wh(8)
    c1t0 = gen_wh(104)
    c1t1 = gen_wh(108)
    input = torch.tensor([[ [ c0t0, c0t1 ], [ c1t0, c1t1 ]]], dtype=torch.float)
    model = GlobalPool2d(C, has_time_dimension=True)
    output = model(input)
    c0t0max, c0t0mean = [[4]], [[2.5]]
    c0t1max, c0t1mean = [[8]], [[6.5]]
    c1t0max, c1t0mean = [[104]], [[102.5]]
    c1t1max, c1t1mean = [[108]], [[106.5]]
    expected_output = torch.tensor([[[c0t0max, c0t1max], [c1t0max, c1t1max], [c0t0mean, c0t1mean], [c1t0mean, c1t1mean]]], dtype=torch.float)
    assert(torch.allclose(output, expected_output))

class GlobalPoolBiasStructure2d(torch.nn.Module):
    """
    Input:       BCHW or BCTHW
    Pool input:  BCHW or BPTHW
    Output:      BCHW or BCTHW

    Torchscript: Trace generalizes across B, T, H, W
    """

    def __init__(self, in_channels, pool_channels, has_time_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.pool_channels = pool_channels
        self.has_time_dimension = has_time_dimension

        if has_time_dimension:
            self.bn_pool = torch.nn.BatchNorm3d(pool_channels)
        else:
            self.bn_pool = torch.nn.BatchNorm2d(pool_channels)
        self.pool = GlobalPool2d(pool_channels, has_time_dimension)
        self.fc_pool = torch.nn.Linear(self.pool.out_channels, in_channels)

    def forward(self, input, pool_input):
        if not torch.jit.is_tracing():
            assert(input.shape[1] == self.in_channels)
            assert(pool_input.shape[1] == self.pool_channels)
        bias = self.bn_pool(pool_input)
        bias = self.pool(bias)
        bias = bias.squeeze(dim=(-1,-2))
        if self.has_time_dimension:
            bias = bias.permute((0, 2, 1))
        bias = self.fc_pool(bias)
        if self.has_time_dimension:
            bias = bias.permute((0, 2, 1))
        bias = bias.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return input + bias

if __name__ == "__main__":
    # test trace generalizability
    C = 2
    P = 6
    input1 = torch.rand(1, C, 3, 4, 5)
    pool1 = torch.rand(1, P, 3, 4, 5)
    input2 = torch.rand(7, C, 9, 10, 11)
    pool2 = torch.rand(7, P, 9, 10, 11)
    model = GlobalPoolBiasStructure2d(C, P, has_time_dimension=True)
    output = model(input2, pool2)
    traced_output = torch.jit.trace(model, (input1, pool1))(input2, pool2)
    assert torch.allclose(traced_output, output)

class PreActivationResBlock(torch.nn.Module):
    """
    Input:       BCTHW
    Output:      BCTHW

    Torchscript: Trace generalizes across B, T, H, W
    """

    def __init__(self, channels, pool_channels=None, time_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.bn1 = torch.nn.BatchNorm3d(channels)
        self.relu1 = torch.nn.ReLU()
        self.conv1 = Conv2dHex(channels, channels, (1, 3, 3), has_time_dimension=True, padding='same')

        if pool_channels is not None:
            self.pool_channels = pool_channels
            self.bias_structure = GlobalPoolBiasStructure2d(channels - pool_channels, pool_channels, has_time_dimension=True)
            post_pool_channels = channels - pool_channels
        else:
            self.pool_channels = None
            self.bias_structure = None
            post_pool_channels = channels

        self.bn2 = torch.nn.BatchNorm3d(post_pool_channels)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = Conv2dHex(post_pool_channels, channels, (1, 3, 3), has_time_dimension=True, padding='same')

    def forward(self, input):
        res = input
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.conv1(res)
        if self.bias_structure is not None:
            pool_input, res = torch.split(res, [self.pool_channels, self.channels - self.pool_channels], dim=1)
            res = self.bias_structure(res, pool_input)
        res = self.bn2(res)
        res = self.relu2(res)
        res = self.conv2(res)
        return input + res

if __name__ == "__main__":
    # test trace generalizability
    C = 8
    input1 = torch.rand(1, C, 3, 4, 5)
    input2 = torch.rand(7, C, 9, 10, 11)

    model = PreActivationResBlock(C)
    output = model(input2)
    traced_output = torch.jit.trace(model, input1)(input2)
    assert torch.allclose(traced_output, output)

    P = 2
    model = PreActivationResBlock(C, pool_channels=P)
    output = model(input2)
    traced_output = torch.jit.trace(model, input1)(input2)
    assert torch.allclose(traced_output, output)

class InputEmbedder(torch.nn.Module):
    """
    Spatial:        BSHW   (S = # spatial features)
    Spatiotemporal: BXTHW  (X = # spatiotemporal features)
    Temporal:       BGT    (G = # temporal features)
    Output:         BCTHW  (C = # channels)

    Torchscript: Trace generalizes across B, T, H, W

    Here's how the embedding works:

    BSHW  --7x7--> BCHW -> BC1HW
    BXTHW -1x7x7-> BCTHW
    BGT -> BTG -FC-> BTC -> BCT -> BCT11
    """

    def __init__(self, spatial_features, spatiotemporal_features, temporal_features,
                 channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spatial_conv = Conv2dHex(spatial_features, channels, (7, 7), has_time_dimension=False, padding='same')
        self.spatiotemporal_conv = Conv2dHex(spatiotemporal_features, channels, (1, 7, 7), has_time_dimension=True, padding='same')
        self.temporal_fc = torch.nn.Linear(temporal_features, channels)

    def forward(self, spatial_input, spatiotemporal_input, temporal_input):
        spatial = self.spatial_conv(spatial_input).unsqueeze(dim=2)
        spatiotemporal = self.spatiotemporal_conv(spatiotemporal_input)
        temporal = self.temporal_fc(temporal_input.permute((0, 2, 1))).permute((0, 2, 1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return spatial + spatiotemporal + temporal

if __name__ == "__main__":
    # test trace generalizability
    S = 2
    X = 3
    G = 4
    C = 5

    B = 1
    H = 6
    W = 7
    T = 8
    spatial1 = torch.rand(B, S, H, W)
    spatiotemporal1 = torch.rand(B, X, T, H, W)
    temporal1 = torch.rand(B, G, T)

    B = 11
    H = 16
    W = 17
    T = 18
    spatial2 = torch.rand(B, S, H, W)
    spatiotemporal2 = torch.rand(B, X, T, H, W)
    temporal2 = torch.rand(B, G, T)

    model = InputEmbedder(S, X, G, C)
    output = model(spatial2, spatiotemporal2, temporal2)
    assert(output.size() == (B, C, T, H, W))
    traced_output = torch.jit.trace(model, (spatial1, spatiotemporal1, temporal1))(spatial2, spatiotemporal2, temporal2)
    assert torch.allclose(traced_output, output)

class Trunk(torch.nn.Module):
    """
    Input:  BCTHW
    Output: BCHW

    Torchscript: Trace generalizes across B, H, W
    """

    def __init__(self, channels, pool_channels, time_size, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_size = time_size

        self.resblocks = torch.nn.Sequential()
        for layer_spec in layers:
            if layer_spec == "respool":
                layer = PreActivationResBlock(channels, pool_channels=pool_channels)
            elif layer_spec == "res":
                layer = PreActivationResBlock(channels)
            elif layer_spec == "convtime":
                layer = torch.nn.Conv3d(channels, channels, (time_size, 1, 1))
            else:
                raise RuntimeError("unknown layer_spec " + layer_spec)
            self.resblocks.append(layer)

    def forward(self, input):
        if not torch.jit.is_tracing():
            assert(input.size()[2] == self.time_size)
        input = self.resblocks(input)
        if not torch.jit.is_tracing():
            assert(input.size()[2] == 1)
        return input.squeeze(dim=2)

if __name__ == "__main__":
    # test trace generalizability
    B = 1
    C = 2
    T = 3
    H = 4
    W = 5
    input1 = torch.rand(B, C, T, H, W)

    B = 11
    H = 14
    W = 15
    input2 = torch.rand(B, C, T, H, W)

    model = Trunk(C, C // 2, T, ["res", "respool", "res", "convtime", "res", "respool", "res"])
    output = model(input2)
    assert(output.size() == (B, C, H, W))
    traced_output = torch.jit.trace(model, input1)(input2)
    assert torch.allclose(traced_output, output)

class PolicyHead(torch.nn.Module):
    """
    Input:  BCHW
    Output: BNHW (N = N_INSTR_TYPES)

    Torchscript: Trace generalizes across B, H, W
    """

    def __init__(self, trunk_channels, head_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trunk_channels = trunk_channels
        self.head_channels = head_channels

        self.conv_in = torch.nn.Conv2d(trunk_channels, head_channels, 1)
        self.conv_pool = torch.nn.Conv2d(trunk_channels, head_channels, 1)
        self.bias_structure = GlobalPoolBiasStructure2d(head_channels, head_channels, has_time_dimension=False)

        self.bn_out = torch.nn.BatchNorm2d(self.head_channels)
        self.relu_out = torch.nn.ReLU()
        self.conv_out = torch.nn.Conv2d(head_channels, N_INSTR_TYPES, 1)

    def forward(self, input):
        pool_input = self.conv_pool(input)
        input = self.conv_in(input)
        input = self.bias_structure(input, pool_input)
        input = self.bn_out(input)
        input = self.relu_out(input)
        input = self.conv_out(input)
        return input

if __name__ == "__main__":
    # test trace generalizability
    B = 1
    C = 6
    H = 4
    W = 5
    N = 2
    input1 = torch.rand(B, C, H, W)

    B = 11
    H = 14
    W = 15
    input2 = torch.rand(B, C, H, W)

    model = PolicyHead(C, N)
    output = model(input2)
    assert(output.size() == (B, N_INSTR_TYPES, H, W))
    traced_output = torch.jit.trace(model, input1)(input2)
    assert torch.allclose(traced_output, output)

class ValueHead(torch.nn.Module):
    """
    Input:  BCHW
    Output: B3

    Torchscript: Trace generalizes across B, H, W
    """

    def __init__(self, trunk_channels, head_channels, value_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trunk_channels = trunk_channels
        self.head_channels = head_channels
        self.value_channels = value_channels

        self.conv_in = torch.nn.Conv2d(trunk_channels, head_channels, 1)
        self.global_pool = GlobalPool2d(head_channels, has_time_dimension=False)
        self.fc_pre = torch.nn.Linear(self.global_pool.out_channels, value_channels)
        self.relu = torch.nn.ReLU()
        self.fc_post = torch.nn.Linear(value_channels, 3)

    def forward(self, input):
        input = self.conv_in(input)
        input = self.global_pool(input)
        input = input.squeeze(dim=(-1, -2))
        input = self.fc_pre(input)
        input = self.relu(input)
        input = self.fc_post(input)
        return input

class Heads(torch.nn.Module):
    def __init__(self, trunk_channels, policy_head_channels, value_head_channels, value_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.policy_head = PolicyHead(trunk_channels, policy_head_channels, *args, **kwargs)
        self.value_head = ValueHead(trunk_channels, value_head_channels, value_channels, *args, **kwargs)

    def forward(self, input):
        policy = self.policy_head(input)
        value = self.value_head(input)
        (policy, value)

if __name__ == "__main__":
    # test trace generalizability
    B = 1
    C = 6
    H = 4
    W = 5
    input1 = torch.rand(B, C, H, W)

    B = 11
    H = 14
    W = 15
    input2 = torch.rand(B, C, H, W)

    model = ValueHead(C, 4, 2)
    output = model(input2)
    assert(output.size() == (B, 3))
    traced_output = torch.jit.trace(model, input1)(input2)
    assert torch.allclose(traced_output, output)



the_model = torch.nn.Sequential()
CHANNELS = 16
HEAD_CHANNELS = 16
VALUE_CHANNELS = 8
the_model.append(InputEmbedder(spatial_features=143,
                               spatiotemporal_features=72,
                               temporal_features=1,
                               channels=CHANNELS))
the_model.append(Trunk(channels=CHANNELS,
                       pool_channels=C // 2,
                       time_size=T,
                       layers=[
                           "res",
                           "respool",
                           "res",
                           "convtime",
                           "res",
                           "respool",
                           "res"
                       ]))
the_model.append(Heads(trunk_channels=CHANNELS,
                       policy_head_channels=HEAD_CHANNELS,
                       value_head_channels=HEAD_CHANNELS,
                       value_channels=VALUE_CHANNELS))
print(the_model)

model_parameters = filter(lambda p: p.requires_grad, the_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("{} trainable parameters".format(params))
