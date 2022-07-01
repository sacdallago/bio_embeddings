import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(x, std):
    pi = torch.tensor(math.pi)
    s2 = 2.0*torch.tensor(std).square()
    x2 = torch.tensor(x).square().neg()

    return torch.exp(x2 / s2) * torch.rsqrt(s2 * pi)


def gaussian_kernel(kernel_size, std=1.0):
    kernel = [gaussian(i - (kernel_size // 2), std)
              for i in range(kernel_size)]

    kernel = torch.tensor(kernel)
    kernel = kernel / kernel.sum()

    return kernel


class SeqNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, affine=True):
        super().__init__()

        if affine:
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

        self.register_buffer(name='eps', tensor=torch.tensor(float(eps)))
        self.register_buffer(name='channels', tensor=torch.tensor(channels))

    def forward(self, x, mask):
        mask_rsum = 1.0 / (mask.sum(dim=(2, 3), keepdims=True) * self.channels)

        x = x * mask

        mean = x.sum(dim=(1, 2, 3), keepdims=True) * mask_rsum

        x = (x - mean) * mask

        var = x.square().sum(dim=(1, 2, 3), keepdims=True) * mask_rsum

        x = x * torch.rsqrt(var + self.eps)

        if self.weight is not None:
            x = (x * self.weight) + self.bias

        return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super().__init__()

        self.func = nn.ReLU(inplace=True)

        self.norm = SeqNorm(channels=out_channels,
                            eps=1e-6, affine=True)

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, 1),
                              stride=(stride, 1),
                              padding=(padding, 0),
                              dilation=(dilation, 1),
                              groups=groups,
                              bias=False,
                              padding_mode=padding_mode)

        # Init Conv Params
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x, mask):
        x = self.func(self.norm(self.conv(x), mask))

        return (x * mask)


class CNN(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.input = Conv(1024, channels, 1, 1, 0)

        self.dwc1 = Conv(channels, channels, 9, 1, 4, groups=channels)
        self.dwc2 = Conv(channels, channels, 21, 1, 10, groups=channels)

        self.dropout = nn.Dropout2d(p=0.50, inplace=True)

        self.output = nn.Conv2d(3*channels, 5, 1, 1, 0)

        # Init Output Params
        nn.init.zeros_(self.output.bias)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x, mask):
        x = self.input(x, mask)

        z1 = self.dwc1(x, mask)
        z2 = self.dwc2(x, mask)

        x = torch.cat([x, z1, z2], dim=1)

        x = self.dropout(x)

        x = self.output(x)

        return (x * mask)


class TmbedModel(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.model = CNN(channels)

        filter_kernel = gaussian_kernel(kernel_size=7, std=1.0)

        self.register_buffer(name='filter_kernel', tensor=filter_kernel)

    def forward(self, x, mask):
        B, N, C = x.shape

        mask = mask.view(B, 1, N, 1)

        x = x.transpose(1, 2).view(B, C, N, 1)

        x = self.model(x, mask)

        x = x.view(B, 5, N)

        x = F.pad(x, pad=(3, 3), mode='constant', value=0.0)

        x = x.unfold(dimension=2, size=7, step=1)

        x = torch.einsum('bcnm,m->bcn', x, self.filter_kernel)

        return x
