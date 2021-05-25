import torch
from torch import nn
from typing import Union


class Reshape(nn.Module):
    def __init__(self, shape: Union[int, tuple]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, (x.shape[0], *self.shape))


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Permute(nn.Module):
    def __init__(self, dims: tuple):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Squeeze(nn.Module):
    def __init__(self, dims: Union[int, tuple, None]):
        super(Squeeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        return torch.squeeze(x, self.dims)


class SqueezeAll(nn.Module):
    def __init__(self):
        super(SqueezeAll, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class UnSqueeze(nn.Module):
    def __init__(self, dims: Union[int, tuple, None]):
        super(UnSqueeze, self).__init__()
        self.dims = dims

    def forward(self, x):
        return torch.unsqueeze(x, self.dims)


class Flatten(nn.Module):
    def __init__(self, start_dim: int, end_dim: int = -1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        if self.end_dim == -1:
            return x.view(*x.shape[:self.start_dim], -1)
        else:
            return x.view(*x.shape[:self.start_dim], -1, *x.shape[self.end_dim+1:])
