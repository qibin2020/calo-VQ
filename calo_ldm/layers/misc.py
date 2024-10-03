import torch
from torch import nn
import torch.nn.functional as F

class FlattenVoxels(nn.Module):
    ''' squash the last three (R,Z,A) dimensions '''
    def forward(self, x):
        return x.flatten(start_dim=-3)


class VoxelSoftmax(nn.Module):
    ''' do softmax over all voxels (last three dims), casting to
        float64 to avoid numerical loss of precision '''
    def forward(self, x):
        x = torch.exp(x.double())
        denom = x.sum(axis=(-1,-2,-3), keepdims=True)
        return x/denom


class FlatVoxelSoftmax(nn.Module):
    def forward(self, x):
        x = torch.exp(x.double())
        denom = x.sum(axis=(-1), keepdims=True)
        return (x/denom).float()


class VoxelReluExpm1Max(nn.Module):
    def forward(self, x):
        x = torch.relu(x)
        x = torch.expm1(x.double())
        denom = x.sum(axis=(-1,-2,-3), keepdims=True).clip(min=1e-9)
        return (x/denom).float()

class VoxelReluExpm1MaxD6(nn.Module):
    def forward(self, x):
        x = torch.relu(x)
        x = torch.expm1(x.double())
        denom = x.sum(axis=(-1,-2,-3), keepdims=True).clip(max=1e6,min=1e-6)
        return (x/denom).float()

class FlatVoxelReluExpm1Max(nn.Module):
    def forward(self, x):
        x = torch.relu(x)
        x = torch.expm1(x.double())
        denom = x.sum(axis=(-1), keepdims=True).clip(min=1e-9)
        return (x/denom).float()


class FlatVoxelCeluExpm1Max(nn.Module):
    def forward(self, x):
        x = F.celu(x)+1
        x = torch.expm1(x.double())
        denom = x.sum(axis=(-1), keepdims=True).clip(min=1e-9)
        return (x/denom).float()


class LogScale(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.register_buffer('a', torch.tensor(a).float())
        self.register_buffer('b', torch.tensor(b).float())
        self.register_buffer('c', torch.tensor(c).float())

    def forward(self, x):
        return torch.log1p(self.a + self.b*x) / self.c


class ZPad(nn.Module):
    def __init__(self, z_pad):
        super().__init__()
        assert len(z_pad) == 2
        self.z_pad = z_pad

    def forward(self, x):
        return F.pad(x, (0,0) + tuple(self.z_pad))


class ZUnPad(nn.Module):
    def __init__(self, z_pad):
        super().__init__()
        assert len(z_pad) == 2
        #self.z_pad = z_pad
        self.left = z_pad[0]
        self.right = -z_pad[1] if z_pad[1] else None

    def forward(self, x):
        return x[:,:,self.left:self.right,:]

class Dummy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.p=nn.Identity()
        
    def forward(self, x, c=None):
        return self.p(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x