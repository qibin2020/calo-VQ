import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import partial

# return a tensor w/ random values from the trangular distribution
def triangular(a, b, c, size, device=None):
    u = torch.rand(size=size, device=device)
    f = (c-a)/(b-a)
    left = a + torch.sqrt(u * (b-a) * (c-a))
    right = b - torch.sqrt((1-u) * (b-a) * (b-c))
    return torch.where(u < f, left, right)

class FFTDownsample(nn.Module):
    def __init__(self, n_drop=2, fft_dim=2, phase_dithering=False):
        super().__init__()
        assert n_drop > 0 and n_drop%2==0
        self.n_drop = n_drop
        if fft_dim==2:
            self._fft = torch.fft.rfft2
            self._ifft = torch.fft.irfft2
        elif fft_dim==1:
            self._fft = torch.fft.rfft
            self._ifft = torch.fft.irfft
        else:
            raise ValueError("fft_dim should be 1 or 2")
        self.phase_dithering = phase_dithering

    def forward(self, x): # R Z A ?
        # x (*, Z, R)
        xhat = self._fft(x)[...,:-self.n_drop//2] # (*, Z, n)
        if self.training and self.phase_dithering:
            width = math.pi/x.shape[-1]
            phi = triangular(-width, 0, width, size=x.shape[:-2], device=x.device)[...,None,None] # (*,1,1)
            nfreq = torch.arange(xhat.shape[-1], device=xhat.device).reshape((1,)*len(x.shape[:-2]) + (1,-1)) # (*,1,n)
            xhat = xhat * torch.exp(1j * phi)
        return self._ifft(xhat)

class FFTInterpolate(nn.Module):
    def __init__(self, n_fill, fft_dim=2):
        super().__init__()
        assert n_fill > 0 and n_fill%2==0
        self.n_fill = n_fill
        if fft_dim == 2:
            self._fft = torch.fft.rfft2
            self._ifft = torch.fft.irfft2
        elif fft_dim == 1:
            self._fft = torch.fft.rfft
            self._ifft = torch.fft.irfft
        else:
            raise ValueError("fft_dim should be 1 or 2")

    def forward(self, x):
        xhat = self._fft(x, dim=-1)
        return self._ifft(F.pad(xhat, (0,self.n_fill//2)), dim=-1)

# implement new fft
# class DownsampleFFT70(nn.Module):
#     def __init__(self, dim, RZA, compress_RZA=(True,True,True)):
#         super().__init__()
#         assert sum(compress_RZA)>0
#         compress_dim=[]
#         if compress_RZA[0]:
#             compress_dim.append(-3)
#         if compress_RZA[1]:
#             compress_dim.append(-2)
#         if compress_RZA[2]:
#             compress_dim.append(-1)
#         self.compress_RZA=compress_RZA
#         self.compress_dim=compress_dim
#         self.fft=partial(torch.fft.fftn,dim=compress_dim)
#         self.ifft=partial(torch.fft.ifftn,dim=compress_dim)

#         self.drop_R=None if not compress_RZA[0] else int(RZA[0]*0.3)//2
#         self.drop_Z=None if not compress_RZA[1] else int(RZA[1]*0.3)//2
#         self.drop_A=None if not compress_RZA[2] else int(RZA[2]*0.3)//2

#         self.idrop_R=None if not compress_RZA[0] else -(int(RZA[0]*0.3)//2)
#         self.idrop_Z=None if not compress_RZA[1] else -(int(RZA[1]*0.3)//2)
#         self.idrop_A=None if not compress_RZA[2] else -(int(RZA[2]*0.3)//2)

#     def forward(self, x): # R Z A 
#         xhat = torch.fft.fftshift(self.fft(x),dim=self.compress_dim)[...,self.drop_R:self.idrop_R,self.drop_Z:self.idrop_Z,self.drop_A:self.idrop_A] 
#         return self.ifft(torch.fft.ifftshift(xhat,dim=self.compress_dim)).real
    
#     def extra_repr(self):
#         return f"compress_RZA={self.compress_RZA}, compress_dim={self.compress_dim}"

# class UpsampleFFT70(nn.Module):
#     def __init__(self, dim, raw_RZA, compress_RZA=(True,True,True)):
#         super().__init__()
#         assert sum(compress_RZA)>0
#         compress_dim=[]
#         if compress_RZA[0]:
#             compress_dim.append(-3)
#         if compress_RZA[1]:
#             compress_dim.append(-2)
#         if compress_RZA[2]:
#             compress_dim.append(-1)
#         self.compress_RZA=compress_RZA
#         self.compress_dim=compress_dim
#         self.fft=partial(torch.fft.fftn,dim=compress_dim)
#         self.ifft=partial(torch.fft.ifftn,dim=compress_dim)
#         self.drop_R=0 if not compress_RZA[0] else int(raw_RZA[0]*0.3)//2
#         self.drop_Z=0 if not compress_RZA[1] else int(raw_RZA[1]*0.3)//2
#         self.drop_A=0 if not compress_RZA[2] else int(raw_RZA[2]*0.3)//2

#     def forward(self, x): # R Z A 
#         xhat = torch.fft.fftshift(self.fft(x),dim=self.compress_dim)
#         xhatpad = F.pad(xhat, (self.drop_A,self.drop_A,self.drop_Z,self.drop_Z,self.drop_R,self.drop_R))
#         return self.ifft(torch.fft.ifftshift(xhatpad,dim=self.compress_dim)).real
    
#     def extra_repr(self):
#         return f"compress_RZA={self.compress_RZA}, compress_dim={self.compress_dim}"

class FFTDownsampleV2(nn.Module):
    def __init__(self, n_drop=2, fft_dim=3, phase_dithering=None):
        super().__init__()
        assert n_drop > 0 and n_drop%2==0
        self.n_drop = n_drop
        if fft_dim==3:
            compress_dim=(-3,-2,-1)
        elif fft_dim==2:
            compress_dim=(-2,-1)
        elif fft_dim==1:
            compress_dim=(-1,)
        else:
            raise ValueError("fft_dim should be 1 or 2 or 3")
        self.compress_dim=compress_dim
        self.fft=partial(torch.fft.fftn,dim=compress_dim)
        self.ifft=partial(torch.fft.ifftn,dim=compress_dim)
        self.drop_A=self.n_drop//2
        self.idrop_A=-self.n_drop//2

    def forward(self, x): # R Z A ?
        xhat = torch.fft.fftshift(self.fft(x),dim=self.compress_dim)[...,self.drop_A:self.idrop_A] 
        return self.ifft(torch.fft.ifftshift(xhat,dim=self.compress_dim)).real

class FFTInterpolateV2(nn.Module):
    def __init__(self, n_fill, fft_dim=3):
        super().__init__()
        assert n_fill > 0 and n_fill%2==0
        self.n_fill = n_fill
        if fft_dim==3:
            compress_dim=(-3,-2,-1)
        elif fft_dim==2:
            compress_dim=(-2,-1)
        elif fft_dim==1:
            compress_dim=(-1,)
        else:
            raise ValueError("fft_dim should be 1 or 2 or 3")
        self.compress_dim=compress_dim
        self.fft=partial(torch.fft.fftn,dim=compress_dim)
        self.ifft=partial(torch.fft.ifftn,dim=compress_dim)
        self.drop_A=self.n_fill//2

    def forward(self, x):
        xhat = torch.fft.fftshift(self.fft(x),dim=self.compress_dim)
        xhatpad = F.pad(xhat, (self.drop_A,self.drop_A,0,0,0,0))
        return self.ifft(torch.fft.ifftshift(xhatpad,dim=self.compress_dim)).real