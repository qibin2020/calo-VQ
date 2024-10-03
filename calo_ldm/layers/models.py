import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .conv2 import CylinderConv2 as CylinderConv
from .conv2 import CylinderConvTranspose2 as CylinderConvTranspose
from .layers import VoxelSoftmax, FlattenVoxels, Sigmoid2

# maps lowercase strings to activations
_activation_aliases = {
    'relu':         nn.ReLU,
    'silu':         nn.SiLU,
    'swish':        nn.SiLU, # common alias
    'elu':          nn.ELU,
    'celu':         nn.CELU,
    'softplus':     nn.Softplus,
    'voxel_softmax': VoxelSoftmax,
    'sigmoid2': Sigmoid2,
}

def _make_activation(name):
    if name is None:
        return None
    try:
        return _activation_aliases[name.lower()]()
    except KeyError:
        raise ValueError(f"Unsupported activation: {name}")

def _parse_spec(spec):
    # spec format: s12:k53:p -> stride(z=2,a=1), k(z=5,a=3), pad_z=True
    #              k3:s13   -> stride(1,3), k(3,3), pad_z=False
    #              p:k35    -> stride(1,1) k(3,5), pad_z=True
    # order and case insensitive.

    s = (1,1)
    k = None
    pad_z = False
    for x in spec.lower().split(':'):
        if x.startswith('s'): s=tuple(map(int, x[1:]))
        if x.startswith('k'): k=tuple(map(int, x[1:]))
        if x == 'p': pad_z = True
    if k is None:
        raise ValueError(f"Invalid specification string: {spec}")
    if len(s) == 1: s = 2*s
    if len(k) == 1: k = 2*k

    return k, s, pad_z


class Encoder(pl.LightningModule):
    def __init__(self, *,
            ch_in, # number of radial calo layers
            ch_out, # channel dimension of the output feature map
            ch, # base number of hidden channels
            ch_mult, # channel size multiplier per layer
            spec, # specifier strings for the layers
            dim_cond = 1, # dimension of the conditional input
            activation='swish', # hidden activation
            output_activation=None, # final output activation
            factor1=1.,
            factor2=3000.,
            factor3=7.,
            ):
        super().__init__()

        if dim_cond not in (0,1):
            raise NotImplementedError("dim_cond>1 is possible but haven't dealt with that case yet.")
        
        self.activation = _make_activation(activation)
        self.output_activation = _make_activation(output_activation)

        # if ch_mult is given as a single int, just repeat it as a tuple
        # with 1 at the beginning (so the first layer outputs size ch)
        if isinstance(ch_mult, int):
            ch_mult = (1,) + (ch_mult,)*(len(spec)-1)
        assert len(ch_mult) == len(spec)

        w_in = ch_in + dim_cond # input has R + C channels
        w_out = ch # base number of output channels
        self.layers = nn.Sequential()
        for mul,sp in zip(ch_mult, spec):
            w_out *= mul
            k, s, p = _parse_spec(sp)
            self.layers.append(CylinderConv(w_in, w_out, k=k, stride=s, pad_z=p))
            w_in = w_out
            self.layers.append(self.activation)

        # final output to reshape to latent dim
        self.layers.append(CylinderConv(w_in, ch_out, k=(1,1), stride=(1,1), pad_z=False))

        if self.output_activation is not None:
            self.layers.append(self.output_activation)
        
        self.factor1=torch.tensor(factor1*1.)
        self.factor2=torch.tensor(factor2*1.)
        self.factor3=torch.tensor(factor3*1.)

    def forward(self, x, cond=None):
        # x (*, R, Z, A)
        # cond (*, 1)
        # assumes x is normalized to e_inc, and cond is logscaled.

        x = torch.log(self.factor1+self.factor2*x)/self.factor3

        if cond is not None:
            # broadcast the condition variable across the input image
            xc = cond[:,None,None].expand((-1,-1,x.shape[-2],x.shape[-1])) # (*, 1, Z, A)
            x = torch.concat([x,xc], axis=-3) # (*, R+1, Z, A)

        out = self.layers(x)
        return out


class Decoder(pl.LightningModule):
    def __init__(self, *,
            ch_in, # channel dimension of the latent space feature map
            ch_out, # number of radial calo layers
            ch,
            ch_div,
            spec,
            dim_cond=1, # dimension of the conditional input. if zero, means don't use condition input
            activation='swish',
            output_activation='calo_softmax',
            learn_R=True,
            R_activation='softplus',
            R_trim=1e-2, # small number to add to the output of R to prevent 0
            ):

        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out

        self.register_buffer('R_trim', torch.tensor(R_trim))
        self.register_buffer('learn_R', torch.tensor(learn_R))

        if not dim_cond in (0,1):
            raise NotImplementedError("dim_cond>1 is possible but haven't dealt with that case yet.")
        
        self.activation = _make_activation(activation)
        self.output_activation = _make_activation(output_activation)

        # if ch_div is given as a single int, just repeat it as a tuple
        # with 1 at the beginning (so the first layer outputs size ch)
        if isinstance(ch_div, int):
            ch_div = (1,) + (ch_div,)*(len(spec)-1)
        assert len(ch_div) == len(spec)
        
        self.dec_layers = nn.Sequential()

        w_in = ch_in + dim_cond
        w_out = ch
        self.conv_out = None
        for div,sp in zip(ch_div, spec):
            w_out //= div
            k, s, p = _parse_spec(sp)
            conv = CylinderConvTranspose(w_in, w_out, k=k, stride=s, pad_z=p)
            self.conv_out = conv
            self.dec_layers.append(conv)
            w_in = w_out
            self.dec_layers.append(self.activation)
        
        # 1x1 conv to final output dims
        self.dec_layers.append(CylinderConv(w_in, ch_out, k=(1,1), stride=(1,1), pad_z=False))
        if self.output_activation is not None:
            self.dec_layers.append(self.output_activation)
        
        if self.learn_R:
            self.R_activation = _make_activation(R_activation)
            self.R_layers = nn.Sequential()
            w_in = ch_in+dim_cond
            w_out = 128
            for _ in range(2):
                self.R_layers.append(CylinderConv(w_in, w_out, k=(3,3), stride=(1,1), pad_z=True))
                w_in = w_out
                self.R_layers.append(self.activation)
            self.R_layers.append(nn.AdaptiveAvgPool2d((1,1)))
            self.R_layers.append(FlattenVoxels())
            self.R_layers.append(nn.Linear(w_out, 1))
            self.R_layers.append(self.R_activation)
        
    def forward(self, x, cond=None):
        # x ~ (*, L, H, W)
        # cond ~ (*, 1)
        # assumes cond is already logscaled

        #log_cond = (torch.log1p(cond) - 10.356)/2.8 # (*, 1)
        if cond is not None:
            # broadcast the condition variable across the input image
            xc = cond[:,None,None].expand((-1,-1,x.shape[-2],x.shape[-1])) # (*, L, H, W)
            x = torch.concat([x,xc], axis=-3) # (*, L+1, H, W)

        # all sum to 1
        out = self.dec_layers(x) # (*, R, Z, A)
        
        if self.learn_R:
            R = self.R_layers(x) # (*, 1)
            R = R + self.R_trim

            out = R[...,None,None].to(out.dtype) * out
            # now out will sum to R

            #return out, R
        
        return out.float()

class Discriminator(nn.Module):
    def __init__(self, *,
            ch_in,
            spec,
            ch=16,
            ch_mult=2,
            dim_cond=0,
            activation='swish',
            factor1=1.,
            factor2=3000.,
            factor3=7.,
            ):
        super().__init__()

        if dim_cond not in (0, 1):
            raise NotImplementedError("todo")
        
        self.activation = _make_activation(activation)

        if isinstance(ch_mult, int):
            ch_mult = (1,) + (ch_mult,)*(len(spec)-1)
        assert len(ch_mult) == len(spec)

        w_in = ch_in + dim_cond
        w_out = ch
        self.layers = nn.Sequential()
        for mul, sp in zip(ch_mult, spec):
            w_out *= mul
            k, s, p = _parse_spec(sp)
            self.layers.append(CylinderConv(w_in, w_out, k=k, stride=s, pad_z=p))
            w_in = w_out
            self.layers.append(self.activation)

        # final output layer to reshape to single logit
        self.layers.append(CylinderConv(w_in, 1, k=(1,1), stride=(1,1), pad_z=False))

        self.factor1=torch.tensor(factor1*1.)
        self.factor2=torch.tensor(factor2*1.)
        self.factor3=torch.tensor(factor3*1.)

    def forward(self, x, cond=None):
        x = torch.log(self.factor1+self.factor2*x)/self.factor3

        if cond is not None:
            xc = cond[:,None,None].expand((-1,-1,x.shape[-2],x.shape[-1]))
            x = torch.concat([x,xc], axis=-3)

        out = self.layers(x)
        return out
