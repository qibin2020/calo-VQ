import torch
from torch import nn
import pytorch_lightning as pl

from ..layers import CylinderConv
from ..layers.misc import LogScale
from ..layers.fft import FFTDownsample,FFTDownsampleV2
from ..util import get_activation_by_name, parse_conv_spec

class Discriminator(pl.LightningModule):
    def __init__(self, *,
            ch_in,
            conv_spec,
            cond_dim=0,
            activation='swish',
            ch_init=-1,
            log_scale_params=None, 
            pooling=None,
            ):
        super().__init__()

        if cond_dim not in (0, 1):
            raise NotImplementedError("todo")

        self.register_buffer('cond_dim', torch.tensor(cond_dim))
        
        activation_class = get_activation_by_name(activation)

        w_in = ch_in + cond_dim
        w_out = ch_init
        self.layers = nn.Sequential()
        for spec in conv_spec:
            ltype, *spec = spec.split(':')
            ltype = ltype.strip()
            if ltype == 'fftd':
                args = {}
                args['n_drop'] = int(spec[0])
                self.layers.append(FFTDownsample(**args))
                continue
            elif ltype == 'fftd1':
                args = {"fft_dim":1}
                args['n_drop'] = int(spec[0])
                self.layers.append(FFTDownsampleV2(**args))
                continue
            elif ltype == 'fftd2':
                args = {"fft_dim":2}
                args['n_drop'] = int(spec[0])
                self.layers.append(FFTDownsampleV2(**args))
                continue
            elif ltype == 'fftd3':
                args = {"fft_dim":3}
                args['n_drop'] = int(spec[0])
                self.layers.append(FFTDownsampleV2(**args))
                continue

            assert ltype == 'cconv'

            k, s, p, w_out = parse_conv_spec(':'.join(spec), w_out)
            self.layers.append(CylinderConv(w_in, w_out, k=k, stride=s, pad_z=p))
            w_in = w_out

            self.layers.append(activation_class())

        if pooling is None:
            # final output layer to reshape to single logit
            self.layers.append(CylinderConv(w_in, 1, k=(1,1), stride=(1,1), pad_z=False))
        else:
            if pooling == 'max':
                self.layers.append(nn.AdaptiveMaxPool2d((1,1)))
            elif pooling == 'avg':
                self.layers.append(nn.AdaptiveAvgPool2d((1,1)))
                self.layers.append(nn.Conv2d(w_in, w_in, kernel_size=(1,1), stride=1))
                self.layers.append(activation_class())
                self.layers.append(nn.Conv2d(w_in, 1, kernel_size=(1,1), stride=1))


        if log_scale_params:
            self.log_scale = LogScale(*log_scale_params)
        else:
            self.register_module('log_scale', None)

    def forward(self, x, cond=None):
        x = self.log_scale(x)

        if self.cond_dim > 0:
            xc = cond[:,None,None].expand((-1,-1,x.shape[-2],x.shape[-1]))
            x = torch.concat([x,xc], axis=-3)

        out = self.layers(x)
        return out
