import torch
from torch import nn

from ..layers import CylinderConv
from ..layers.misc import LogScale, ZPad
from ..layers.fft import FFTDownsample
from ..util import get_activation_by_name, parse_conv_spec
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, *,
            ch_in, # number of radial calo layers
            ch_out, # channel dimension of the output feature map
            conv_spec, # specifier strings for the layers
            cond_dim = 1, # dimension of the conditional input
            activation='swish', # hidden activation
            output_activation=None, # final output activation
            init_ch=-1,
            log_scale_params=None,
            z_pad=None,
            z_padding_strategy=None,
            ):
        super().__init__()

        if cond_dim not in (0,1):
            raise NotImplementedError("cond_dim>1 is possible but haven't dealt with that case yet.")
        self.register_buffer('cond_dim', torch.tensor(cond_dim))
        
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.z_padding_strategy = z_padding_strategy
        
        activation_class = get_activation_by_name(activation)
        output_activation_class = get_activation_by_name(output_activation)

        if self.z_padding_strategy == 'internal':
            self.z_pad = ZPad(z_pad)
        else:
            self.register_module('z_pad', None)

        if log_scale_params is not None:
            self.log_scale = LogScale(*log_scale_params)
        else:
            self.register_module('log_scale', None)

        w_in = ch_in + cond_dim # input has R + C channels
        w_out = init_ch # base number of output channels
        self.layers = nn.Sequential()
        for spec in conv_spec:
            ltype, *spec = spec.split(':')
            ltype = ltype.strip()
            if ltype == 'fftd':
                args = {}
                args['n_drop'] = int(spec[0])
                if len(spec) > 1:
                    args['fft_dim'] = int(spec[1])
                if len(spec) > 2:
                    phase_dithering = spec[2].strip().lower()
                    assert phase_dithering in ('true', 'false')
                    args['phase_dithering'] = (phase_dithering == 'true')
                self.layers.append(FFTDownsample(**args))
                continue
            if ltype == 'zpad':
                self.layers.append(ZPad((int(spec[0]),int(spec[1]))))
                continue

            assert ltype == 'cconv'

            k, s, p, w_out = parse_conv_spec(':'.join(spec), w_out)

            self.layers.append(CylinderConv(w_in, w_out, k=k, stride=s, pad_z=p))
            w_in = w_out

            self.layers.append(activation_class())

        # final output to reshape to latent dim
        self.layers.append(CylinderConv(w_in, ch_out, k=(1,1), stride=(1,1), pad_z=False))

        if output_activation_class is not None:
            self.layers.append(output_activation_class())
        

    def forward(self, x, cond=None):
        # x (*, R, Z, A)
        # cond (*, 1)
        # assumes x is normalized to e_inc, and cond is logscaled.

        if self.z_pad:
            x = self.z_pad(x)

        if self.log_scale:
            x = self.log_scale(x)

        if self.cond_dim > 0:
            # WARNING!!! We are also broadcasting this into the zpadding layers!
            # broadcast the condition variable across the input image
            xc = cond[:,None,None].expand((-1,-1,x.shape[-2],x.shape[-1])) # (*, 1, Z, A)
            x = torch.concat([x,xc], axis=-3) # (*, R+1, Z, A)

        #out = self.layers(x)
        out = x
        for layer in self.layers:
            out = layer(out)
            #print('out', out.shape)
        return out


