import torch
from torch import nn
import pytorch_lightning as pl

from ..layers import CylinderConv, CylinderConvTranspose
from ..layers.misc import ZUnPad, FlattenVoxels
from ..layers.fft import FFTInterpolate
from ..util import get_activation_by_name, parse_conv_spec
import torch.nn.functional as F

class Decoder(pl.LightningModule):
    def __init__(self, *,
            ch_in, # channel dimension of the latent space feature map
            ch_out, # number of radial calo layers
            conv_spec,
            cond_dim=1, # dimension of the conditional input. if zero, means don't use condition input
            activation='swish',
            output_activation='voxel_softmax',
            learn_R=True,
            R_activation='softplus',
            R_trim=1e-2, # small number to add to the output of R to prevent 0
            ch_init=-1,
            z_pad=None,
            z_padding_strategy=None,
            ):

        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out

        assert z_padding_strategy in (None,'none','internal','preprocess')
        if z_padding_strategy == 'internal':
            self.z_unpad = ZUnPad(z_pad)
        else:
            self.register_module('z_unpad', None)

        self.register_buffer('R_trim', torch.tensor(R_trim).float())
        self.learn_R = learn_R

        if not cond_dim in (0,1):
            raise NotImplementedError("cond_dim>1 is possible but haven't dealt with that case yet.")
        self.register_buffer('cond_dim', torch.tensor(cond_dim))
        
        activation_class = get_activation_by_name(activation)
        output_activation_class = get_activation_by_name(output_activation)

        self.dec_layers = nn.Sequential()
        self.output_activation_class=output_activation_class() if output_activation is not None else None

        w_in = ch_in + cond_dim
        w_out = ch_init
        self._adaptive_layer = None
        for spec in conv_spec:
            ltype, *spec = spec.split(':')
            ltype = ltype.strip()

            if ltype == 'ffti':
                self.dec_layers.append(FFTInterpolate(int(spec[0])))
                continue
            if ltype == 'zcrop':
                self.dec_layers.append(ZUnPad((int(spec[0]), int(spec[1]))))
                continue
            
            assert ltype == 'cconvT'
            k, s, p, w_out = parse_conv_spec(':'.join(spec), w_out)
            conv = CylinderConvTranspose(w_in, w_out, k=k, stride=s, pad_z=p)
            w_in = w_out

            self.dec_layers.append(conv)
            self.dec_layers.append(activation_class())

            self._adaptive_layer = conv
        
        # 1x1 conv to final output dims
        self.dec_layers.append(CylinderConv(w_in, ch_out, k=(1,1), stride=(1,1), pad_z=False))

        # if output_activation_class is not None:
        #     self.dec_layers.append(output_activation_class())
        
        print("Use decoder to learn R: ",self.learn_R)
        if self.learn_R:
            R_activation_class = get_activation_by_name(R_activation)
            self.R_layers = nn.Sequential()
            w_in = ch_in+cond_dim
            w_out = 128
            # test new R layer
            # self.R_layers.append(FlattenVoxels())
            # self.R_layers.append(nn.Linear(4420, w_out))
            # for _ in range(2):
            #     self.R_layers.append(nn.Linear(w_out, w_out))
            #     self.R_layers.append(activation_class())
            # self.R_layers.append(nn.Linear(w_out, 1))
            # self.R_layers.append(R_activation_class())
            # original one
            for _ in range(2):
                self.R_layers.append(CylinderConv(w_in, w_out, k=(3,3), stride=(1,1), pad_z=True))
                w_in = w_out
                self.R_layers.append(activation_class())
            self.R_layers.append(nn.AdaptiveAvgPool2d((1,1)))
            self.R_layers.append(FlattenVoxels())
            self.R_layers.append(nn.Linear(w_out, 1))
            self.R_layers.append(R_activation_class())

    def get_adaptive_layer_weights(self):
        return self._adaptive_layer.weights
        
    def forward(self, x, cond=None):
        # x ~ (*, L, H, W)
        # cond ~ (*, 1)
        # assumes cond is already logscaled

        #log_cond = (torch.log1p(cond) - 10.356)/2.8 # (*, 1)
        if self.cond_dim > 0:
            # broadcast the condition variable across the input image
            xc = cond[:,None,None].expand((-1,-1,x.shape[-2],x.shape[-1])) # (*, L, H, W)
            x = torch.concat([x,xc], axis=-3) # (*, L+1, H, W) # another Z ??

        # all sum to 1
        #pix = self.dec_layers(x) # (*, R, Z, A)
        pix = x
        for layer in self.dec_layers:
            pix = layer(pix)
            #print('pix', pix.shape)

        if self.z_unpad is not None:
            pix = self.z_unpad(pix)

        if self.output_activation_class:
            pix=self.output_activation_class(pix).float()
        
        out = {'pixels_U_pred': pix}
        if self.learn_R:
            R = self.R_layers(x) # (*, 1)
            R = R + self.R_trim
            out['R_pred'] = R

        return out
