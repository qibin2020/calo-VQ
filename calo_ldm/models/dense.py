import torch
from torch import nn

from ..util import get_activation_by_name
from ..layers.misc import LogScale

class Dense(nn.Module):
    def __init__(self, *,
            ch_in,
            ch_out,
            channels,
            cond_dim,
            cond_bins=None,
            activation='relu',
            output_activation=None,
            log_scale_params=None,
            z_pad=None,
            z_padding_strategy=None,
            learn_R=False,
            ):
        super().__init__()

        assert z_pad is None
        assert z_padding_strategy is None

        self.ch_in = ch_in
        self.ch_out = ch_out

        if learn_R:
            raise NotImplementedError

        if activation is not None:
            activation = get_activation_by_name(activation)

        if log_scale_params is not None:
            self.log_scale = LogScale(*log_scale_params)
        else:
            self.register_module('log_scale', None)

        if cond_bins:
            self.embedding = nn.Embedding(cond_bins, cond_dim)
        else:
            self.register_module('embedding', None)

        self.layers = nn.Sequential()
        f_in = ch_in + cond_dim
        for f_out in channels:
            self.layers.append(nn.Linear(f_in, f_out))
            f_in = f_out

            self._adaptive_layer = self.layers[-1] # for compatibility w/ adaptive discriminator

            if activation is not None:
                self.layers.append(activation())

        self.layers.append(nn.Linear(f_in, ch_out))
        if output_activation is not None:
            self.layers.append(get_activation_by_name(output_activation)())

    def get_adaptive_layer_weights(self):
        return self._adaptive_layer.weight

    def forward(self, x, cond):
        # x (*, P)
        # cond long(*,C) or float(*,E)
        assert cond.dim()==2
        if self.embedding is not None:
            cond = self.embedding(cond.squeeze()) # (*, E)

        if self.log_scale is not None:
            x = self.log_scale(x)

        x = torch.concat([x, cond], axis=-1) # (*, P+E)

        return self.layers(x)

class Encoder(Dense):
    def __init__(self, *, pix_out, **kwargs):
        pix_ch_out = kwargs['ch_out']
        kwargs['ch_out'] = kwargs['ch_out']*pix_out
        super().__init__(**kwargs)
        self.pix_out = pix_out
        self.pix_ch_out = pix_ch_out

    def forward(self, x, cond):
        x = super().forward(x, cond)
        x = x.unflatten(-1, (-1,self.pix_out))
        return x

# simple wrapper class to get the right return type for the decoder network specifically
class Decoder(Dense):
    def __init__(self, *, pix_in, **kwargs):
        pix_ch_in = kwargs['ch_in']
        kwargs['ch_in'] = kwargs['ch_in']*pix_in
        super().__init__(**kwargs)
        self.pix_in = pix_in
        self.pix_ch_in = pix_ch_in

    def forward(self, x, cond):
        x = x.flatten(start_dim=-2)
        return {'pixels_U_pred': super().forward(x, cond)}
