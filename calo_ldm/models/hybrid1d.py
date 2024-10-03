import torch
from torch import nn

from ..util import get_activation_by_name
from ..layers.misc import LogScale

class Encoder(nn.Module):
    def __init__(self, *,
            pix_in,
            seq_out,
            ch_hidden,
            ch_out,
            dense_channels,
            conv_channels,
            cond_dim,
            cond_bins=None,
            log_scale_params='inherit',
            z_pad=None,
            z_padding_strategy=None,
            dense_activation='relu',
            conv_activation='relu',
            output_activation='relu',
            ):
        super().__init__()

        assert z_pad is None
        assert z_padding_strategy is None
        assert log_scale_params != 'inherit'

        self.cond_bins=cond_bins
        if cond_bins:
            self.embedding = nn.Embedding(cond_bins, cond_dim)
        else:
            self.register_module('embedding', None)

        if log_scale_params is not None:
            self.log_scale = LogScale(*log_scale_params)
        else:
            self.register_module('log_scale', None)

        self.ch_out = ch_out
        self.ch_hidden = ch_hidden

        self.dense_layers = nn.Sequential()
        f_in = pix_in + cond_dim
        for f_out in dense_channels:
            self.dense_layers.append(nn.Linear(f_in, f_out))
            f_in = f_out
            self.dense_layers.append(get_activation_by_name(dense_activation)())

        self.dense_layers.append(nn.Linear(f_in, seq_out*ch_hidden))
        self.dense_layers.append(get_activation_by_name(dense_activation)())

        self.conv_layers = nn.Sequential()
        f_in = ch_hidden
        for f_out in conv_channels:
            self.conv_layers.append(nn.Conv1d(f_in, f_out, kernel_size=3, padding='same'))
            f_in = f_out
            self.conv_layers.append(get_activation_by_name(conv_activation)())

        self.conv_layers.append(nn.Conv1d(f_in, ch_out, kernel_size=1))
        if output_activation is not None:
            self.conv_layers.append(get_activation_by_name(output_activation)())


    def forward(self, x, cond):
        # x (*, P)
        if x.dim()!=2:
            raise NotImplementedError(f"What?! x.dim()={x.dim()}")
        # cond long(*,E) or float(*,E)
        if cond.dim()!=2:
            raise NotImplementedError(f"What?! cond.dim()={cond.dim()}")

        if self.embedding is not None:
            cond = self.embedding(cond.squeeze()) # (*, E)

        if self.log_scale is not None:
            x = self.log_scale(x) # (*, P)

        x = torch.concat([x, cond], axis=-1) # (*, P+E)

        h = self.dense_layers(x) # (*, H*S)
        h = h.unflatten(-1, (self.ch_hidden, -1)) # (*, H, S)

        out = self.conv_layers(h) # (*, C, S)

        return out

# class Decoder(nn.Module):
#     def __init__(self, *,
#             seq_in,
#             pix_out,
#             ch_in,
#             conv_channels,
#             dense_channels,
#             cond_dim,
#             cond_bins=None,
#             conv_activation='relu',
#             dense_activation='relu',
#             output_activation='flat_voxel_softmax',
#             learn_R=False,
#             z_pad=None,
#             z_padding_strategy=None,
#             ):
#         super().__init__()

#         if learn_R:
#             raise NotImplementedError
        
#         assert z_pad is None
#         assert z_padding_strategy is None

#         if cond_bins:
#             self.embedding = nn.Embedding(cond_bins, cond_dim)
#         else:
#             self.register_module('embedding', None)

#         self.conv_layers = nn.Sequential()
#         f_in = ch_in + cond_dim
#         for f_out in conv_channels:
#             self.conv_layers.append(nn.Conv1d(f_in, f_out, kernel_size=3, padding='same'))
#             f_in = f_out
#             self.conv_layers.append(get_activation_by_name(conv_activation)())

#         self.dense_layers = nn.Sequential()
#         f_in = seq_in*f_out
#         for f_out in dense_channels:
#             self.dense_layers.append(nn.Linear(f_in, f_out))
#             f_in = f_out
#             self.dense_layers.append(get_activation_by_name(dense_activation)())

#         self.dense_layers.append(nn.Linear(f_in, pix_out))
#         self._adaptive_layer = self.dense_layers[-1]

#         if output_activation is not None:
#             self.dense_layers.append(get_activation_by_name(output_activation)())

#     def get_adaptive_layer_weights(self):
#         return self._adaptive_layer.weight

#     def forward(self, x, cond):
#         # x (*, C, S)
#         # cond long(*,) or float(*,E)

#         if self.embedding is not None:
#             cond = self.embedding(cond.squeeze()) # (*, E)

#         cond = cond.unsqueeze(-1).expand( (-1,)*(len(x.shape)-1) + (x.shape[-1],))

#         x = torch.concat([x, cond], axis=-2) # (*, C+E, S)

#         h = self.conv_layers(x) # (*, H, S)

#         h = h.flatten(start_dim=-2)

#         out = self.dense_layers(h) # (*, P)

#         return {'pixels_U_pred': out}

# multi-layer normalization decoder. Apply individual softmax to each layer.
class DecoderMH(nn.Module):
    def __init__(self, *,
            seq_in,
            pix_out,
            ch_in,
            conv_channels,
            dense_channels,
            cond_dim,
            cond_bins=None,
            conv_activation='relu',
            dense_activation='relu',
            output_activation='flat_voxel_softmax',
            learn_R=False,
            z_pad=None,
            z_padding_strategy=None,
            layer_seg=None,
            layer_seg_dim=None,
            ):
        super().__init__()

        if learn_R:
            raise NotImplementedError

        assert output_activation
        
        assert z_pad is None
        assert z_padding_strategy is None

        assert layer_seg

        if cond_bins:
            self.embedding = nn.Embedding(cond_bins, cond_dim)
        else:
            self.register_module('embedding', None)

        self.conv_layers = nn.Sequential()
        f_in = ch_in + cond_dim
        for f_out in conv_channels:
            self.conv_layers.append(nn.Conv1d(f_in, f_out, kernel_size=3, padding='same'))
            f_in = f_out
            self.conv_layers.append(get_activation_by_name(conv_activation)())

        self.dense_layers = nn.Sequential()
        f_in = seq_in*f_out
        for f_out in dense_channels:
            self.dense_layers.append(nn.Linear(f_in, f_out))
            f_in = f_out
            self.dense_layers.append(get_activation_by_name(dense_activation)())

        self.dense_layers.append(nn.Linear(f_in, pix_out))
        self._adaptive_layer = self.dense_layers[-1]

        # if output_activation is not None:
        #     self.dense_layers.append(get_activation_by_name(output_activation)())
        
        self.output_layerss=[]
        self.layer_seg=list(layer_seg)
        self.layer_seg_dim=int(layer_seg_dim)
        for s in self.layer_seg:
            self.output_layerss.append(get_activation_by_name(output_activation)())
        
    def get_adaptive_layer_weights(self):
        return self._adaptive_layer.weight

    def forward(self, x, cond):
        # x (*, C, S)
        if x.dim()!=3:
            raise NotImplementedError(f"What?! x.dim()={x.dim()}")
        # cond long(*,E) or float(*,E)
        if cond.dim()!=2:
            raise NotImplementedError(f"What?! cond.dim()={cond.dim()}")

        if self.embedding is not None:
            cond = self.embedding(cond.squeeze()) # (*, E)

        cond = cond.unsqueeze(-1).expand( (-1,)*(x.dim()-1) + (x.shape[-1],))

        # print("[red]x[/red]",x.shape)
        x = torch.concat([x, cond], axis=-2) # (*, C+E, S)
        # print("[red]x'[/red]",x.shape)

        h = self.conv_layers(x) # (*, H, S)
        # print("[red]h[/red]",h.shape)

        h = h.flatten(start_dim=-2) # ??
        # print("[red]h'[/red]",h.shape)

        out = self.dense_layers(h) # (*, P)

        split_layers = torch.split(out, self.layer_seg, dim=self.layer_seg_dim)

        activated_layers = [l(i) for l,i in zip(self.output_layerss,split_layers)]

        output = torch.cat(activated_layers,self.layer_seg_dim) 

        return {'pixels_U_pred': output}
