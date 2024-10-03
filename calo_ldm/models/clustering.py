import torch
from torch import nn
import torch.nn.functional as F

from ..layers import CylinderConv
from ..util import get_activation_by_name, parse_conv_spec

import copy
def _layer_clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class ClusterEncoder(nn.Module):
    def __init__(self,
            in_features,
            out_features,
            conv_spec,
            w_out=3,
            h_out=4,
            dropout=0.1,
            activation='relu',
            ch_init=-1,
            ):
        super().__init__()

        self.out_features = out_features
        
        activation_class = get_activation_by_name(activation)

        self.layers = nn.Sequential()

        f_in = in_features
        f_out = ch_init
        for spec in conv_spec:
            k, s, pad_z, f_out = parse_conv_spec(spec, f_out)

            self.layers.append(CylinderConv(f_in, f_out, k=k, stride=s, pad_z=pad_z))
            f_in = f_out

            if dropout:
                self.layers.append(nn.Dropout2d(dropout))

            self.layers.append(activation_class())

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(f_out*w_out*h_out, out_features))
        
    def forward(self, x, *, output_layers=None):
        x = torch.log1p(x) / 14
        
        if output_layers is None:
            x = self.layers(x)
            return x
        
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in output_layers:
                outputs.append(x)
        return outputs

# FIXME: the Decoder and ClusterNet need to be updated as well,
# but for now we just need the encoder model.

class ClusterDecoder(nn.Module):
    def __init__(self, d_model, N=1, nhead=4):
        super().__init__()
        
        self.d_model = d_model
        self.decoder = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nhead) # , batch_first=True
        self.decoders = _layer_clones(self.decoder, N)
        self.generator = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
        self.embedding = nn.Sequential(
            nn.Linear(4, self.d_model),
            #nn.ReLU(),
        )
    
    def forward(self, tgt, mem):
        x = tgt
        
        #torch.log1p_(x[:,:,3])
        xe = torch.log1p(x[:,:,3:4]) / 14
        #x[:,:,3] /= 14
        
        x = torch.cat([x[:,:,:3], xe], axis=-1)
        
        x = self.embedding(x)
        
        msk = subsequent_mask(x.shape[-2]).to(x.device)#[None]
        #print('msk', msk.shape)
        #msk = torch.tile(msk, (tgt.shape[0], 1, 1))
        #print('msk', msk.shape)
        for layer in self.decoders:
            x = layer(x, mem, tgt_mask=msk)
            
        out = self.generator(x)
        xyz = out[:,:,:3]
        #e = torch.expm1((torch.celu(out[:,:,3:4])+1) * 10)
        e = torch.expm1((torch.sigmoid(out[:,:,3:4])) * 14)
        cat = out[:,:,4:5]
        
        out = torch.cat([xyz, e, cat], axis=-1)
        
        return out

class ClusterNet(nn.Module):
    def __init__(self, d_model, N=4, nhead=4):
        super().__init__()
        
        self.d_model = d_model
        self.encoder = CylinderEncoder(self.d_model)
        self.decoder = ClusterDecoder(self.d_model, N=N, nhead=nhead)
    
    def forward(self, x, tgt):
        mem = self.encoder(x)[:,None]
        return mem
        # out = self.decoder(tgt, mem)
        # return outss
