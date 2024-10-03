import torch
from torch import nn

from calo_ldm.util import instantiate_from_config

class ClusterPerceptualLoss(nn.Module):
    # Learned perceptual metric
    def __init__(self,
            encoder_config,
            checkpoint_file,
            layer_indices=[0,3,6,9,12,15],
            use_dropout=True):
        super().__init__()
        print(">>>Init ClusterPerceptual model.....")

        self.layer_indices = layer_indices
        self.encoder = instantiate_from_config(encoder_config)

        layer_channels = [self.encoder.layers[i].out_features for i in self.layer_indices]

        print(f"Loading ClusterEncoder from checkpoint file {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file,map_location="cpu")
        if checkpoint['repr'].startswith('ClusterNet'):
            # this is a full ClusterNet checkpoint, we need to extract
            # just the encoder part of it
            print(f"NOTE: we were given a ClusterNet checkpoint, will try to extract the encoder weights.")
            state_dict = {k[8:]:v for k,v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        else:
            # otherwise assume we've been given just the encoder params
            state_dict = checkpoint['model_state_dict']
        self.encoder.load_state_dict(state_dict, strict=False)

        self.scaling_layer = ScalingLayer()
        # compress channel
        self.lins=[NetLinLayer(ch, use_dropout=use_dropout) for ch in layer_channels] # never .to() since it don't know current ddp GPU when init
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.encoder(in0_input, output_layers=self.layer_indices), self.encoder(in1_input, output_layers=self.layer_indices)
        feats0, feats1, diffs = {}, {}, {}
        lins = [lin.to(in0_input) for lin in self.lins]
        # lins = self.lins
        for kk in range(len(self.layer_indices)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.layer_indices))]
        val = res[0]
        for l in range(1, len(self.layer_indices)):
            val += res[l]
            
        # outs0_E, outs1_E = self.net(in0_input), self.net(in1_input) # energy regression
        # val += (outs0_E-outs1_E)**2
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # self.register_buffer('shift', torch.Tensor([0, 0, 0])[None, :, None, None]) # to match the original pretrained model feature??!
        # self.register_buffer('scale', torch.Tensor([1, 1, 1])[None, :, None, None])

    def forward(self, inp):
        # return (inp - self.shift) / self.scale
        return inp


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

