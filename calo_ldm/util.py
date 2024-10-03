import torch
from torch import nn
import torch.nn.functional as F
from math import floor, ceil

from .layers.misc import (
        VoxelSoftmax, FlatVoxelSoftmax,
        VoxelReluExpm1Max, FlatVoxelReluExpm1Max,
        FlatVoxelCeluExpm1Max,
        VoxelReluExpm1MaxD6,
        )

import importlib
import sys
import re

def recursive_to(obj, device):
    if isinstance(obj, dict):
        new = {k: recursive_to(v, device) for k,v in obj.items()}
        return new
    elif isinstance(obj, (list, tuple)):
        new = [recursive_to(o, device) for o in obj]
        return new
    else:
        return obj.to(device)

_activation_aliases = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'swish': nn.SiLU,
    'silu': nn.SiLU,
    'softplus': nn.Softplus,
    'voxel_softmax': VoxelSoftmax,
    'flat_voxel_softmax': FlatVoxelSoftmax,
    'voxel_relu_expm1_max': VoxelReluExpm1Max,
    'voxel_relu_expm1_max_dyn6': VoxelReluExpm1MaxD6,
    'flat_voxel_relu_expm1_max': FlatVoxelReluExpm1Max,
    'flat_voxel_celu_expm1_max': FlatVoxelCeluExpm1Max,
}
def get_activation_by_name(name):
    if name is None:
        return None
    return _activation_aliases[name.lower()]


def instantiate_from_config(config, passthru={}, overwrite=False):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", {})
    for k,v in passthru.items():
        if not k in params:
            print(f">> Using passthru value for {k} in {config['target']}")
            params[k] = v
        else:
            if overwrite:
                params[k] = v
                print(f"Warning inherited option would overwrite the config!! Now {k}={params[k]}",file=sys. stderr)
            else:
                print(f"Warning inherited option would not overwrite the config!! Still use {k}={params[k]}",file=sys. stderr)
    return get_obj_from_str(config["target"])(**params)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def parse_conv_spec(spec, w_out_prev=None):
    # spec format:
    #   :kAB, :kA -> k=(A,B), k=(A,A)
    #   :sAB, :sA -> stride=(A,B), stride=(A,A)
    #   :p        -> pad_z=True
    #   :*N       -> w_out -> w_out*N
    #   :/N       -> w_out -> w_out//N
    #   :+N       -> w_out -> w_out+N
    #   :-N       -> w_out -> w_out-N
    #   :cN       -> w_out -> N
    # order and case insensitive. you can add extra colons and/or
    # spaces e.g. for readability, they will be ignored.
    # defaults: k=(3,3), stride=(1,1), pad_z=False, w_out=w_out
    # examples:
    #   s12:k53:p -> stride(z=2,a=1), k(z=5,a=3), pad_z=True
    #   k3:s13   -> stride(1,3), k(3,3), pad_z=False
    #   p:k35    -> stride(1,1) k(3,5), pad_z=True
    if w_out_prev is not None:
        w_out = w_out_prev

    s = (1,1)
    k = (3,3)
    pad_z = False
    for x in spec.lower().split(':'):
        x = x.strip()
        if x == '': continue
        elif x.startswith('s'): s=tuple(map(int, x[1:]))
        elif x.startswith('k'): k=tuple(map(int, x[1:]))
        elif x.startswith('*'): w_out *= int(x[1:])
        elif x.startswith('/'): w_out //= int(x[1:])
        elif x.startswith('+'): w_out += int(x[1:])
        elif x.startswith('-'): w_out -= int(x[1:])
        elif x.startswith('c'): w_out = int(x[1:])
        elif x == 'p': pad_z = True
        else:
            raise ValueError(f"Unkown specification key: {x}")

    if len(s) == 1: s = 2*s
    if len(k) == 1: k = 2*k

    if w_out_prev is None:
        return k, s, pad_z
    return k, s, pad_z, w_out


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    elif global_step==threshold:
        print("DISC enabled!")
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()  # consider probs
    cluster_use = torch.sum(avg_probs > 0) / n_embed # count all non zero embed
    return perplexity, cluster_use

def _cal_RZA(RZA,hR,hZ,hA):
    R,Z,A=RZA
    if hR:
        R=floor(R/2)
    if hZ:
        Z=floor(Z/2)
    if hA:
        A=ceil(A/2)
    return [R,Z,A]

def cal_RZA(RZA,p):
    r=parse_spec_str(p)
    return _cal_RZA(RZA,r["hR"],r["hZ"],r["hA"])

def parse_spec_str(p):
    mat = re.match("[RAZO]+(\d+)[FX]*",p)
    if not mat:
        print("Not implemented spec!",p)
        assert False 
    dim=int(mat.group(1))
    hR="R" in p
    hZ="Z" in p
    hA="A" in p
    X="X" in p
    FFT="F" in p
    O="O" in p
    return {
        "hR":hR,
        "hZ":hZ,
        "hA":hA,
        "XFORMER":X,
        "FFT":FFT,
        "O":O,
        "dim":dim,
    }

# conv2s version #############################################
def _cal_ZA(ZA,hZ,hA):
    Z,A=ZA
    if hZ:
        Z=floor(Z/2)
    if hA:
        A=ceil(A/2)
    return [Z,A]

def cal_ZA(ZA,p):
    r=parse_spec_str_conv2s(p)
    return _cal_ZA(ZA,r["hZ"],r["hA"])

def parse_spec_str_conv2s(p):
    mat = re.match("[AZO]+(\d+)[FX]*",p)
    if not mat:
        print("Not implemented spec!",p)
        assert False 
    dim=int(mat.group(1))
    hZ="Z" in p
    hA="A" in p
    X="X" in p
    FFT="F" in p
    O="O" in p
    return {
        "hZ":hZ,
        "hA":hA,
        "XFORMER":X,
        "FFT":FFT,
        "O":O,
        "dim":dim,
    }
    