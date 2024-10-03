#!/usr/bin/env python
from omegaconf import OmegaConf
from calo_ldm.util import instantiate_from_config

import os
import sys
import argparse
import torch
import h5py
import numpy as np
from glob import glob

import time
from torch.distributions.categorical import Categorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import ExpTransform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

LN10=2.30258509299
DIST={
    "1_photon":Categorical(probs=torch.tensor([10]*11 + [5,3,2,1])),
    "1_pion":Categorical(probs=torch.tensor([10]*10 + [9.8,5,3,2,1])),
    "2":TransformedDistribution(Uniform(3*LN10,6*LN10),ExpTransform()),
    "3":TransformedDistribution(Uniform(3*LN10,6*LN10),ExpTransform()), # need 10^...; 
}

PREDEFINED={
    "1_photon":np.repeat(2**np.arange(8,23),[
            10000, 10000, 10000, 10000, 10000, 
            10000, 10000, 10000, 10000,10000, 
            10000,  5000,  3000,  2000,  1000]),
    "1_pion":np.repeat(2**np.arange(8,23),[
            10000, 10000, 10000, 10000, 10000, 
            10000, 10000, 10000, 10000, 10000,  
            9800,  5000,  3000,  2000,  1000]),
}

for v in PREDEFINED.values():
    np.random.shuffle(v)

def sample_cond(model,particle,batch_size):
    batch={}
    if particle in ["2","3"]:
        batch["E_inc"]=DIST[particle].sample((batch_size,)).unsqueeze(-1).to(model.device)
    else:
        batch["E_inc_binned"]=DIST[particle].sample((batch_size,)).unsqueeze(-1).to(model.device)
        batch["E_inc"]=2**(8+batch["E_inc_binned"])

    batch = model.vq_model.preprocess_cond(batch)
    batch = model.preprocess_cond(batch)
    return batch

def sample_cond_direct(model,particle,conds):
    batch={}
    if particle in ["2","3"]:
        batch["E_inc"]=conds.squeeze().unsqueeze(-1).to(model.device)
    else:
        batch["E_inc"]=conds.squeeze().unsqueeze(-1).to(model.device)
        batch["E_inc_binned"]=(torch.log2(batch["E_inc"])-8).long()

    batch = model.vq_model.preprocess_cond(batch)
    batch = model.preprocess_cond(batch)
    return batch

def submission_format(particle,showers,incident_energies):
    if particle in ["2","3"]:
        dim=45*16*9 if particle=="2" else 45*50*18
        return {
                'showers': showers.permute((0,2,3,1)).reshape(-1,dim),
                'incident_energies': incident_energies,
                }
    else:
        return {
                'showers': showers,
                'incident_energies': incident_energies,
                }

def sample_cond_from_dist(model, particle, N, batch_size, **kws):
    n_gen = 0
    incident_energies = []
    showers = []
    while n_gen < N:
        n_gen += batch_size
        batch = sample_cond(model,particle,batch_size)
        gen_post = model.sample_fullchain(batch)
        incident_energies.append(batch['E_inc'].cpu())
        showers.append(gen_post['pixels_E_pred'].cpu())
        
    incident_energies = torch.concat(incident_energies, axis=0)[:N]
    showers = torch.concat(showers, axis=0)[:N]
    return submission_format(particle,showers,incident_energies)

def sample_cond_from_file(model, particle, batch_size, **kws):
    showers = []
    incident_energies = []
    C=PREDEFINED[particle]
    nbatches=int(C.shape[-1] / batch_size)
    if C.shape[-1] % batch_size !=0:
        nbatches+=1
    n_gen=0
    print("Generating based on cond of inputs file...")
    for i in range(nbatches):
        _c= C[batch_size*i:batch_size*(i+1)] if batch_size*(i+1)<=C.shape[-1] else C[batch_size*i:None]
        c=torch.from_numpy(_c)
        conds = sample_cond_direct(model,particle,c) # convert to proper format
        # gen
        gen_post = model.sample_fullchain(conds)
        n_gen+=conds['E_inc'].shape[-1]
        #
        incident_energies.append(conds['E_inc'].cpu())
        showers.append(gen_post['pixels_E_pred'].cpu())

    print(f"Gen:{n_gen}")
    incident_energies = torch.concat(incident_energies, axis=0)
    showers = torch.concat(showers, axis=0)
    return submission_format(particle,showers,incident_energies)

def sanity_check(data):
    incident_energies = data['incident_energies']
    showers = data['showers']
    print(f"--> cond shape {incident_energies.shape}, E shape {showers.shape}")
    print(f"--> Emin {torch.min(showers)}, Emax {torch.max(showers)}, C min {torch.min(incident_energies)}, Cmax {torch.max(incident_energies)}")
    print(f"--> Etot min {torch.min(showers.sum(axis=(-1)))}, Etot max {torch.max(showers.sum(axis=(-1)))}, Etot/C min {torch.min(showers.sum(axis=(-1))[:,None] / incident_energies)}, Etot/C max {torch.max(showers.sum(axis=(-1))[:,None] / incident_energies)}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='Output file (default latent.h5 for encode, reco.h5 for decode)')
    parser.add_argument('--model', required=True, help='Path the the model log directory')
    parser.add_argument('--type', required=True, choices=['1_photon', '1_pion', '2', '3'], help='Particle type')
    parser.add_argument('--checkpoint', default='auto', help='Checkpoint file to run on, relative to the model checkpoints/ dir. Default: last.ckpt')
    parser.add_argument('--batch-size', default=512, type=int, help='batch size')
    parser.add_argument('--nevts', default=100000, type=int, help="N generated. Should not be used when condition file is given")
    parser.add_argument('--device', default=None, help='torch device to run on')
    parser.add_argument('--debug', action="store_true", help="Debug mode")
    parser.add_argument('--timing', action="store_true", help="timing mode, no save data and run multiple times")
    args = parser.parse_args()

    if args.device:
        device = args.device
    print("Will run on device:", device)

    if not args.debug and os.path.exists(args.out):
        print(f"Output file {args.out} already exists! Abort.")
        sys.exit(1)

    config_files = list(sorted(glob(os.path.join(args.model, 'configs', '*.yaml'))))
    print("Loading from config files:", config_files)
    config = OmegaConf.merge(*[OmegaConf.load(c) for c in config_files])
    model = instantiate_from_config(config['model'])

    if args.checkpoint=="auto":
        ckpts=list(glob(os.path.join(args.model, 'checkpoints',"e*.ckpt")))
        ckpt_path = sorted(ckpts)[-1]
    else:
        ckpt_path = os.path.join(args.model, 'checkpoints', args.checkpoint)
    print("Loading checkpoint from", ckpt_path)
    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt['state_dict'],strict=False)
    for p in model.parameters():
        p.requires_grad = False
    print(model)
    model.to(device)
    model.eval()

    if args.timing:
        for i in range(3):
            start = time.time()
            if args.type in ["2","3"]:
                ret = sample_cond_from_dist(model, 
                                particle=args.type, 
                                batch_size=args.batch_size, 
                                N=args.nevts, 
                                debug=args.debug)
            else:
                ret = sample_cond_from_file(model, 
                                particle=args.type, 
                                batch_size=args.batch_size, 
                                debug=args.debug)
            end = time.time()
            nevts=ret['incident_energies'].shape[0]
            print(f"Run {i}: Generation time (no DL) total {end - start:.3}s, {nevts/1000:.0}k, bs={args.batch_size}, Per shower {(end - start)/nevts*1000:.5}ms")
        exit()

    start = time.time()
    if args.type in ["2","3"]:
        result = sample_cond_from_dist(model, 
                        particle=args.type, 
                        batch_size=args.batch_size, 
                        N=args.nevts, 
                        debug=args.debug)
    else:
        result = sample_cond_from_file(model, 
                            particle=args.type, 
                            batch_size=args.batch_size, 
                            debug=args.debug)
    end = time.time()

    print(f"Generation time (no DL) total {end - start:.3}s, Per shower {(end - start)/len(result['incident_energies'])*1000:.5}ms")
    
    assert sanity_check(result)

    print("Saving result to", args.out)
    with h5py.File(args.out, 'w') as fout:
        for k, v in result.items():
            print('\t'+k, v.shape)
            fout[k] = v
    print("Done.")
    sys.exit()

    