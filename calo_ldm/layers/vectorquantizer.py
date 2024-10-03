import torch
import torch.nn as nn

import numpy as np
import einops as ein

"""
see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
____________________________________________
Discretization bottleneck part of the VQ-VAE.
Inputs:
- n_e : number of embeddings
- e_dim : dimension of embedding
- beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
_____________________________________________
"""

# NB: this is the VectorQuantizer2 module from MishaLaskin/vqvae, we have just named
# it VectorQuantizer here since we don't need both.
class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=True, legacy=True, pixels_dim=3):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        assert pixels_dim in (1,3,4)
        self.pixels_dim = pixels_dim # number of pixel dimensions (1 for flat ds1, 3 for 3d ds2 and ds3)

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            print("INFO: VQ remap enabled!")
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        if sane_index_shape:
            print("INFO: VQ sane shape enabled (default): all the codes in same shape as latent except quantized channel. e.g. ")
        else:
            print("WARN: VQ sane shape disabled!! all the codes in the shape of flattened 1D even batch!")
            raise NotImplementedError # not support any more

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        if self.pixels_dim == 3:
            # reshape z -> (batch, height, width, channel) and flatten
            z = ein.rearrange(z, 'b c h w -> b h w c').contiguous() # channels last!!!!#(N,C,Z,A) -> (N,..,C)
        elif self.pixels_dim == 1:
            z = ein.rearrange(z, 'b c h -> b h c').contiguous() # channels last!!!!# (N,C,L)  -> (N,..,C)
        elif self.pixels_dim == 4: # 3D + channel. NCRZA
            z = ein.rearrange(z, 'b c r z a -> b r z a c').contiguous() # channels last!!!!# (N,C,L)  -> (N,..,C)

        assert z.shape[-1]==self.e_dim
        z_flattened = z.view(-1, self.e_dim) # (N,E)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, ein.rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape) # ()
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.pixels_dim == 3:
            # reshape back to match original input shape
            z_q = ein.rearrange(z_q, 'b h w c -> b c h w').contiguous()
        elif self.pixels_dim == 1:
            z_q = ein.rearrange(z_q, 'b h c -> b c h').contiguous()
        elif self.pixels_dim == 4:
            z_q = ein.rearrange(z_q, 'b r z a c -> b c r z a').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape: # return meaningful codes instead flattened one
            if self.pixels_dim == 3:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[-2], z_q.shape[-1]) # (N, H, W) 
            elif self.pixels_dim == 1:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[-1]) #  (N,H)
            elif self.pixels_dim == 4:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[-3], z_q.shape[-2], z_q.shape[-1]) # (N, R, Z, A) 

        return {
            "quant":z_q,
            "qloss":loss,
            "perplexity":perplexity,
            "min_encodings":min_encodings,
            "min_encoding_indices":min_encoding_indices,
        }

    def get_codebook_entry(self, indices, sane_shape=None): 
        if self.sane_index_shape:
            assert sane_shape is None
            sane_shape = indices.shape # codes already in sane shape
            batch_len = sane_shape[0]
            indices = indices.reshape(batch_len,-1)
            
        if self.remap is not None:
            batch_len = sane_shape[0]
            # indices = indices.reshape(batch_len,-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(batch_len,-1) # flatten again???

        assert indices.dim()==2
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if self.pixels_dim == 3:
            z_q=z_q.reshape(*sane_shape,-1)
            # reshape back to match original input shape
            z_q = ein.rearrange(z_q, 'b h w c -> b c h w').contiguous()
            assert  ( self.sane_index_shape and z_q.shape[-1] == sane_shape[-1] and z_q.shape[-2] == sane_shape[-2] ) \
                or ( (not self.sane_index_shape) and z_q.shape[-1]*z_q.shape[-2] == indices.shape(-1) )
        elif self.pixels_dim == 1:
            z_q = ein.rearrange(z_q, 'b h c -> b c h').contiguous()
            assert (self.sane_index_shape and z_q.shape[-1] == sane_shape[-1] ) \
                or ( (not self.sane_index_shape) and z_q.shape[-1] == indices.shape(-1) )
        elif self.pixels_dim == 4:
            z_q=z_q.reshape(*sane_shape,-1)
            # reshape back to match original input shape
            z_q = ein.rearrange(z_q, 'b r z a c -> b c r z a').contiguous()
            assert  ( self.sane_index_shape and z_q.shape[-1] == sane_shape[-1] and z_q.shape[-2] == sane_shape[-2] and z_q.shape[-3] == sane_shape[-3]) \
                or ( (not self.sane_index_shape) and z_q.shape[-1]*z_q.shape[-2]*z_q.shape[-3] == indices.shape(-1) )

        return z_q
