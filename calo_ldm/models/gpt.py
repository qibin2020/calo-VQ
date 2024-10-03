"""
NOTE: This is adapted from Andrej Karpathy's minGPT, via Xiulong's mod.

GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

import calo_ldm.layers.transformer as xfmr
from calo_ldm.util import instantiate_from_config, recursive_to

from omegaconf import OmegaConf

from glob import glob
import os.path
import math

from functools import reduce

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, codebook_size, sequence_len, **kwargs):
        self.codebook_size = codebook_size
        self.sequence_len = sequence_len
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    """  the full GPT language model, with a context size of sequence_len """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.codebook_size, config.n_embd - config.cond_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.sequence_len, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # linear layer to transform skel latent size to match embedding size
        # self.linear = nn.Linear(512,)
        # transformer
        self.blocks = nn.Sequential(*[xfmr.Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.codebook_size, bias=False)

        self.sequence_len = config.sequence_len
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, cond, targets=None):
        # print("idx",idx.shape)
        # print("cond",cond.shape)
        b, t = idx.size()
        t += 1
        assert t <= self.sequence_len, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        # print("token_embeddings",token_embeddings.shape)
        # print("test",torch.cat([torch.zeros(b,1,token_embeddings.size(-1), device=idx.device),
        #                             token_embeddings], dim=1).shape)
        token_embedding_cat = torch.cat(
                                [torch.cat([torch.zeros(b,1,token_embeddings.size(-1), device=idx.device),
                                    token_embeddings], dim=1),
                                torch.cat([cond]*t, dim=1)],
                                dim=-1)

        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embedding_cat + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def sample(self, inp):
        b, t, _ = inp.size()
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        # print(position_embeddings.shape, inp.shape)
        x = self.drop(inp + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class CondGPT(pl.LightningModule):
    def __init__(self, *,
            codebook_size,
            sequence_shape, # now shape(as list) is used instead of len, e.g. [32], 
            # sequence_len, # we use shaped codes for later usage. ds1: h, ds/3: h,w
            n_layer,
            n_head,
            vq_config,
            n_embd=512,
            #hidden_size=64,
            cond_bins=0, # number of bins for condition variable, if zero, don't bin.
            cond_dim=1, # guessing at what this means?
            cond_proj=False, # use Linear projection after embed for cond
            predict_R=False,
            R_seq_len=None, # How many bits or *codes to use for R if we are going to predict it
            R_bits=None, # How many *bits or codes to use for R if we are going to predict it
            metric_R=True, # do R metrics
            R_renorm=True,  #renorm R after unpadding
            R_max=1.3,
            record_freq=1,
            pure_mode=False,
            monitor=None,
            debug_mode=0, # which to plot for the metric
            # 0: sample v.s. truth: eval performance eval (default)
            # (1: sample v.s. recon(forward): debug GPT)
            # 2: sample v.s. reco(forward+codes lookup): debug GPT (recomm.)
            # (-1: recon(forward) v.s. truth: debug VAE)
            # -2: reco(forward+codes lookup) v.s. truth: debug VQVAE (recomm.)
            # (-3: reco(forward+codes lookup) v.s. recon(forward): debug codes lookup)
            use_vq_cond=True,
            ):
        super().__init__()
        # self.nonseg_dim=nonseg_dim
        self.codebook_size = codebook_size
        self.cond_bins = cond_bins
        self.predict_R = predict_R
        assert R_bits!=None or R_seq_len!=None
        if self.cond_bins == 0:
            cond_dim = 1
        self.use_vq_cond=use_vq_cond

        if monitor is not None:
            self.monitor = monitor

        if vq_config.get('logdir', None) is not None:
            vq_config['model_config'] = glob(os.path.join(vq_config['logdir'],'configs','*-project.yaml'))[-1]
            vq_config['checkpoint'] = glob(os.path.join(vq_config['logdir'],'checkpoints','epoch=*.ckpt'))[-1]
        self.vq_model = instantiate_from_config(OmegaConf.load(vq_config['model_config'])['model'],
                passthru={'ckpt_path': vq_config.get('checkpoint', None)}, overwrite=True)
        # freeze the VQ model
        for p in self.vq_model.parameters():
            p.requires_grad = False
        self.vq_model.eval()
        assert self.codebook_size == self.vq_model.n_embed
        # assert sequence_len == self.vq_model.encoder.seq_out # better to check this...
        # assert sequence_len == self.vq_model.decoder.seq_in
        self.sequence_shape=sequence_shape
        assert len(sequence_shape)+1 in [2,3,4] # shape of codes with batch dimension ; ok now we support ds3
        sequence_len = reduce(lambda x, y: x*y, sequence_shape)
        self.sequence_len=sequence_len
        if self.vq_model.is_ds23:
            if self.vq_model.dataset_name=="2": # R Z A
                self.input_dim=(9, 45, 16)
            else:
                self.input_dim=(18, 45, 50)

        if self.predict_R:
            self.bits_per_code = int(math.log2(self.codebook_size))
            assert 2**self.bits_per_code == self.codebook_size 
            # each code should embed one complete binary number
            if R_seq_len:
                self.R_seq_len=R_seq_len
                self.R_bits=self.R_seq_len*self.bits_per_code
            else:
                self.R_bits=R_bits
                self.R_seq_len = math.ceil(self.R_bits/self.bits_per_code)
            # 
            print(f"Using {self.R_seq_len} initial sequence elements to predict R of {self.R_bits} bits")
            if self.vq_model.layer_seg:
                print(f"Layer-norm R enabled, in total ({len(self.vq_model.layer_seg)}+1) * {self.R_seq_len} sequence elements")
                sequence_len = sequence_len + self.R_seq_len * (len(self.vq_model.layer_seg)+1)
            else:
                sequence_len = sequence_len + self.R_seq_len

        self.config = GPTConfig(
                codebook_size=codebook_size, sequence_len=sequence_len,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                cond_dim=cond_dim,
                )
        self.gpt = GPT(self.config)
        self.cond_proj=cond_proj
        if self.cond_bins > 0:
            self.lab_emb = nn.Embedding(self.cond_bins, cond_dim) # cond_bins = 100, hid = 64
            self.proj = nn.Linear(cond_dim, cond_dim, bias=False) # hidden_size = cond_dim = 16
        else:
            self.register_module('lab_emb', None)
            self.register_module('proj', None)

        self.n_embd=n_embd
        self.R_renorm=R_renorm if self.vq_model.is_ds23 else False
        self.R_max=R_max
        self.record_freq=record_freq
        self.on_record=False
        self.pure_mode=pure_mode
        self.debug_mode=debug_mode
        assert debug_mode in [0,1,2,-1,-2,-3]

        # loopback test of R coding
        s=torch.rand([1000,1000])
        s_codes=self.convertR(s)
        s_rep=self.decodeR(s_codes)
        print("loopback test R: error sum",(s_rep-s).sum()/s.sum())
        if(abs((s_rep-s).sum()/s.sum())>0.01):
            print("? do you have a good R enbemding function?")
            assert False

    def forward(self, cond, idx, targets): #skel_input
        if self.cond_bins:
            cond = self.lab_emb(cond)
            if self.cond_proj:
                cond = self.proj(cond)
            logits, loss = self.gpt(idx, cond, targets)
        else:
            cond = cond.unsqueeze(-2) # --> (N,dumm,1)
            logits, loss = self.gpt(idx, cond, targets)
        return logits, loss
    
    def convertR(self,R_in): # convert R(float) to N-base number, N=n_embed
        if R_in.dim()!=2:
            raise NotImplementedError(f"Wrong R dim {R_in.shape}")
        # encode R as a fixed-point representation in the first log2(b) steps
        # of the latent sequence. the inverse of this process is in postprocess_codes
        R = ((2**self.R_bits)*R_in/self.R_max).long()
        R = torch.clip(R, 0, 2**self.R_bits-1)
        mask = 2**self.bits_per_code - 1
        rcodes = []
        R = R.unsqueeze(-1)
        for _ in range(self.R_seq_len):
            rcodes.append(R & mask)
            R = R >> self.bits_per_code
        return torch.concat(rcodes[::-1], axis=-1).reshape(R_in.shape[0],-1) # reorder as msb to lsb
    
    def decodeR(self,_codes):
        if _codes.dim()!=2:
            raise NotImplementedError(f"Wrong codes dim {_codes.shape}")
        codes = _codes.reshape(_codes.shape[0],-1,self.R_seq_len)
        R_pred = codes[:,:,0]
        for i in range(self.R_seq_len-1):
            R_pred = R_pred << self.bits_per_code
            R_pred = R_pred | codes[:,:,i+1]
        return (R_pred.double() / 2**self.R_bits * self.R_max).float().reshape(_codes.shape[0],-1) # GPT_R mustbe 2D

    def preprocess_cond(self,batch):
        # deal with cond 
        if self.cond_bins == 0: # unbinned cond
            # take log and normalized to range [-1, 1]
            batch['gpt_cond'] = batch['log_E_inc']
        else:
            if not self.use_vq_cond: # allow gpt-cond is different as vq
                if not self.vq_model.is_ds23:
                    raise NotImplementedError("Error: for ds1 must use E_inc binned as it-is. Not support rebin")
                else:
                    batch['gpt_cond'] = ((torch.log10(self.batch['E_inc'].squeeze(-1))-3)/3*self.cond_bins).long() 
            else:
                # use directly VQ cond
                batch['gpt_cond']=batch['cond']
        return batch
    
    def preprocess(self, batch): 
        # difference between R_true and gpt_R_true ?
        # R_true is the same shape as E
        # gpt_R_true always (N,1)
        if 'gpt_R_true' in batch: 
            return batch # prevent double processing
        
        batch = self.vq_model.preprocess(batch)
        batch = self.preprocess_cond(batch)

        if not self.predict_R:
            return batch
        
        # deal with R_true
        batch['gpt_R_true'] = batch.pop('R_true').squeeze()
        if batch['gpt_R_true'].dim()==1: # back-compatible with no-layer-norm ds1
            batch['gpt_R_true']=batch['gpt_R_true'].unsqueeze(-1)
        elif batch['gpt_R_true'].dim()!=2:
            raise NotImplementedError(f"Wrong R_true dimension {batch['gpt_R_true'].shape}")
        # N,R,Z,A or N,X --> N,Z
        
        # convert R into codes
        if not self.vq_model.layer_seg: # no seg --> only one R
            batch['R_codes'] = self.convertR(batch['gpt_R_true'])
        else:
            batch['gpt_R_unique_trues'] = batch.pop('R_unique_trues') 
            batch['gpt_R_unique_trues'] = [R_layer.squeeze().unsqueeze(-1) for R_layer in batch['gpt_R_unique_trues']] # N,R,Z,A or N,X --> N,Z
            # test adding sum R
            batch['gpt_R_unique_trues'].insert(0,sum(batch['gpt_R_unique_trues']))
            batch['R_codes'] = torch.concat(
                                    [self.convertR(R_layer) 
                                        for R_layer in batch['gpt_R_unique_trues']
                                    ], axis=-1)
        return batch

    def postprocess_codes(self, codes): 
        if isinstance(codes, list): # gpt is generate each code, need concat
            codes = torch.cat(codes, axis=1)

        ret = {}
        if self.predict_R:
            if not self.vq_model.layer_seg: # only global R
                ret['gpt_R_unique_preds'] = self.decodeR(codes[:,:self.R_seq_len]) # GPT_R mustbe 2D
                codes = codes[:,self.R_seq_len:]
                # print("DEB1",ret['gpt_R_unique_preds'].shape,codes.shape)
                ret['R_pred'] = ret['gpt_R_unique_preds'].reshape(
                    -1 , # batch
                    *((1,) * (len(self.sequence_shape) +1 )) # latent (1 dim for ds1 and 2 for ds2/3)  + channel (1)
                    ) # R_pred in vq should be same as input shape # need review.
            else:
                ret['gpt_R_unique_preds']=[] # use layer norm R
                for _ in range(len(self.vq_model.layer_seg)+1):
                    dcodes=codes[:,:self.R_seq_len]
                    ret['gpt_R_unique_preds'].append(self.decodeR(dcodes))  
                    # decodeR could handle multiR but for easily interleave, process one by one
                    codes = codes[:,self.R_seq_len:]
                R_sum = ret['gpt_R_unique_preds'][0]
                R_layers = ret['gpt_R_unique_preds'][1:]
                R_sum_unorm = sum(R_layers)
                R_factor=torch.nan_to_num(R_sum/R_sum_unorm,nan=1)
                R_layers = [R_layer*R_factor for R_layer in R_layers]
                # print("R SHAPE", self.vq_model.layer_seg, len(R_layers),R_layers[0].shape,R_layers[-1].shape)
                # print("R VALUE", R_layers[0],R_layers[-1])
                assert R_layers[0].dim()==2 # (N,Z)
                if self.vq_model.is_ds23:
                    # (N,Z) --> (N,1,Z,1)
                    # later can use vq_model func do that
                    # ret['R_pred'] = torch.cat(R_layers,dim=1).unsqueeze(-2).unsqueeze(-1).expand(ret['R_pred'].shape[0],*self.input_dim)
                    # need list
                    # R_4Dim = torch.cat(R_layers,dim=1).unsqueeze(-2).unsqueeze(-1)
                    # assert R_4Dim.dim()==4
                    R_4Dim = [ R.unsqueeze(-2).unsqueeze(-1) for R in R_layers ]
                    ret['R_pred'] = self.vq_model.interleaveR(R_4Dim)
                else:
                    ret['R_pred'] = self.vq_model.interleaveR(R_layers)

        # recover shape
        ret['codes_pred'] = self.unflatten_codes(codes)

        return ret

    # the reason use two type of codes is to ensure the VQ works as expected!
    # prevent any wired shape bug
    def flatten_codes(self,vq_codes):
        ret=vq_codes.reshape(-1,self.sequence_len)
        # assert ret.shape[0] == batch_size
        return ret
    def unflatten_codes(self,GPT_codes): 
        ret=GPT_codes.reshape(-1,*self.sequence_shape)
        # assert ret.shape[0] == batch_size
        return ret
    
    def trainval_step(self, batch, batch_idx, split):
        self.vq_model.eval()
        with torch.no_grad():
            batch = self.preprocess(batch)
            codes = self.vq_model.encode_codes(batch['pixels_R'], batch['cond']) # codes in sane shape
            # codes = self.vq_model.predict_codes(batch,debug=False)['min_encoding_indices']
            # flatten to GPT codes (N,*)
            # print(codes.shape)
            codes = self.flatten_codes(codes)
            # codes = codes.reshape(batch['E_inc'].shape[0] ,-1) # flatten to (N,*)

            if self.predict_R:
                codes = torch.cat([batch['R_codes'], codes], axis=1) # (*, R_len + H*W)

        idx, targets = codes[:,:-1], codes
        logits, loss = self(batch['gpt_cond'], idx, targets)

        if split == 'train':
            self.log(f"train/loss", loss, on_step=True, on_epoch=True)
        else:
            self.log(f"{split}/loss", loss, on_step=False, on_epoch=True)

        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self.trainval_step(batch, batch_idx, split='train')
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        logits, loss = self.trainval_step(batch, batch_idx, split='val')

        # generate some test pattern
        if      not self.pure_mode \
                and (not self.trainer.running_sanity_check) \
                and self.current_epoch % self.record_freq == 0:
            self.on_record=True
            self.vq_model.on_record=True
            self.vq_model.do_metric=True
            self.vq_model.do_more_metric=True

            batch=self.preprocess(batch)
            R_true = batch['gpt_R_true']
            print()
            if self.vq_model.is_ds23:
                R_true=R_true.unsqueeze(-2).unsqueeze(-1)
                
            # R_true=batch.pop("R_true")
            # isolate the conditions from batch so that ensure it sees nothing truth!
            # cond_gpt = batch['gpt_cond'] 
            # cond_vq = batch['cond']
            
            # generation mode
            gen_post = self.sample_fullchain(batch)

            # reco mode -- only test the forward step without codes lookup
            # add R_true temporarily as 
            batch["R_true"]=R_true
            # print("R_true",R_true[0,0],R_true[0,20],R_true.sum())
            r=self.vq_model(batch,test_code=True)
            # print("DEBUG f1 indicies",r["indices"][100,15],r["indices"][100,21])
            # print("DEBUG f1 R_true",batch["R_true"][100,...].sum())
            reco_post1=self.vq_model.postprocess(batch, r, renorm=self.R_renorm, force_pred=False)
            # reco_post1=self.vq_model.postprocess(batch, self.vq_model(batch,test_code=True), renorm=self.R_renorm, force_pred=False) # test code loopback
            del batch["R_true"]

            # reco mode -- fullinclude codes lookup
            # codes = self.vq_model.predict_codes(batch)['min_encoding_indices']
            # codes = codes.reshape(batch['E_inc'].shape[:1] + (-1,)) # (*, H*W)
            codes = self.vq_model.encode_codes(batch['pixels_R'], batch['cond'])
            # print("codes''[0]",codes[0,])
            # print("codes''[1]",codes[1,...])
            codes = self.flatten_codes(codes)
            codes = torch.cat([batch['R_codes'], codes], axis=1) # (*, R_len + H*W)
            reco=self.postprocess_codes(codes)
            # print("DEBUG f2 indicies",reco["codes_pred"][100,15],reco["codes_pred"][100,21])
            # print("DEBUG f2 R_pred",reco["R_pred"][100,...].sum())
            # print("codes'**[0]",reco["codes_pred"][0,...])
            # print("codes'**[1]",reco["codes_pred"][1,...])
            # print("R_pred2",reco["R_pred"][0,0],reco["R_pred"][0,20],reco["R_pred"].sum())
            reco_post2 = self.vq_model.decode_codes_fullchain(batch, reco, post=True, renorm=self.R_renorm, force_pred=True) # note, R_pred will be first used is decoder learn_R
            # print("===================================")
            # let's accumulate the gen and finally generate sth in the later
            if self.debug_mode==0:
                truth_b={ 
                    "pixels_E_orig":batch["pixels_E_orig"].clone(),
                    "E_inc":batch["E_inc"].clone(),
                }
                self.vq_model.val_accumulate(truth_b, gen_post, do_post=False, renorm=False) # renorm already done in decode_code.
            elif self.debug_mode==1:
                reco_b={ 
                    "pixels_E_orig":reco_post1["pixels_E_pred"].clone(),
                    "E_inc":batch["E_inc"].clone(),
                }
                self.vq_model.val_accumulate(reco_b,gen_post, do_post=False, renorm=False) 
            elif self.debug_mode==2:
                reco_b={ 
                    "pixels_E_orig":reco_post2["pixels_E_pred"].clone(),
                    "E_inc":batch["E_inc"].clone(),
                }
                self.vq_model.val_accumulate(reco_b,gen_post, do_post=False, renorm=False) 
            elif self.debug_mode==-1:
                truth_b={ 
                    "pixels_E_orig":batch["pixels_E_orig"].clone(),
                    "E_inc":batch["E_inc"].clone(),
                }
                self.vq_model.val_accumulate(truth_b,reco_post1, do_post=False, renorm=False) 
            elif self.debug_mode==-2:
                truth_b={ 
                    "pixels_E_orig":batch["pixels_E_orig"].clone(),
                    "E_inc":batch["E_inc"].clone(),
                }
                self.vq_model.val_accumulate(truth_b,reco_post2, do_post=False, renorm=False) 
            elif self.debug_mode==-3:
                reco_b={ 
                    "pixels_E_orig":reco_post1["pixels_E_pred"].clone(),
                    "E_inc":batch["E_inc"].clone(),
                }
                self.vq_model.val_accumulate(reco_b,reco_post2, do_post=False, renorm=False) 
        else:
            self.on_record=False
            self.vq_model.on_record=False

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        return opt
    
    def sample(self, cond, steps=None, temperature=1.0, sample=True, top_k=None): #
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of sequence_len, unlike an RNN that has an infinite context window.
        """
        sequence_len = self.gpt.sequence_len
        if steps is None:
            steps = sequence_len

        with torch.no_grad():
            # input cond = (N,1)
            if self.cond_bins:
                cond = self.lab_emb(cond) # auto add one more dim
                # honestly I don't know why we had a projection at all before?
                if self.cond_proj:
                    # print(self.proj.weight.shape)
                    cond = self.proj(cond)
                    #print('cond_proj', cond.shape)
            else:
                cond = cond.unsqueeze(-2)
            # cond = (N,1,1)
            self.gpt.eval()
            # x = skel_latent
            # print("BAD shape1",cond.shape)
            x = torch.cat([torch.zeros((cond.shape[0], 1, self.n_embd-cond.shape[-1]), device=cond.device), cond], dim=2) # concat every step
            indices = []
            for k in range(steps):
                x_cond = x if x.size(1) <= sequence_len else x[:, -sequence_len:]  # crop context if needed
                # print("cond shape",x_cond.shape)
                logits = self.gpt.sample(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                indices.append(ix)
                emb_ix = self.gpt.tok_emb(ix)
                # emb_ix += skel_latent # add at every steps
                emb_ix = torch.cat([emb_ix, cond], dim=-1) # concat at every steps
                # append to the sequence and continue
                x = torch.cat((x, emb_ix), dim=1)

        return self.postprocess_codes(indices)

    @torch.no_grad()
    def sample_fullchain(self,batch):
        gen = self.sample(batch['gpt_cond']) 
        gen_post = self.vq_model.decode_codes_fullchain(batch, gen, post=True, renorm=self.R_renorm, force_pred=True) # ensure R_pred is used instead of R_true
        return gen_post

    @staticmethod
    def top_k_logits(logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out
    
    @torch.no_grad()
    def log_images(self, batch, inc_calo=True, plot_ema=False, **kwargs): # only_inputs=False, 
        if True:
            # this doesn't work for ds1, just kill it for now
            return {}
        log = dict()
        if self.pure_mode: # in gpt training, the image logging also needs sampling so still slow. it will be disabled by the option
            return log
        batch=self.preprocess(batch)
        # cond_gpt = batch['gpt_cond']
        # cond_vq = batch['cond']
        post = self.sample_fullchain(batch)
        # gen = self.sample(cond_gpt)
        # post = self.vq_model.decode_codes_fullchain(batch, gen, post=True, renorm=self.R_renorm, force_pred=True)
        # post = self.vq_model.decode_code(batch, gen, renorm=self.R_renorm)
        # post = self.vq_model.decode_code(q=gen["codes_pred"], cond=cond, R_true=gen["R_pred"]) # note, R_pred will be first used is decoder learn_R
        post = recursive_to(post , self.device)

        log["references"] = torch.log1p(batch['pixels_E'])/14
        log["generations"] = torch.log1p(post['pixels_E_pred'])/14

        if inc_calo:
            log["references_calo"] = self.vq_model.to_calo(batch['pixels_E'], batch['E_inc'])
            log["generations_calo"] = self.vq_model.to_calo(post['pixels_E_pred'], batch['E_inc'])

        if self.vq_model.use_ema and plot_ema:
            with self.ema_scope():
                post = self.sample_fullchain(batch)
                # gen = self.sample(cond_gpt)
                # post = self.vq_model.decode_codes_fullchain(batch, gen, post=True, renorm=self.R_renorm, force_pred=True)
                # post = self.vq_model.decode_code(batch, gen, renorm=self.R_renorm)
                # post = self.vq_model.decode_code(q=gen["codes_pred"], cond=cond, R_true=gen["R_pred"]) # note, R_pred will be first used is decoder learn_R
                log["generations_ema"] = torch.log1p(post_ema['pixels_E_pred'])/14
        
        return log
