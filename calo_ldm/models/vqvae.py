import torch
import pytorch_lightning as pl

from torch.nn import Conv2d # don't panic, just a convenience for 1x1 convolutions
                            # (a.k.a. Linear layer applied to the -3 axis)
import torch.nn.functional as F

from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from packaging import version

from calo_ldm.layers import VectorQuantizer
from calo_ldm.layers.misc import ZPad, ZUnPad
from calo_ldm.util import instantiate_from_config, recursive_to

from calo_ldm.ema import LitEma

from ..plot import CaloPlotter, HighLevelFeatures2, phyPlotter

import numpy as np

from torchvision.utils import make_grid
import torchvision.transforms as T
import sys

class VQModel(pl.LightningModule):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 loss_config,
                 n_embed,
                 embed_dim,
                 cond_dim=0,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=True, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 z_pad=None,
                 z_padding_strategy=None, # (none, preprocess, internal)
                 # note set to internal z_pad equal to manually use zpad layer directly in conv_spec (remeber to add unpad also)
                 do_metric=False,
                 do_more_metric=False,
                 record_freq=1, # the freq when accumulating the phy metrics. 
                 metric_evts=10000,
                 log_scale_params=(1,3e3,7), # log(a+bx)/c
                 reco_normalization='R', # U, R (or E?)   unit:: all pixels sum to 1; R:: all pixels sum to R; E:: all pixels sum to E_tot (where E_tot = R*E_incident)
                 disc_normalization='R', # U, R (or E?)
                 learn_R=False, # whether or not the decoder network should try to predict R
                 do_R_metric=None, # whether or not the decoder network should try to predict R
                 dec_condition_R=False, # whether or not decoder should be conditioned on R_true
                 dataset_name="1_photon",
                 legacy=False, # legacy mode, e.g. no layer-wise norm
                 ):
        super().__init__()
        self.legacy=legacy
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key

        self.reco_normalization = reco_normalization
        self.disc_normalization = disc_normalization
        self.learn_R=learn_R
        assert not learn_R # now we use GPT learn the R
        self.do_R_metric=do_R_metric if do_R_metric is not None else learn_R
        
        self.record_freq=record_freq
        self.on_record=False
        assert dataset_name in ['1_photon', '1_pion', '2', '3']
        self.is_ds23=(dataset_name not in ['1_photon', '1_pion'])
        if not self.is_ds23:
            self.do_R_metric=False

        self.dataset_name = dataset_name
        if self.legacy:
            self.layer_seg=None
            self.layer_seg_dim=False
            if dataset_name=="2":
                self.raw_input_dim=(45,16,9)
            elif dataset_name=="3":
                self.raw_input_dim=(45,50,18)
        elif dataset_name == "1_pion":
            self.layer_seg=[8,100,100,5,150,160,10]
            self.layer_seg_dim=-1
            # self.raw_input_dim=(533,)
        elif dataset_name == "1_photon":
            self.layer_seg=[8,160,190,5,5]
            self.layer_seg_dim=-1
            # self.raw_input_dim=(368,)
        elif dataset_name=="2":
            self.layer_seg=[1] * 45
            self.layer_seg_dim=-2 # Z
            self.layer_nonseg_dim=[-1,-3] # A, R
            self.raw_input_dim=(45,16,9)
        elif dataset_name=="3":
            self.layer_seg=[1] * 45
            self.layer_seg_dim=-2 # Z
            self.layer_nonseg_dim=[-1,-3] # A, R
            self.raw_input_dim=(45,50,18)
        else:
            self.layer_seg=None
            self.layer_seg_dim=False
        
        if not z_padding_strategy in (None, 'none', 'preprocess', 'internal'):
            raise ValueError(f"Unsupported z_padding_strategy: {z_padding_strategy}")
        self.z_padding_strategy = z_padding_strategy
        if self.z_padding_strategy:
            self.z_pad = ZPad(z_pad)
            self.z_unpad = ZUnPad(z_pad)
        else:
            self.z_pad = None
            self.z_unpad = None

        print("Overwrite mode enabled! the submodule config in yaml or cmdline will be overwriten by the option in main module(params)",file=sys. stderr)
        self.encoder = instantiate_from_config(encoder_config, {
            'cond_dim': cond_dim, "log_scale_params": log_scale_params,
            "z_pad": z_pad, "z_padding_strategy": z_padding_strategy}, overwrite=True)

        decoder_append={
            'cond_dim': cond_dim, "learn_R": learn_R,
            "z_pad": z_pad, "z_padding_strategy": z_padding_strategy,
            #"dec_condition_R": dec_condition_R, TODO
            }
        if self.layer_seg:
            decoder_append["layer_seg"]=list(self.layer_seg)
            decoder_append["layer_seg_dim"]=int(self.layer_seg_dim)
            print("INFO: layer E normalization enabled, must use DecoderMH. seg=",self.layer_seg,"dim=",self.layer_seg_dim)
        self.decoder = instantiate_from_config(decoder_config, decoder_append, overwrite=True)

        self.loss = instantiate_from_config(loss_config, {
            'cond_dim': cond_dim, 'n_embed': n_embed, 
            "log_scale_params": log_scale_params,
            "learn_R": learn_R,
            "disc_normalization": disc_normalization,
            "reco_normalization": reco_normalization,
            "unpad_z": z_pad if z_padding_strategy=="preprocess" else None,
            'dataset_name': dataset_name,
            }, overwrite=True)

        assert sane_index_shape # now we only support meaning ful codes shape instead of flattened one
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape,
                                        pixels_dim=(1 if not self.is_ds23 else 3)
                                        )
        if not self.is_ds23:
            self.quant_conv = torch.nn.Conv1d(self.encoder.ch_out, embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv1d(embed_dim, self.encoder.ch_out, 1)
        else:
            self.quant_conv = torch.nn.Conv2d(self.encoder.ch_out, embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.encoder.ch_out, 1)

        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}. (not used in calo)")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        # make plot need
        self.plotter=CaloPlotter(ds=dataset_name)
        self.HLF=HighLevelFeatures2(ds=dataset_name)
        self.HLF_ref=HighLevelFeatures2(ds=dataset_name)

        self.t__x=None # numpy obj growing
        self.t__xrec=None
        self.t_y=None

        self.METRICS=None
        self.METRICS_REF=None

        self.resizer=T.Resize(1200,antialias=True)

        self.phyP=phyPlotter(virtual=True, ds=dataset_name)

        self.do_metric=do_metric
        self.do_more_metric=do_more_metric # do the metric of each layers. wil be slow.
        if self.do_more_metric:
            print("Warning detailed metrics will be recorded. Watch your clock.")
        self.metric_evts=metric_evts

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x, cond):
        h = self.encoder(x, cond)
        h = self.quant_conv(h)
        ret = self.quantize(h)
        return ret

    # def predict_codes(self, batch,debug=True): # we move to better functions
    #     batch = self.preprocess(batch)
    #     if "R_true" in batch:
    #         del batch["R_true"] # to remove the R_true when generation
    #     ret = self.encode(batch['pixels_R'], batch['cond'])
    #     # if debug:
    #         # print("[red]enc2[/red]",ret["quant"][0,0,0],ret["min_encoding_indices"][0],batch['cond'][0,0])
    #     return ret

    @torch.no_grad() # this is not used in training
    def encode_codes(self, x, cond):
        h = self.encoder(x, cond)
        h = self.quant_conv(h)
        ret = self.quantize(h)
        return ret["min_encoding_indices"]

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, cond):
        # print("[red]zb1[/red]",quant.shape, quant.type(), quant.sum(), quant[0,...])
        quant1 = self.post_quant_conv(quant) # n_embe(256) --> n_enc_output (128)
        # print("[red]zc1[/red]",quant1[0,0,0],quant1.sum())
        ret = self.decoder(quant1, cond)
        return ret
    
    @torch.no_grad() # the get_codebook_entry step should not be BP
    def decode_codes(self, codes, cond):
        # print("[red]codes[/red]",codes.shape)
        assert codes.dim() != 1 # now we only support sane shape of codes (to prevent any bug)
        z = self.quantize.get_codebook_entry(codes) # 256=N, 255=, 69=1+4*17 # , (self.embed_dim,) + codes.shape
        # print("[red]z0[/red]",z.shape)
        z = self.post_quant_conv(z) # n_embe(256) --> n_enc_output (128)
        # print("[red]z1[/red]",z.shape)
        # print("[red]cond[/red]",cond.shape)
        ret=self.decoder(z, cond) # {pixels_pred_U }
        return ret

    # @torch.no_grad() 
    # def decode_code(self, batch, pred, post=True, renorm=True, force_pred=False): # TODO: make it simpler.
    #     #quant_b = self.quantize.embed_code(code_b)
    #     #q=q.unflatten(-1,(17,4)) 
    #     q = pred['codes_pred']
    #     ret = self.decode_codes(q,batch['cond'])
    #     # # print("[red]za2[/red]",q.shape, q)
    #     # # print("[bold magenta]decode code in[/bold magenta]",q.shape)
    #     # z = self.quantize.get_codebook_entry(q, (self.embed_dim,) + q.shape) # 256=N, 255=, 69=1+4*17
    #     # # print("[red]zb2[/red]",z.shape, z.type(), z.sum(), z[0,...])
    #     # z = self.post_quant_conv(z) # n_embe(256) --> n_enc_output (128)
    #     # # print("[red]zc2[/red]",z1[0,0,0],z1.sum())
    #     # # print("cond",batch['cond'].shape)
    #     # # print("[bold magenta]decode code out[/bold magenta]",z.shape)
    #     pred.update(ret) # add 
    #     # print("[red]pred2[/red]",pred["pixels_U_pred"][0,0],batch['cond'][0,0])
    #     if not post:
    #         return pred
    #     post = self.postprocess(batch, pred, force_pred=force_pred)
    #     # do renorm:
    #     if self.is_ds23 and renorm:
    #         post=self.renorm_R(post)
    #     return post

    @torch.no_grad() 
    def decode_codes_fullchain(self, batch, pred, post=True, renorm=True, force_pred=False): # TODO: make it simpler.
        q = pred['codes_pred']
        ret = self.decode_codes(q,batch['cond'])
        pred.update(ret) # add 
        if not post:
            return pred
        post = self.postprocess(batch, pred, renorm=renorm, force_pred=force_pred)
        # # do renorm: already in the postprocess
        # if self.is_ds23 and renorm:
        #     post=self.renorm_R(post)
        return post

    def forward(self, batch, test_code=False):
        enc = self.encode(batch['pixels_R'], batch['cond'])
        # print("[red]enc1, za1[/red]",enc["quant"][0,0,0],enc["min_encoding_indices"].shape,enc["min_encoding_indices"],batch['cond'][0,0])
        if test_code:
            # run loopback test, pass through actual codes, might be slow but prevent bug
            codes = enc["min_encoding_indices"]
            pred = self.decode_codes(codes, batch['cond'])
            # print("codes'[0]",codes[0,...])
            # print("codes'[1]",codes[1,...])
        else:
            pred = self.decode(enc["quant"], batch['cond'])
        # print("[red]pred1[/red]",pred["pixels_U_pred"][0,0],batch['cond'][0,0])
        pred['qloss'] = enc["qloss"]
        pred['indices'] = enc["min_encoding_indices"]
        # print("codes[0]",pred['indices'][0,...])
        # print("codes[1]",pred['indices'][1,...])
        R = pred.get('R_pred', batch['R_true'])
        assert R.dim()==4 or R.dim()==2
        # if 'R' in (self.reco_normalization, self.disc_normalization):
        if True: # always need this since loss recording
            # force broadcast R
            pred['pixels_R_pred'] = pred['pixels_U_pred'] * R
            pred['pixels_E_pred'] = pred['pixels_R_pred'] * batch["E_inc"][...,None,None] # how about ds1??
        if 'E' in (self.reco_normalization, self.disc_normalization):
            raise NotImplementedError("Un-normalized (raw) not supported now for network input")
        return pred
    
    def interleaveR(self,unique_Rs): #  repeat R for each layer. [R1,R2,...] --> [R1,R1,R1,R2,R2,...]
        layer_Rs = torch.cat(
                    unique_Rs, self.layer_seg_dim
                ) 
        return torch.repeat_interleave(
            layer_Rs,torch.tensor(
                self.layer_seg,
                dtype=torch.int,device=layer_Rs.device), 
                dim=self.layer_seg_dim, 
                output_size=sum(self.layer_seg)) 
    
    def preprocess_cond(self,batch):
        if not self.is_ds23:
            batch['log_E_inc'] = 2*(torch.log2(batch['E_inc'])-8)/14-1 # [-1,1]
            # now we do not consider interoplation
            batch["cond"] = batch['E_inc_binned'] # 'E_inc_binned' is special for ds1, generate from dataloader
            # if force_cond_bins or self.encoder.cond_bins:
            #     batch["cond"] = batch['E_inc_binned']
            # else:
            #     batch["cond"] = batch['log_E_inc']
        else:
            batch['log_E_inc'] = 2*(torch.log10(batch['E_inc'])-3)/3-1 # in range [-1,1] 
            batch['cond'] = batch['log_E_inc'] # ds2/3 should not use binned cond
        
        return batch

    def preprocess(self, batch):
        if 'pixels_E_orig' in batch:
            return batch
        # we ensure E,R,U all in same dimension. 
        # E_inc will always be N,1 and will be broadcast, same as cond
        if not self.is_ds23:
            batch['pixels_E_orig'] = batch['pixels_E'].clone() 
            batch['pixels_R'] = batch['pixels_E'] / batch['E_inc']
            batch = self.preprocess_cond(batch)
            
            if not self.layer_seg:
                batch['R_true'] = batch['pixels_R'].sum(axis=-1, keepdims=True)
            else:
                layer_Es=torch.split(batch['pixels_R'], self.layer_seg, dim=self.layer_seg_dim) # [(N,X),]
                batch["R_unique_trues"] = [Es.sum(axis=self.layer_seg_dim, keepdims=True) # [(N,1),]
                                                for Es in layer_Es]
                batch['R_true'] = self.interleaveR(batch["R_unique_trues"])
                pass

            batch['pixels_U'] = torch.nan_to_num(
                batch['pixels_R'] / batch['R_true'],
                1./batch['pixels_R'].shape[-1]) 

            return batch

        batch = self.preprocess_cond(batch)

        if 'pixels_E' in batch and 'pixels_E_orig' not in batch: # determine whether need to renorm (fix z_padding)
            batch['pixels_E_orig'] = batch['pixels_E'].clone() 
            batch['pixels_R_orig'] = batch['pixels_E'] / batch['E_inc'][...,None,None] # Einc broadcast
            if not self.layer_seg:
                batch['R_true'] = batch['pixels_R_orig'].sum(axis=(-1,-2,-3),keepdims=True) # keep same dim (but singleton)   
                # batch['pixels_U_orig'] = torch.where( batch['R_true']>0,
                #     batch['pixels_R_orig'] / batch['R_true'],
                #     torch.ones_like(batch['pixels_R_orig']) / np.prod(batch['pixels_R_orig'].shape[-3:])
                #     )
                # simple way
                batch['pixels_U_orig'] = torch.nan_to_num(
                    batch['pixels_R_orig'] / batch['R_true'],
                    1./np.prod(batch['pixels_R_orig'].shape[-3:]))            
            else:
                R_singletons = batch['pixels_R_orig'].sum(axis=self.layer_nonseg_dim, keepdims=True) #(N,1,Z,1)
                # if seg!=[1]*N
                R_grouped = torch.split(R_singletons, self.layer_seg, dim=self.layer_seg_dim) # (N,1,Z',1)
                batch["R_unique_trues"] = [Es.sum(axis=self.layer_seg_dim, keepdims=True) # [(N,1,1,1),]
                                                for Es in R_grouped]
                # print("DEBUG Rs",len(batch["R_unique_trues"]),sum(batch["R_unique_trues"]))
                batch['R_true'] = self.interleaveR(batch["R_unique_trues"]) # broadcast ok
                # if seg==[1]*N, use simple expand is enough
                # batch['R_true'] = R_singletons.expand(*(batch['pixels_R_orig'].shape)) #(N,R,Z,A)
                # batch["R_unique_trues"] = torch.split(R_singletons, self.layer_seg, dim=self.layer_seg_dim) # [(N,1,1,1)]
                batch['pixels_U_orig'] = torch.nan_to_num(
                    batch['pixels_R_orig'] / batch['R_true'],
                    1./np.prod(self.layer_nonseg_dim))
            
            if self.z_padding_strategy == 'preprocess': # NRZA
                batch['pixels_U'] = self.z_pad(batch['pixels_U_orig'])
                batch['pixels_R'] = self.z_pad(batch['pixels_R_orig'])
                batch['pixels_E'] = self.z_pad(batch['pixels_E_orig'])
            else:
                batch['pixels_U'] = batch['pixels_U_orig']
                batch['pixels_R'] = batch['pixels_R_orig']
                batch['pixels_E'] = batch['pixels_E_orig']

        return batch

    @torch.no_grad()  
    def postprocess(self, batch, pred, renorm=True, force_pred=False):
        post = {}
        if 'R_pred' in pred:
            R = pred['R_pred']
        else:
            if force_pred:
                raise NotImplementedError(f"Error: no R_pred but requested to")
            R = batch['R_true']
        #Einc=batch.get("E_inc", torch.pow(10.,(batch["log_E_inc"]+1)*3/2+3))
        if not 'E_inc' in batch:
            raise NotImplementedError("Dangerous, just make sure it's in batch before calling.")
            Einc = torch.pow(10., (batch['log_E_inc']+1)*3/2+3)
        else:
            Einc = batch['E_inc']

        post['pixels_U_pred'] = pred['pixels_U_pred']
        if len(Einc.shape) < 2:
            Einc = Einc.unsqueeze(-1)
        post['pixels_R_pred'] = pred['pixels_U_pred']*R
        if not self.is_ds23:
            post['pixels_E_pred'] = post['pixels_R_pred']*Einc
        else:
            post['pixels_E_pred'] = post['pixels_R_pred']*Einc[...,None,None]

        if self.z_padding_strategy == 'preprocess': # NRZA
            for k in post:
                # unpad all of the pixel things
                post[k] = self.z_unpad(post[k])
        
        if self.is_ds23 and renorm: # renorm only defined on ds2 and 3 and when 
            # use with caution since now it is in (layer) partition norm mode
            post=self.renorm_R(post)
        
        return post

    @torch.no_grad()  
    def renorm_R(self,post):
        if self.layer_seg or self.z_padding_strategy!="preprocess": 
            # when layer-wise normalization, no need to renorm to correct the z_pad issue
            # when z_pad is internal (e.g. encoder+decoder, decoder already unpad using layer). no need to renornm
            return post
        factor = post["pixels_U_pred"].detach().sum(axis=(-1,-2,-3),keepdims=True)
        post["pixels_U_pred"]=post["pixels_U_pred"].detach() / factor
        if "pixels_E_pred" in post:
            post["pixels_E_pred"]=post["pixels_E_pred"].detach() / factor
        if "pixels_R_pred" in post:
            post["pixels_R_pred"]=post["pixels_R_pred"].detach() / factor
        return post

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        b = self.preprocess(batch) # , self.image_key

        pred = self(b)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(b, pred, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False) # could not both log at step and epoch!! default is auto.
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(b, pred, optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        b = self.preprocess(batch)
        pred = self(b,test_code=True) # we test code lookup process
        aeloss, log_dict_ae = self.loss(b, pred,
                                        0, self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        )
        discloss, log_dict_disc = self.loss(b, pred,
                                            1, self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # collect results
        if self.do_metric \
            and (not self.trainer.running_sanity_check) \
            and (self.current_epoch % self.record_freq==0):
            self.on_record=True
            self.val_accumulate(batch, pred)
        else:
            self.on_record=False

        return self.log_dict

    @torch.no_grad()    
    def val_accumulate(self,batch, pred, do_post=True, renorm=True):
        
        if ((self.t__xrec is None) or (self.t__xrec.shape[0] < self.metric_evts)):
            post = self.postprocess(batch, pred, renorm=renorm) if do_post else pred
            # if renorm and self.is_ds23: # for better plotting. 
            #     post=self.renorm_R(post)
            # print("VSHAPE",post['pixels_E_pred'].shape,batch['pixels_E_orig'].shape,batch['E_inc'].shape)
            __xrec=post['pixels_E_pred'].detach().cpu()
            self.t__xrec=__xrec if self.t__xrec is None else torch.cat((self.t__xrec, __xrec), 0)
            # in validation, the x_truth and cond won't change of different epoch!
            if self.METRICS_REF is None:
                __x=batch['pixels_E_orig'].detach().cpu()
                self.t__x=__x if self.t__x is None else torch.cat((self.t__x, __x), 0)
                __y=batch['E_inc'].detach().cpu()
                self.t_y=__y if self.t_y is None else torch.cat((self.t_y, __y), 0)
            return True
        return False # buffer full, stop accumulating

    @torch.no_grad()
    def calMstrics(self): # tf histos
        if not self.do_metric or not self.on_record:
            return
        if self.t__xrec is None:
            print("Error! No accumulating validation data. Stop metric cal...",file=sys.stderr)
            return
        self.METRICS={
            "log10_Cond": torch.log10(1+self.t_y[...,0]),
            "log10_Etot": torch.log10(1+torch.sum(self.t__xrec,axis=(-1,-2,-3))),
        }
        if self.do_R_metric:
            self.METRICS.update({
                "R": torch.sum(self.t__xrec,axis=(-1,-2,-3))/self.t_y[...,0],
                "R_diff": torch.sum(self.t__xrec,axis=(-1,-2,-3))/self.t_y[...,0] - torch.sum(self.t__x,axis=(-1,-2,-3))/self.t_y[...,0],
            })
        # send tensor to numpy
        self.__xrec=self.t__xrec.permute((0,2,3,1)).reshape(-1,np.prod(self.raw_input_dim)).cpu().numpy()
        self.HLF.CalculateFeatures(self.__xrec)
        if self.METRICS_REF is None: # only do once
            self._y=self.t_y.cpu().numpy()
            self.HLF.Einc=self._y

        temp={
            "Elayers": self.make_metrics("Elayers",self.HLF.GetElayers(),log=True),
            "ECEtas": self.make_metrics("ECEtas",self.HLF.GetECEtas()),
            "ECPhis": self.make_metrics("ECPhis",self.HLF.GetECPhis()),
            "WidthEtas": self.make_metrics("WidthEtas",self.HLF.GetWidthEtas()),
            "WidthPhis": self.make_metrics("WidthPhis",self.HLF.GetWidthPhis()),
        }
        for k,v in temp.items():
            self.METRICS.update(v)
        del temp

        if self.METRICS_REF is None:
            self.METRICS_REF={
                "log10_Etot": torch.log10(1+torch.sum(self.t__x,axis=(-1,-2,-3))),
            }
            self.__x=self.t__x.permute((0,2,3,1)).reshape(-1,np.prod(self.raw_input_dim)).cpu().numpy()
            self.HLF_ref.Einc=self._y
            self.HLF_ref.CalculateFeatures(self.__x)

            temp={
                "Elayers": self.make_metrics("Elayers",self.HLF_ref.GetElayers(),log=True),
                "ECEtas": self.make_metrics("ECEtas",self.HLF_ref.GetECEtas()),
                "ECPhis": self.make_metrics("ECPhis",self.HLF_ref.GetECPhis()),
                "WidthEtas": self.make_metrics("WidthEtas",self.HLF_ref.GetWidthEtas()),
                "WidthPhis": self.make_metrics("WidthPhis",self.HLF_ref.GetWidthPhis()),
                }
            for k,v in temp.items():
                self.METRICS_REF.update(v)
            del temp
    
    @torch.no_grad()
    def calMstrics_DS1(self): # tf histos
        if not self.do_metric or not self.on_record:
            return
        if self.t__xrec is None:
            print("Error! No accumulating validation data. Stop metric cal...",file=sys.stderr)
            return
        self.METRICS={
            "log10_Cond": torch.log10(1+self.t_y[...,0]),
            "log10_Etot": torch.log10(1+torch.sum(self.t__xrec,axis=(-1))),
        }
        if self.do_R_metric:
            self.METRICS.update({
                "R": torch.sum(self.t__xrec,axis=(-1))/self.t_y[...,0],
                "R_diff": torch.sum(self.t__xrec,axis=(-1))/self.t_y[...,0] - torch.sum(self.t__x,axis=(-1))/self.t_y[...,0],
            })
        # send tensor to numpy
        self.__xrec=self.t__xrec.cpu().numpy()
        self.HLF.CalculateFeatures(self.__xrec)
        if self.METRICS_REF is None: # only do once
            self._y=self.t_y.cpu().numpy()
            self.HLF.Einc=self._y

        temp={
            "Elayers": self.make_metrics("Elayers",self.HLF.GetElayers(),log=True),
            "ECEtas": self.make_metrics("ECEtas",self.HLF.GetECEtas()),
            "ECPhis": self.make_metrics("ECPhis",self.HLF.GetECPhis()),
            "WidthEtas": self.make_metrics("WidthEtas",self.HLF.GetWidthEtas()),
            "WidthPhis": self.make_metrics("WidthPhis",self.HLF.GetWidthPhis()),
        }
        for k,v in temp.items():
            self.METRICS.update(v)
        del temp

        if self.METRICS_REF is None:
            self.METRICS_REF={
                "log10_Etot": torch.log10(1+torch.sum(self.t__x,axis=(-1))),
            }
            self.__x=self.t__x.cpu().numpy()
            self.HLF_ref.Einc=self._y
            self.HLF_ref.CalculateFeatures(self.__x)

            temp={
                "Elayers": self.make_metrics("Elayers",self.HLF_ref.GetElayers(),log=True),
                "ECEtas": self.make_metrics("ECEtas",self.HLF_ref.GetECEtas()),
                "ECPhis": self.make_metrics("ECPhis",self.HLF_ref.GetECPhis()),
                "WidthEtas": self.make_metrics("WidthEtas",self.HLF_ref.GetWidthEtas()),
                "WidthPhis": self.make_metrics("WidthPhis",self.HLF_ref.GetWidthPhis()),
                }
            for k,v in temp.items():
                self.METRICS_REF.update(v)
            del temp
    
    @torch.no_grad()
    def make_grid(self,l,nrow=5):
        S=[torch.from_numpy(d).permute((2,0,1)) for d in l]
        grid=make_grid(S,nrow=nrow)
        return self.resizer(grid)
    
    @torch.no_grad()
    def make_img(self,d):
        return torch.from_numpy(d).permute((2,0,1))
    
    @torch.no_grad()
    def make_metric(self,d):
        return torch.from_numpy(d)
    
    @torch.no_grad()
    def make_metrics(self,b,l, log=False):
        if log:
            return {f"{b}_{k}":torch.log10(1+torch.from_numpy(d)) for k,d in l.items()}
        else:
            return {f"{b}_{k}":torch.from_numpy(d) for k,d in l.items()}
    
    @torch.no_grad()
    def getMetricsHists(self): # move out to callback??
        if not self.do_metric or not self.on_record:
            return {},{}
        if self.METRICS is None:
            print("Error! No metrics accumulated during validation check your codes!",file=sys.stderr)
            return {},{}
        pics={}
        chi2={}
        if self.do_more_metric: # note this would be slow...
            for k,f in zip ([   "E_layers","Enorm_layers",
                                "ECEtas",
                                "ECPhis",
                                "ECWidthEtas",
                                "ECWidthPhis"], 
                            [   self.phyP.plot_E_layers,self.phyP.plot_Enorm_layers,
                                self.phyP.plot_ECEtas,
                                self.phyP.plot_ECPhis,
                                self.phyP.plot_ECWidthEtas,
                                self.phyP.plot_ECWidthPhis ]):
                r=f(self.HLF,self.HLF_ref)
                pics[k]=self.make_grid(r[0])
                for i,c in enumerate(r[1]):
                    chi2[f"{k}_{i}"]=torch.tensor(c)
                chi2[f"avg_{k}"]=torch.tensor(np.array(r[1]).mean())
                chi2[f"std_{k}"]=torch.tensor(np.array(r[1]).std())
                chi2[f"min_{k}"]=torch.tensor(np.array(r[1]).min())
                chi2[f"max_{k}"]=torch.tensor(np.array(r[1]).max())
        r=self.phyP.plot_Etot_Einc(self.HLF,self.HLF_ref)
        pics["Etot_Einc"]=self.make_img(r[0])
        chi2["Etot_Einc"]=torch.tensor(r[1])
    
        r=self.phyP.plot_cell_dist(self.__xrec,self.__x)
        pics["cell_dist"]=self.make_img(r[0])
        chi2["cell_dist"]=torch.tensor(r[1])

        r=self.phyP.plot_cellnorm_dist(self.__xrec,self.__x)
        pics["cellnorm_dist"]=self.make_img(r[0])
        chi2["cellnorm_dist"]=torch.tensor(r[1])

        r=self.phyP.plot_cellnormlin_dist(self.__xrec,self.__x)
        pics["cellnormlin_dist"]=self.make_img(r[0])
        chi2["cellnormlin_dist"]=torch.tensor(r[1])

        pics["AverageRec"]=self.make_img(self.HLF.DrawAverageShower(data=self.__xrec))
        
        # get R_0p78_pred
        if self.do_R_metric:
            R = torch.sum(self.t__xrec,axis=(-1,-2,-3))/self.t_y[...,0]
            chi2["R_lt0p7"]= ( R < 0.7 ).float().mean()
            chi2["R_gt0p8"]= ( R > 0.8 ).float().mean()
            R_ref = torch.sum(self.t__x,axis=(-1,-2,-3))/self.t_y[...,0]
            chi2["R_lt0p7_ref"]= ( R_ref < 0.7 ).float().mean()
            chi2["R_gt0p8_ref"]= ( R_ref > 0.8 ).float().mean()
        if self.METRICS_REF is None: # only run once
            pics["AverageRef"] = self.make_img(self.HLF_ref.DrawAverageShower(data=self.__x))
        return pics, chi2
    
    @torch.no_grad()
    def getMetrics(self):
        if not self.do_metric:
            return {},{}
        return self.METRICS,self.METRICS_REF
    
    @torch.no_grad()
    def cleanMetrics(self):
        if not self.do_metric or not self.on_record:
            return 
        self.HLF.reset()
        # self.HLF_ref.reset()
        del self.METRICS
        self.METRICS=None
        # del self.METRICS_REF
        # self.METRICS_REF=None
        # delete numpys
        # del self._y
        # del self.__x
        del self.__xrec
        # clear tensors
        # del self.t_y
        # del self.t__x
        del self.t__xrec
        # reset all
        # self._y=None
        # self.__x=None
        self.__xrec=None
        # self.t_y=None
        # self.t__x=None
        self.t__xrec=None
        
    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.get_adaptive_layer_weights()

    @torch.no_grad()
    def log_images(self, batch, inc_calo=True, plot_ema=False, **kwargs): # only_inputs=False, 
        if not self.is_ds23:
            # not implemented!!
            return {}
        log = dict()
        b = self.preprocess(batch) # , self.image_key
        b = recursive_to(b, self.device)

        pred = self(b)
        post = recursive_to(self.postprocess(batch, pred), self.device) # no need force renorm since the plotting not sensitive to Etot/R

        log["inputs"] = torch.log1p(b['pixels_E'])/14
        log["reconstructions"] = torch.log1p(post['pixels_E_pred'])/14

        if inc_calo:
            log["inputs_calo"] = self.to_calo(b['pixels_E'], b['E_inc'])
            log["reconstructions_calo"] = self.to_calo(post['pixels_E_pred'], b['E_inc'])

        if self.use_ema and plot_ema:
            with self.ema_scope():
                pred_ema = self(b)
                post_ema = self.postprocess(batch, pred_ema)
                log["reconstructions_ema"] = torch.log1p(post_ema['pixels_E_pred'])/14
        return log

    @torch.no_grad()
    def to_calo(self, _x, _y): # direactly change to the calo image
        if not self.is_ds23:
            raise NotImplementedError("Indivadual plotting not support ds1 yet")
        _x = _x.detach().cpu().numpy()
        _y = _y.detach().cpu().numpy() # it is in ZAR order
        x=_x/_y[...,None,None] # broadcast the y(Einc); or can we ensure Einc ~ E shape
        r=self.plotter.draw(x,_y, virtual=True)[None,...]# HWC
        r=torch.from_numpy(r).permute((0,3,1,2))
        return r # NCHW

class Interface(VQModel):
    def encode(self, x, cond):
        h = self.encoder(x, cond)
        h = self.quant_conv(h)
        return h
    
    def decode(self, h, force_not_quantize=False, cond=None):
        # print("DECODE trying to decode h", h.shape)
        if force_not_quantize:
            quant = h
        else:
            ret = self.quantize(h)
            quant=ret["quant"]

        quant = self.post_quant_conv(quant)
        pred = self.decoder(quant, cond)
        return pred

    def encode_cond(self, xc):
        return xc

    def decode_cond(self, xc):
        return xc
