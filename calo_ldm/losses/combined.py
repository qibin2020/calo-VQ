import torch
from torch import nn
import pytorch_lightning as pl
import sys

from calo_ldm.util import (
        instantiate_from_config,
        hinge_d_loss, vanilla_d_loss,
        adopt_weight, measure_perplexity,
        )
from calo_ldm.layers.misc import ZUnPad
from .cluster_perceptual import ClusterPerceptualLoss
import numpy as np

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


class CombinedLoss(pl.LightningModule):
    def __init__(self, disc_start, codebook_weight=1.0, 
                 disc_factor=1.0, disc_weight=1.0,
                #  use_actnorm=False, disc_in_channels=3, disc_conditional=False, disc_num_layers=3, disc_ndf=64,# legacy
                 disc_loss="hinge", n_embed=None,
                  
                 pixel_power=2,
                 pixel_weight=1, 

                 R1_weight=0, 
                 R2_weight=0,
                 ec_weight=0,
                 width_weight=0,

                 disc_config=None,
                 perceptual_config=None,
                 perceptual_weight=0.,
                 cond_dim=0,
                 log_scale_params=None,
                 reco_normalization=None,
                 disc_normalization=None,
                #  learn_R=True,
                 unpad_z=None,
                 dataset_name=None,
                 adaptive_max=1e4,
                 adaptive_min=0.0,
                 **kws
                 ): # , prepro="log"
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_power=pixel_power
        self.pixel_weight = pixel_weight
        self.R1_weight=R1_weight
        self.R2_weight=R2_weight
        self.reco_normalization = reco_normalization
        self.disc_normalization = disc_normalization
        # self.learn_R = learn_R
        self.ec_weight = ec_weight
        self.width_weight = width_weight
        self.adaptive_max=adaptive_max
        self.adaptive_min=adaptive_min

        self.discriminator = instantiate_from_config(
            disc_config, {
                'cond_dim': cond_dim, 
                "log_scale_params": log_scale_params
                }, overwrite=False)
        self.discriminator_iter_start = disc_start

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"CALORECOWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        # self.disc_conditional = disc_conditional
        self.n_embed = n_embed

        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight and perceptual_config is not None:
            self.perceptual_loss = instantiate_from_config(perceptual_config)
        
        self.unpad_z=unpad_z
        self.unpad_fn=ZUnPad(unpad_z) if unpad_z else None
        assert dataset_name in ['1_photon', '1_pion', '2', '3']
        self.dataset_name=dataset_name
        self.is_ds23=True
        if dataset_name == "2" : 
            alpha = np.linspace(0,2*np.pi,16, endpoint=False)
            r = (np.arange(9)*1.5) * 4.65 # mm
        elif dataset_name == "3" : 
            alpha = np.linspace(0,2*np.pi,50, endpoint=False)
            r = (np.arange(18)*1.5)*4.65/2 # mm
        elif dataset_name == "1_photon":
            # binning_dataset_1_photons.xml
            # R first, e.g. R1,A1 R2,A1 R2,A1....
            f_alpha=lambda r,a:np.repeat(np.linspace(0,2*np.pi,a, endpoint=False),r).tolist()
            alpha=[0]*8 + f_alpha(16,10) + f_alpha(19,10) + [0]*5 + [0]*5
            f_r=lambda l:[(i+j)/2. for i,j in zip(l[:-1],l[1:])]
            r=f_r([0,5,10,30,50,100,200,400,600]) \
                + f_r([0,2,4,6,8,10,12,15,20,30,40,50,70,90,120,150,200])*10 \
                + f_r([0,2,5,10,15,20,25,30,40,50,60,80,100,130,160,200,250,300,350,400])*10 \
                + f_r([0,50,100,200,400,600]) \
                + f_r([0,100,200,400,1000,2000]) 
            self.is_ds23=False
        elif dataset_name == "1_pion":
            # R first, e.g. R1,A1 R2,A1 R2,A1....
            f_alpha=lambda r,a:np.repeat(np.linspace(0,2*np.pi,a, endpoint=False),r).tolist()
            alpha=[0]*8 \
                + f_alpha(10,10) \
                + f_alpha(10,10) \
                + [0]*5 \
                + f_alpha(15,10) \
                + f_alpha(16,10) \
                + [0]*10 
            f_r=lambda l:[(i+j)/2. for i,j in zip(l[:-1],l[1:])]
            r=f_r([0,5,10,30,50,100,200,400,600]) \
                + f_r([0,1,4,7,10,15,30,50,90,150,200])*10 \
                + f_r([0,5,10,20,30,50,80,130,200,300,400])*10 \
                + f_r([0,50,100,200,400,600])\
                + f_r([0,10,20,30,50,80,100,130,160,200,250,300,350,400,1000,2000])*10 \
                + f_r([0,10,20,30,50,80,100,130,160,200,250,300,350,400,600,1000,2000])*10 \
                + f_r([0,50,100,150,200,250,300,400,600,1000,2000])
            self.is_ds23=False
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'.")
        if self.is_ds23:
            rr, aa = np.meshgrid(alpha,r) # (9,16)
            eta = rr*np.cos(aa) # (9,16)
            phi = rr*np.sin(aa)
            self.register_buffer('eta_grid', torch.tensor(eta)[None,:,None,:].float()) # (1, R, 1, A)
            self.register_buffer('phi_grid', torch.tensor(phi)[None,:,None,:].float()) # ""
        else:
            eta = r*np.cos(alpha) 
            phi = r*np.sin(alpha)
            self.register_buffer('eta_grid', torch.tensor(eta)[None,:].float()) # (1, N)
            self.register_buffer('phi_grid', torch.tensor(phi)[None,:].float()) # ""

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, self.adaptive_min, self.adaptive_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, batch, pred, optimizer_idx, global_step, last_layer=None, split='train'):
        codebook_loss = pred['qloss']

        reco_pix_true = batch[f'pixels_{self.reco_normalization}']
        reco_pix_pred = pred[f'pixels_{self.reco_normalization}_pred']

        l1_loss = torch.abs(reco_pix_true - reco_pix_pred)
        if self.is_ds23: # 1+3D input for ds2/3
            l2_loss = torch.square(l1_loss).sum(axis=(-1,-2,-3)).mean()
            l1_loss = l1_loss.sum(axis=(-1,-2,-3)).mean()
        else: # 1+1D input for ds1
            l2_loss = torch.square(l1_loss).sum(axis=(-1)).mean()
            l1_loss = l1_loss.sum(axis=(-1)).mean()
        
        if self.pixel_power == 1:
            rec_loss = l1_loss
        elif self.pixel_power == 2:
            rec_loss = l2_loss
        else:
            lp_loss = torch.pow(torch.abs(reco_pix_true - reco_pix_pred), self.pixel_power).sum(axis=-1).mean()
            rec_loss = lp_loss

        energy_pred = pred["pixels_E_pred"]
        if self.unpad_fn:
            energy_pred =  self.unpad_fn(energy_pred) # no need for renorm since it is a very small shift to total energy
        energy_true = batch["pixels_E_orig"] # (*, 9, 45, 16)
        sum_axis=(-1,-3) if self.is_ds23 else (-1,)
        energy_sum_pred=energy_pred.sum(axis=sum_axis) # (*, 45)
        energy_sum_true=energy_true.sum(axis=sum_axis) # (*, 45)
        eta_EC_true = (self.eta_grid * energy_true).sum(axis=sum_axis)/(energy_sum_true+1e-16) # (*, 45)
        phi_EC_true = (self.phi_grid * energy_true).sum(axis=sum_axis)/(energy_sum_true+1e-16)

        eta_square_true = (self.eta_grid * self.eta_grid * energy_true).sum(axis=sum_axis)/(energy_sum_true+1e-16)
        phi_square_true = (self.phi_grid * self.phi_grid * energy_true).sum(axis=sum_axis)/(energy_sum_true+1e-16)

        eta_width_true = torch.sqrt((eta_square_true - eta_EC_true**2).clip(min=0.))
        phi_width_true = torch.sqrt((phi_square_true - phi_EC_true**2).clip(min=0.))


        eta_EC_pred = (self.eta_grid * energy_pred).sum(axis=sum_axis)/(energy_sum_pred+1e-16) # (*,45)
        phi_EC_pred = (self.phi_grid * energy_pred).sum(axis=sum_axis)/(energy_sum_pred+1e-16) # (*,45)
        eta_square_pred = (self.eta_grid * self.eta_grid * energy_pred).sum(axis=sum_axis)/(energy_sum_pred+1e-16)
        phi_square_pred = (self.phi_grid * self.phi_grid * energy_pred).sum(axis=sum_axis)/(energy_sum_pred+1e-16)
        eta_width_pred = torch.sqrt((eta_square_pred - eta_EC_pred**2).clip(min=1e-16))
        phi_width_pred = torch.sqrt((phi_square_pred - phi_EC_pred**2).clip(min=1e-16))

        ec_loss = 0.5*(torch.square(eta_EC_true-eta_EC_pred) + torch.square(phi_EC_true-phi_EC_pred)).mean()
        width_loss = 0.5*(torch.square(eta_width_true-eta_width_pred) + torch.square(phi_width_true-phi_width_pred)).mean()

        if self.perceptual_weight:
            R = pred.get('R_pred', batch['R_true'])
            # we force the R broadcasting to pred_U
            reco_pix_E_pred = (pred['pixels_U_pred'] * R) * batch['E_inc']
            if self.unpad_fn:
                reco_pix_E_pred=self.unpad_fn(reco_pix_E_pred)
            # still need NRZA
            p_loss = self.perceptual_loss(batch['pixels_E_orig'].contiguous(), reco_pix_E_pred.contiguous()) # vector along N
            p_loss = p_loss.reshape_as(rec_loss)
            p_loss=p_loss.mean()
        else:
            p_loss = torch.tensor(0, device=rec_loss.device)

        # average over batch
        rec_loss = rec_loss.mean() # scalar
        nll_loss = self.pixel_weight * rec_loss + self.perceptual_weight * p_loss 

        if self.ec_weight > 0:
            nll_loss = nll_loss + self.ec_weight * ec_loss
        if self.width_weight > 0:
            nll_loss = nll_loss + self.width_weight * width_loss

        # if self.learn_R:
        #     R_true = batch['R_true']
        #     R_pred = pred['R_pred']
        #     R2_loss = torch.square(R_true - R_pred).mean() # scalar
        #     R1_loss = torch.abs(R_true - R_pred).mean() # scalar
        #     R_lt0p7_pred =  (R_pred < 0.7).float().mean()  # scalar (mean over batch)
        #     R_lt0p7_true =  (R_true < 0.7).float().mean() 
        #     R_gt0p8_pred =  (R_pred > 0.8).float().mean() 
        #     R_gt0p8_true =  (R_true > 0.8).float().mean() 
        #     nll_loss = nll_loss + self.R2_weight * R2_loss + self.R1_weight * R1_loss # scalar
        
        # E_cell metrics
        N_100kev=torch.logical_and(energy_pred < 0.1, energy_pred > 0.).float().mean() - torch.logical_and(energy_true < 0.1, energy_true > 0.).float().mean()
        N_exact0=(energy_pred == 0. ).float().mean() - (energy_true == 0. ).float().mean()

        # now the GAN part
        # DISC should always aee the original results
    
        disc_pix_true = batch[f'pixels_{self.disc_normalization}']
        disc_pix_pred = pred[f'pixels_{self.disc_normalization}_pred']

        if optimizer_idx == 0:
            # generator update
            if self.discriminator_weight > 0:
                #if cond is None:
                #    assert not self.disc_conditional
                #    logits_fake = self.discriminator(reconstructions.contiguous())
                #else:
                #    assert self.disc_conditional
                #    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                logits_fake = self.discriminator(disc_pix_pred.contiguous(), batch['cond'])

                g_loss = -torch.mean(logits_fake)
            
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0, device=disc_pix_true.device)

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                disc_factor=torch.tensor(disc_factor, device=disc_pix_true.device)

            else:
                g_loss = disc_factor = d_weight = torch.tensor(0.0, device=disc_pix_true.device)

            #codebook_loss = codebook_loss.mean() # scalar
            assert len(codebook_loss.shape) <= 1
            loss = nll_loss + \
                   d_weight * disc_factor * g_loss + \
                   self.codebook_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.detach().clone(),
                   "{}/quant_loss".format(split): codebook_loss.detach().clone(),
                   "{}/nll_loss".format(split): nll_loss.detach().clone(),
                   "{}/l1_loss".format(split): l1_loss.detach().clone(),
                   "{}/l2_loss".format(split): l2_loss.detach().clone(),
                   "{}/ec_loss".format(split): ec_loss.detach().clone(),
                   "{}/width_loss".format(split): width_loss.detach().clone(),
                   "{}/rec_loss".format(split): rec_loss.detach().clone(),
                   "{}/p_loss".format(split): p_loss.detach().clone(),
                   "{}/d_weight".format(split): d_weight.detach().clone(),
                   "{}/disc_factor".format(split): disc_factor.detach().clone(),
                   "{}/g_loss".format(split): g_loss.detach().clone(),
            }
            # if self.learn_R:
            #     log.update({
            #        "{}/R1_loss".format(split): R1_loss.detach().clone(),
            #        "{}/R2_loss".format(split): R2_loss.detach().clone(),
            #        "{}/R_pred".format(split): R_pred.detach().clone().mean(),
            #        "{}/R_true".format(split): R_true.detach().clone().mean(),
            #        "{}/R_lt0p7_pred".format(split): R_lt0p7_pred.detach().clone(),
            #        "{}/R_lt0p7_true".format(split): R_lt0p7_true.detach().clone(),
            #        "{}/R_gt0p8_pred".format(split): R_gt0p8_pred.detach().clone(),
            #        "{}/R_gt0p8_true".format(split): R_gt0p8_true.detach().clone(),
            #        })

            with torch.no_grad():
                perplexity, cluster_usage = measure_perplexity(pred['indices'], self.n_embed)

                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/perplexity_normed"] = perplexity / self.n_embed
                log[f"{split}/cluster_usage"] = cluster_usage

                log[f"{split}/L1_eta_EC_zavg"] = torch.abs(eta_EC_pred - eta_EC_true).mean() 
                log[f"{split}/L1_phi_EC_zavg"] = torch.abs(phi_EC_pred - phi_EC_true).mean() 
                log[f"{split}/L1_eta_WD_zavg"] = torch.abs(eta_width_pred - eta_width_true).mean() 
                log[f"{split}/L1_phi_WD_zavg"] = torch.abs(phi_width_pred - phi_width_true).mean()
                
                log[f"{split}/L1_eta_EC_zmax"] = torch.abs(eta_EC_pred - eta_EC_true).max(axis=-1).values.mean() 
                log[f"{split}/L1_phi_EC_zmax"] = torch.abs(phi_EC_pred - phi_EC_true).max(axis=-1).values.mean()  
                log[f"{split}/L1_eta_WD_zmax"] = torch.abs(eta_width_pred - eta_width_true).max(axis=-1).values.mean()  
                log[f"{split}/L1_phi_WD_zmax"] = torch.abs(phi_width_pred - phi_width_true).max(axis=-1).values.mean()  

                log[f"{split}/N_100kev"] = N_100kev.detach().clone() 
                log[f"{split}/N_exact0"] = N_exact0.detach().clone() 

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            #if cond is None:
            #    logits_real = self.discriminator(inputs.contiguous().detach())
            #    logits_fake = self.discriminator(reconstructions.contiguous().detach())
            #else:
            #    logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
            #    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            logits_real = self.discriminator(disc_pix_true.contiguous().detach(), batch['cond'])
            logits_fake = self.discriminator(disc_pix_pred.contiguous().detach(), batch['cond'])

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start) # how about start D opt in first step??
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

# class CombinedLossRes(pl.LightningModule):
#     def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
#                  disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
#                  use_actnorm=False, disc_conditional=False,
#                  disc_ndf=64, disc_loss="hinge", n_embed=None,
#                  pixel_loss="l1", lp_weight=1, R1_weight=0, R2_weight=0,
#                  disc_config=None,
#                  perceptual_config=None,
#                  perceptual_weight=1.0,
#                  cond_dim=0,
#                  factor1=1.,
#                  factor2=3000.,
#                  factor3=7.,
#                  ): # , prepro="log"
#         super().__init__()
#         assert disc_loss in ["hinge", "vanilla"]
#         # assert pixel_loss in ["l1", "l2"]

#         self.codebook_weight = codebook_weight
#         self.pixel_weight = pixelloss_weight
#         self.R1_weight=R1_weight
#         self.R2_weight=R2_weight

#         if pixel_loss == "l1":
#             self.pixel_loss = l1
#         elif pixel_loss == "l2":
#             self.pixel_loss = l2
#         elif pixel_loss == "sl1":
#             self.pixel_loss = torch.nn.SmoothL1Loss(reduction="none")
#         elif pixel_loss == "huber":
#             self.pixel_loss = torch.nn.HuberLoss(reduction="none")
#         else:
#             assert False

#         self.lp_weight = lp_weight

#         self.discriminator = instantiate_from_config(disc_config, {'cond_dim': cond_dim, "factor1": factor1, "factor2": factor2, "factor3": factor3})
#         self.discriminator_iter_start = disc_start

#         if disc_loss == "hinge":
#             self.disc_loss = hinge_d_loss
#         elif disc_loss == "vanilla":
#             self.disc_loss = vanilla_d_loss
#         else:
#             raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
#         print(f"CALORECOWithDiscriminator running with {disc_loss} loss.")
#         self.disc_factor = disc_factor
#         self.discriminator_weight = disc_weight
#         self.disc_conditional = disc_conditional
#         self.n_embed = n_embed

#         self.perceptual_weight = perceptual_weight
#         if self.perceptual_weight and perceptual_config is not None:
#             self.perceptual_loss = instantiate_from_config(perceptual_config)

#     def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
#         if last_layer is not None:
#             nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
#         else:
#             nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

#         d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#         d_weight = d_weight * self.discriminator_weight
#         return d_weight

#     def forward(self, codebook_loss, inputs, reconstructions, _inputs, _reconstructions, R, optimizer_idx,
#             global_step, last_layer=None, cond=None, split="train", predicted_indices=None, seperate_R=False,
#             ):
#         # if not exists(codebook_loss):
#         #     codebook_loss = torch.tensor([0.]).to(inputs.device)
#         #rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
#         if not seperate_R:
#             assert R is None
        
#         R_true = inputs.sum(axis=(-1,-2,-3))[...,None,None,None]
#         if seperate_R:
#             inputs=inputs/R_true

#         rec_loss = self.pixel_loss(
#                 inputs.double().contiguous(),
#                 reconstructions.double().contiguous())
#         # sum over all pixels
#         rec_loss = torch.sum(rec_loss, axis=(-1,-2,-3)).float()

#         if self.perceptual_weight:
#             # p_loss ~ (batch_size, )
#             p_loss = self.perceptual_loss(_inputs.contiguous(), _reconstructions.contiguous())
#         else:
#             p_loss = torch.tensor([0.0], device=inputs.device)

#         # average over batch
#         nll_loss = self.lp_weight * rec_loss + self.perceptual_weight * p_loss 
#         nll_loss = torch.mean(nll_loss)

#         ''' todo:
#         x_norm = _inputs
#         rec_norm = _reconstructions
#         R_loss = l2((x_norm.sum(axis=(-1,-2,-3)) - rec_norm.sum(axis=(-1,-2,-3))) / y_cond)
#         '''
#         R_pred = R if seperate_R else reconstructions.sum(axis=(-1,-2,-3))[...,None,None,None]
#         R2_loss = torch.square(R_true - R_pred).mean()
#         R1_loss = torch.abs(R_true - R_pred).mean()

#         nll_loss = nll_loss + self.R2_weight * R2_loss + self.R1_weight * R1_loss

#         # now the GAN part
#         # DISC should always aee the original results
#         if seperate_R:
#             inputs=inputs*R_true
#             reconstructions=reconstructions*R_pred

#         if optimizer_idx == 0:
#             # generator update
#             if self.discriminator_weight > 0:
#                 #if cond is None:
#                 #    assert not self.disc_conditional
#                 #    logits_fake = self.discriminator(reconstructions.contiguous())
#                 #else:
#                 #    assert self.disc_conditional
#                 #    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
#                 logits_fake = self.discriminator((reconstructions-inputs.detach()).contiguous(), cond)

#                 g_loss = -torch.mean(logits_fake)
            
#                 try:
#                     d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
#                 except RuntimeError:
#                     assert not self.training
#                     d_weight = torch.tensor(0.0, device=inputs.device)

#                 disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
#                 disc_factor=torch.tensor(disc_factor, device=inputs.device)

#             else:
#                 g_loss = disc_factor = d_weight = torch.tensor(0.0, device=inputs.device)

#             loss = nll_loss + \
#                    d_weight * disc_factor * g_loss + \
#                    self.codebook_weight * codebook_loss.mean()

#             log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
#                    "{}/quant_loss".format(split): codebook_loss.detach().mean(),
#                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
#                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
#                    "{}/p_loss".format(split): p_loss.detach().mean(),
#                    "{}/d_weight".format(split): d_weight.detach(),
#                    "{}/disc_factor".format(split): disc_factor.detach(),
#                    "{}/g_loss".format(split): g_loss.detach().mean(),
#                    "{}/R1_loss".format(split): R1_loss.detach().mean(),
#                    "{}/R2_loss".format(split): R2_loss.detach().mean(),
#                    "{}/R_pred".format(split): R_pred.detach().mean(),
#                    "{}/R_true".format(split): R_true.detach().mean(),
#                    }
#             if predicted_indices is not None:
#                 assert self.n_embed is not None
#                 with torch.no_grad():
#                     perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_embed)
#                 log[f"{split}/perplexity"] = perplexity
#                 log[f"{split}/perplexity_normed"] = perplexity / self.n_embed
#                 log[f"{split}/cluster_usage"] = cluster_usage
#             return loss, log

#         if optimizer_idx == 1:
#             # second pass for discriminator update
#             #if cond is None:
#             #    logits_real = self.discriminator(inputs.contiguous().detach())
#             #    logits_fake = self.discriminator(reconstructions.contiguous().detach())
#             #else:
#             #    logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
#             #    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
#             logits_real = self.discriminator(torch.zeros_like(inputs), cond)
#             logits_fake = self.discriminator((reconstructions-inputs.detach()).contiguous().detach(), cond)

#             disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start) # how about start D opt in first step??
#             d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

#             log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
#                    "{}/logits_real".format(split): logits_real.detach().mean(),
#                    "{}/logits_fake".format(split): logits_fake.detach().mean()
#                    }
#             return d_loss, log
