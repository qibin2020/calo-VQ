import torch
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

import os
import psutil
import time
import numpy as np
from PIL import Image

from omegaconf import OmegaConf

from calo_ldm.models import VQModel
from calo_ldm.models import CondGPT

### all 3 callbacks taken directly from stable-diffusion repo

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, no_local=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, batch_frequency_calo=None, metric_freq=10):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        if batch_frequency_calo is None:
            batch_frequency_calo = batch_frequency
        self.batch_freq_calo = batch_frequency_calo
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.logger_log_hists = {
            pl.loggers.TestTubeLogger: self._testtube_hists,
        }
        self.logger_log_dict = {
            pl.loggers.TestTubeLogger: self._testtube_dict,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.no_local= no_local
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

        # place to cache random matrices for converting to rgb
        self.color_mats = {}
        self.metric_freq=metric_freq

        self.train_record_epoch=False # validation image: when any training step be logged(arb. image per step; not the phy metrics), the whole val epoch will be also logged

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split, merge=True):
        for k in images:
            if merge:
                grid = torchvision.utils.make_grid(images[k])
            else:
                grid=images[k]
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            # print(tag,grid.shape)
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _testtube_hists(self, pl_module, hist_dict, batch_idx, split, ref=False):
        for k,v in hist_dict.items():
            if ref and k!="R" and "log" not in k: continue
            if v.size(dim=0) < 2:
                print("Warning! unepxected hist",k,v.shape)
                continue
            if k=="R":
                pl_module.logger.experiment.add_histogram(
                    f"{split}/{'ref_' if ref else ''}{k}", torch.clamp(v,min=0,max=2),
                    global_step=pl_module.global_step, bins=np.linspace(0., 2., 201))
            else:
                pl_module.logger.experiment.add_histogram(
                    f"{split}/{'ref_' if ref else ''}{k}", v,
                    global_step=pl_module.global_step)

    @rank_zero_only
    def _testtube_dict(self, pl_module, scalar_dict, batch_idx, split):
        for k,v in scalar_dict.items():
            pl_module.logger.experiment.add_scalar(
                f"{split}/chi2_{k}", v,
                global_step=pl_module.global_step)
            
    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"): # this will be called each step, be careful
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx,split=split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            do_log_img_step=True
                        
            if isinstance(pl_module, CondGPT) and pl_module.pure_mode: # pure mode will not log anything
                do_log_img_step=False
            
            if isinstance(pl_module, CondGPT) and "val" not in split: # gpt training will not log any image (no sense), but validation will do
                do_log_img_step=False

            logger = type(pl_module.logger)
        
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            if do_log_img_step:
                with torch.no_grad():
                    # images from LatentDiffusion model have the following keys:
                    # ['inputs', 'reconstruction', 'diffusion_row', 'samples', 'samples_x0_quantized', 'samples_inpainting', 'mask', 'samples_outpainting', 'progressive_row']
                    images = pl_module.log_images(batch, split=split,
                            inc_calo=self.check_frequency(check_idx, calo=True, split=split), **self.log_images_kwargs)

                    for k in images:
                        N = min(images[k].shape[0], self.max_images)
                        images[k] = images[k][:N]
                        if isinstance(images[k], torch.Tensor):
                            if images[k].shape[-3] not in (1,3): # check for nonstandard color dim
                                images[k] = self.to_rgb(images[k])
                            images[k] = images[k].detach().cpu()
                            if self.clamp:
                                images[k] = torch.clamp(images[k], -1., 1.)

                    if not self.no_local: 
                        self.log_local(pl_module.logger.save_dir, split, images,
                                        pl_module.global_step, pl_module.current_epoch, batch_idx)

                    logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                    logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train: 
                pl_module.train() # ??

            if "val" not in split: # trig on val log
                self.train_record_epoch=True
            else:
                self.train_record_epoch=False # val log only once
    
    def log_img_metrics(self, pl_module, pl_module_real, split="train"): #only peform at epoch validation
        assert split=="val"
        with torch.no_grad():
            plots_dict, chi2_dict = pl_module_real.getMetricsHists() # only collect

        logger = type(pl_module.logger)
        logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
        logger_log_images(pl_module, plots_dict, pl_module.global_step, split, merge=False)

        logger_log_dict = self.logger_log_dict.get(logger, lambda *args, **kwargs: None)
        logger_log_dict(pl_module, chi2_dict, pl_module.global_step, split)
    
    def log_metrics(self, pl_module, pl_module_real, split="train"): #only peform at epoch validation
        assert split=="val"
        with torch.no_grad():
            metrics_dict,ref_dict = pl_module_real.getMetrics() # only collect

        logger = type(pl_module.logger)
        logger_log_hists = self.logger_log_hists.get(logger, lambda *args, **kwargs: None)
        logger_log_hists(pl_module, metrics_dict, pl_module.global_step, split)
        logger_log_hists(pl_module, ref_dict, pl_module.global_step, split, ref=True)

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        ch = x.shape[-3]
        if not ch in self.color_mats:
            self.color_mats[ch] = torch.randn(3, ch, 1, 1).to(x)
        x = torch.nn.functional.conv2d(x, weight=self.color_mats[ch])
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    def check_frequency(self, check_idx, calo=False, split="???"):
        if "val" in split:
            return self.train_record_epoch
        if calo:
            batch_freq = self.batch_freq_calo
        else:
            batch_freq = self.batch_freq
        if ((check_idx % batch_freq) == 0 and (check_idx != 0 or self.log_first_step)) or ((check_idx in self.log_steps) and (check_idx<batch_freq)):
            # try:
            #     self.log_steps.pop(0)
            # except IndexError as e:
            #     print(e)
            #     pass
            return True
        return False
    

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
    
    def getMem(self):
        pid = os.getpid()
        process = psutil.Process()
        children = psutil.Process(pid).children(recursive=True)
        total_memory = process.memory_info().rss
        for child in children:
            total_memory += child.memory_info().rss
        return total_memory/1024/1024/1024

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.trainer.running_sanity_check:
            return

        if isinstance(pl_module, VQModel):
            pl_module_real=pl_module
        elif isinstance(pl_module, CondGPT):
            pl_module_real=pl_module.vq_model
        else:
            print("Logging: not supported module",pl_module)
            return 

        if pl_module.on_record and pl_module.current_epoch % self.metric_freq == 0:
            premem=self.getMem()
            rank_zero_info(f"--> Evaluate metrics... epoch={pl_module.current_epoch}\n")
            rank_zero_info(f"--> Before: {premem:.1f} GB CPU mem)\n")
            start= time.time()
            if pl_module_real.is_ds23:
                pl_module_real.calMstrics() # cal histograms and prepare HLF
            else:
                pl_module_real.calMstrics_DS1() # cal histograms and prepare HLF
            self.log_metrics(pl_module, pl_module_real, split="val") # hisotgrams
            self.log_img_metrics(pl_module, pl_module_real, split="val") # images and chi2
            pl_module_real.cleanMetrics()
            end=time.time()
            rank_zero_info(f"--> Used {end-start:.1f} seconds\n")
            postmem=self.getMem()
            rank_zero_info(f"--> Used {postmem:.1f} GB CPU mem ({postmem-premem:+.1f} GB)\n")

class CUDACallback(Callback):
    def getMem(self):
        pid = os.getpid()
        process = psutil.Process()
        children = psutil.Process(pid).children(recursive=True)
        total_memory = process.memory_info().rss
        for child in children:
            total_memory += child.memory_info().rss
        return total_memory/1024/1024/1024

    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        rank_zero_info(f"Mem count: {self.getMem():.1f} GB")
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak GPU memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


