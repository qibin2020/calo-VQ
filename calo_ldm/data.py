import torch
from torch.utils.data import (
        Dataset, DataLoader,
        RandomSampler, SequentialSampler, BatchSampler
        )
import pytorch_lightning as pl

import os
import psutil
from functools import partial
import h5py

from calo_ldm.util import instantiate_from_config

class CaloHDF5(Dataset):
    def __init__(self, file_path, do_rotations=False,
            do_inversions=False, do_ds1_augmentations=False,do_noise_kev=0.,
            load_partial=None):
        super().__init__()

        print(f"Load from {file_path}",os.path.exists(file_path))
        if do_rotations:
            print("RANDOM ROTATION ENABLED")
        if do_inversions:
            print("RANDOM INVERSION ENABLED")
        if do_ds1_augmentations:
            print("DATASET 1 AUGMENTATIONS ENABLED")

        self.file_path = file_path
        self.do_rotations = do_rotations
        self.do_inversions = do_inversions
        self.do_ds1_augmentations = do_ds1_augmentations

        self.do_noise_kev=do_noise_kev

        if load_partial:
            print("WARNING: load partial dataset (only for debug!)")
        self.dataset_len = self._preload(load_partial)

    @staticmethod
    def ds1_augmentations(data):
        if data.shape[-1] == 368:
            # photons dataset
            A = 10 # angular bins
            l0, l1, l2, l3, l4 = torch.split(data, [8,160,190,5,5], dim=-1)
            l1 = l1.unflatten(-1, (A,-1))
            l2 = l2.unflatten(-1, (A,-1))

            rot = torch.rand(data.shape[0])>0.5
            l1_rot = torch.roll(l1, A//2, dims=-2)
            l2_rot = torch.roll(l2, A//2, dims=-2)
            l1 = torch.where(rot[:,None,None], l1_rot, l1)
            l2 = torch.where(rot[:,None,None], l2_rot, l2)

            flip = torch.rand(data.shape[0])>0.5
            l1_flip = torch.flip(l1, dims=(-2,))
            l2_flip = torch.flip(l2, dims=(-2,))
            l1 = torch.where(flip[:,None,None], l1_flip, l1)
            l2 = torch.where(flip[:,None,None], l2_flip, l2)

            l1 = l1.flatten(start_dim=-2)
            l2 = l2.flatten(start_dim=-2)
            data = torch.cat([l0, l1, l2, l3, l4], axis=-1)

            return data
        elif data.shape[-1] == 533:
            A = 10 # angular bins
            l0, l1, l2, l3, l4, l5, l6 = torch.split(data, [8,100,100,5,150,160,10], dim=-1)
            l1 = l1.unflatten(-1, (A, -1))
            l2 = l2.unflatten(-1, (A, -1))
            l4 = l4.unflatten(-1, (A, -1))
            l5 = l5.unflatten(-1, (A, -1))

            rot = torch.rand(data.shape[0])>0.5
            l1_rot = torch.roll(l1, A//2, dims=-2)
            l2_rot = torch.roll(l2, A//2, dims=-2)
            l4_rot = torch.roll(l4, A//2, dims=-2)
            l5_rot = torch.roll(l5, A//2, dims=-2)
            l1 = torch.where(rot[:,None,None], l1_rot, l1)
            l2 = torch.where(rot[:,None,None], l2_rot, l2)
            l4 = torch.where(rot[:,None,None], l4_rot, l4)
            l5 = torch.where(rot[:,None,None], l5_rot, l5)

            flip = torch.rand(data.shape[0])>0.5
            l1_flip = torch.flip(l1, dims=(-2,))
            l2_flip = torch.flip(l2, dims=(-2,))
            l4_flip = torch.flip(l4, dims=(-2,))
            l5_flip = torch.flip(l5, dims=(-2,))
            l1 = torch.where(flip[:,None,None], l1_flip, l1)
            l2 = torch.where(flip[:,None,None], l2_flip, l2)
            l4 = torch.where(flip[:,None,None], l4_flip, l4)
            l5 = torch.where(flip[:,None,None], l5_flip, l5)

            l1 = l1.flatten(start_dim=-2)
            l2 = l2.flatten(start_dim=-2)
            l4 = l4.flatten(start_dim=-2)
            l5 = l5.flatten(start_dim=-2)

            data = torch.cat([l0,l1,l2,l3,l4,l5,l6], axis=-1)

            return data


        raise NotImplementedError

    def __getitem__(self, index):
        data = self._data[index]
        e_inc = self._e_inc[index]

        if self.do_rotations:
            nroll = torch.randint(0, data.shape[-1], (1,)).item()
            data = torch.roll(data, nroll, dims=-1)
        if self.do_inversions:
            assert len(index) > 0, "do_rotations only works in batch index mode"
            flip = torch.rand(data.shape[0])>0.5
            flip = flip.reshape((-1,) + (1,)*(len(data.shape)-1))
            flipped = torch.flip(data, dims=(-1,))
            data = torch.where(flip, flipped, data)
        if self.do_ds1_augmentations:
            assert len(index) > 0, "do_ds1_augmentations only works in batch index mode"
            data = self.ds1_augmentations(data)

        ret = {
            'pixels_E': data.contiguous(),
            'E_inc': e_inc,
        }
        if self._cond is not None:
            ret['E_inc_binned'] = self._cond[index,:]

        return ret

    def __len__(self):
        return self.dataset_len
    
    def _preload(self,partial=None):
        pid = os.getpid()
        process = psutil.Process()
        children = psutil.Process(pid).children(recursive=True)
        total_memory = process.memory_info().rss
        for child in children:
            total_memory += child.memory_info().rss
        print(f"Before preload: CPU mem {total_memory/1024/1024/1024:.1f} GB")
        with h5py.File(self.file_path, 'r') as h5_file:
            data = torch.from_numpy(h5_file['showers'][:partial]).float()
            self._e_inc = torch.from_numpy(h5_file['incident_energies'][:partial]).float()
            self._cond = None
            if data.shape[-1] == 45*16*9: # Z A R --> R Z A
                # ds 2
                data = data.reshape(-1,45,16,9)
                data = data.permute(0,3,1,2)
            elif data.shape[-1] == 45*50*18:
                # ds 3
                data = data.reshape(-1,45,50,18)
                data = data.permute(0,3,1,2)
                ## avoid all zero event...
                #data[data.sum(axis=(-1,-2,-3))==0] = 1e-9
            elif data.shape[-1] in (368,533,):
                # ds1
                cond = (torch.log2(self._e_inc)-8).long() # integers 0-14: hardcoded!
                self._cond = cond
            else:
                raise NotImplementedError

            # self._data = data.contiguous()
            self._data = data
            # adding noise
            if self.do_noise_kev>0:
                self._data += torch.rand(self._data.shape) * self.do_noise_kev * 1e-3
        # monitor the mem
        total_memory = process.memory_info().rss
        for child in children:
            total_memory += child.memory_info().rss
        print(f"After preload: CPU mem {total_memory/1024/1024/1024:.1f} GB")
        return len(self._e_inc)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, pin_memory=False, batched_indices=True):
        super().__init__()
        self.batch_size = batch_size
        self.batched_indices = batched_indices
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if self.num_workers > 1: 
            print("NOTE: multiple dataloader, watch out your memory!!")
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        # if test is not None:
        #     self.dataset_configs["test"] = test
        #     self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        self.wrap = wrap
        self.pin_memory = pin_memory


    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if self.batched_indices:
            sampler = BatchSampler(RandomSampler(self.datasets['train']),
                        batch_size=self.batch_size,
                        drop_last=True) # drop small batch in case we are using batchnorm statistics
            return DataLoader(self.datasets['train'], batch_size=None,
                    num_workers=self.num_workers, pin_memory=self.pin_memory,
                    sampler=sampler, worker_init_fn=init_fn)
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=init_fn, pin_memory=self.pin_memory)

    def _val_dataloader(self, shuffle=False):
        assert not shuffle # we need fixed dataset for metric eval.
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        if self.batched_indices:
            sampler = BatchSampler(SequentialSampler(self.datasets['validation']),
                        batch_size=self.batch_size,
                        drop_last=True)
            return DataLoader(self.datasets['validation'], batch_size=None,
                    num_workers=self.num_workers, pin_memory=self.pin_memory,
                    sampler=sampler, worker_init_fn=init_fn)

        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, pin_memory=self.pin_memory)
