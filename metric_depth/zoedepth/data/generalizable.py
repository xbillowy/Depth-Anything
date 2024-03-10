import os
import cv2
import copy
import torch
import numpy as np
from PIL import Image
from glob import glob
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

from zoedepth.data.easyvolcap import EasyVolcap

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict


class Generalizable(Dataset):
    def __init__(self,
                 config
                 ):

        self.dataset = config.dataset
        # Get the paths to the data, modify them in the config file `metric_depth/zoedepth/utils/config.py`
        self.data_roots = config.data_roots  # this will overwrite the dataset configs
        self.meta_roots = config.meta_roots  # this will overwrite thet data_roots configs,
        self.excd_roots = config.excd_roots  # this will exclude the data_roots

        # Prepare for dataset config
        if self.meta_roots and self.data_roots:
            log(yellow(f'data_roots entries will be replace by meta_roots: {self.meta_roots}'))

        if self.meta_roots:
            data_roots = []
            # Load dataset configs here
            # Will treat every subdirectory containing a images folder as a dataset folder
            for meta_root in self.meta_roots:
                meta_data_roots = sorted(glob(join(meta_root, '*')))
                for data_root in meta_data_roots:
                    if os.path.isdir(data_root) and data_root.split('/')[-1] not in self.excd_roots:
                        if os.path.exists(join(data_root, 'images')):
                            data_roots.append(data_root)

        if data_roots:
            log(yellow(f'dataset_cfgs entries will be replace by data_roots: {data_roots}'))

        if config.split == 'test':
            data_roots = data_roots[::5]

        if data_roots:
            # Load dataset configs here
            dataset_cfgs = []
            for data_root in data_roots:
                dataset_cfg = copy.deepcopy(config)
                dataset_cfg.data_root = data_root
                dataset_cfgs.append(dataset_cfg)

        # Reuse these reusable contents
        self.datasets: List[EasyVolcap] = [EasyVolcap(cfg) for cfg in dataset_cfgs]
        self.lengths = torch.as_tensor([len(d) for d in self.datasets])
        self.accum_lengths = self.lengths.cumsum(dim=-1)

    def extract_dataset_index(self, index: Union[dotdict, int]):
        # Maybe think of a better way to update input of __getitem__
        if isinstance(index, dotdict): sampler_index, n_srcs = index.index, index.n_srcs
        else: sampler_index = index

        # Dataset index will indicate the sample to use
        dataset_index = torch.searchsorted(self.accum_lengths, sampler_index, right=True)  # 2 will not be inserted as 2
        sampler_index = sampler_index - (self.accum_lengths[dataset_index - 1] if dataset_index > 0 else 0)  # maybe -1
        # MARK: This is nasty, pytorch inconsistency of conversion
        sampler_index = sampler_index.item() if isinstance(sampler_index, torch.Tensor) else sampler_index  # convert to int

        if isinstance(index, dotdict): index.index = sampler_index
        else: index = sampler_index

        return dataset_index, index

    @property
    def split(self):
        return self.datasets[0].split

    @property
    def n_views(self):
        return 1

    @property
    def n_latents(self):
        return sum(self.lengths)  # for samplers

    def __getitem__(self, index: Union[dotdict, int]):
        dataset_index, index = self.extract_dataset_index(index)
        dataset = self.datasets[dataset_index]  # get the dataset to sample from
        return dataset.__getitem__(index)

    def __len__(self):
        return self.n_views * self.n_latents  # there's no notion of epoch here


def get_generalizable_loader(config, batch_size=1, mode='train', **kwargs):
    # FIXME: find a better way to handle EasyVolcap test dataset
    if mode == 'online_eval' and 'test' not in config.dataset:
        config.split = 'test'
        config.view_sample = [0, 1, 1]

    # Build the dataloader
    dataloader = DataLoader(Generalizable(config), batch_size=batch_size, **kwargs)

    return dataloader
