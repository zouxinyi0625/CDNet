# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

import copy
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys

sys.path.append("../")

identity = lambda x: x


class SimpleDataset:
    def __init__(self, args, transform, target_transform=identity, split=None):
        self.transform = transform
        self.target_transform = target_transform

        data_path = args.benchmarks_dir + args.testset + '/images'
        if args.split == "val":
            data_path += "/test/"
        elif args.split == "all":
            data_path += "/all/"
        print("loading unlabeled data from {}".format(data_path))

        self.d = ImageFolder(data_path, transform=transform, target_transform=target_transform)

    def __getitem__(self, i):
        return self.d[i]

    def __len__(self):
        return len(self.d)


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop' or transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Scale' or transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, split=None):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.split = split

    def get_data_loader(self, aug, num_workers=12):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform, split=self.split)

        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


if __name__ == '__main__':
    pass
