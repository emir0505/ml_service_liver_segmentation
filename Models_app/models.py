from django.db import models
from glob import glob
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from torchvision.transforms import v2 as T
import albumentations as A
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from albumentations.pytorch import ToTensorV2


def get_numbers_livers(paths_masks: list) -> set:
    numbers = set()
    for path_i in glob(paths_masks):
        index_split = path_i.rfind('-') + 1
        num = path_i[index_split:-4]
        if num in numbers:
            raise 'there is already such a number'
        numbers.add(num)
    return numbers


def convert_number_to_path(number, flag: bool, part='') -> str:
    """
    flag == True -> path to features
    flag == False -> path to target
    """

    mask_feature = ''
    mask_target = ''

    if flag:
        return mask_feature + str(number) + '.nii'
    return mask_target + str(number) + '.nii'


def get_paths_df(paths_features_masks, paths_targets_masks) -> tuple:
    paths_features = []
    paths_targets = []
    numbers_targets = get_numbers_livers(paths_targets_masks)
    for index, path in enumerate(paths_features_masks):
        for num in get_numbers_livers(path):
            if num in numbers_targets:
                paths_features.append(convert_number_to_path(num, True, index + 1))
                paths_targets.append(convert_number_to_path(num, False))
    return np.array(paths_features), np.array(paths_targets)


class LiverDataset:
    def __init__(self, transformations, cnt_read_images3d=5, train=True):
        paths_features_masks = []
        paths_targets_masks = ''
        self.train = train

        paths_features, paths_targets = get_paths_df(paths_features_masks, paths_targets_masks)

        size = int(len(paths_features) * 0.1)
        mask = np.arange(len(paths_features))
        np.random.seed(0)
        indexs = np.random.choice(mask, size=size, replace=False)
        mask[indexs] = 0
        mask[mask != 0] = 1
        mask = mask.astype(bool)

        if train:
            paths_features = paths_features[mask]
            paths_targets = paths_targets[mask]
        else:
            paths_features = paths_features[~mask]
            paths_targets = paths_targets[~mask]

        self.features, self.targets = self.load_data(paths_features, paths_targets, cnt_read_images3d, transformations)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)

    def load_data(self, paths_features, paths_targets, cnt_3d_imgs=10, transformations=None) -> tuple:
        cnt = 0
        features = []
        targets = []

        mean = -644.2137072615333
        std = 673.311976351113

        for index, (path_feature, path_target) in enumerate(zip(paths_features, paths_targets)):
            if cnt == cnt_3d_imgs:
                break
            print(cnt)
            cnt += 1

            img_3d_feature = nib.load(path_feature).get_fdata().transpose(2, 0, 1)
            img_3d_target = nib.load(path_target).get_fdata().transpose(2, 0, 1)

            for img_feature, img_target in zip(img_3d_feature, img_3d_target):
                if len(np.unique(img_target)) != 3:
                    continue
                if len(img_target[img_target == 2]) <= 250:
                    continue

                img_feature = (img_feature - mean) / std

                if self.train:
                    trans = transformations[0](image=img_feature, mask=img_target)
                else:
                    trans = transformations[1](image=img_feature, mask=img_target)
                img_feature_, img_target_ = trans['image'], trans['mask']

                img_feature_ = torch.tensor(img_feature_).unsqueeze(0).float()
                img_target_ = torch.tensor(img_target_).unsqueeze(0).long()

                features.append(img_feature_)
                targets.append(img_target_)

        return np.array(features), np.array(targets)


class CNA(nn.Module):
    def __init__(self, in_nc, out_nc, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_nc, out_nc, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_nc)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)

        return out


class UnetBlock(nn.Module):
    def __init__(self, in_nc, inner_nc, out_nc, inner_block=None):
        super().__init__()

        self.conv1 = CNA(in_nc, inner_nc, stride=2)
        self.conv2 = CNA(inner_nc, inner_nc)
        self.inner_block = inner_block
        self.conv3 = CNA(inner_nc, inner_nc)
        self.conv_cat = nn.Conv2d(inner_nc + in_nc, out_nc, 3, padding=1)

    def forward(self, x):
        _, _, h, w = x.shape

        inner = self.conv1(x)
        inner = self.conv2(inner)
        # print(inner.shape)
        if self.inner_block is not None:
            inner = self.inner_block(inner)
        inner = self.conv3(inner)

        inner = F.interpolate(inner, size=(h, w), mode='bilinear')
        inner = torch.cat((x, inner), axis=1)
        out = self.conv_cat(inner)

        return out


class UNET(nn.Module):
    def __init__(self, in_nc=1, nc=32, out_nc=3, num_downs=6):
        super().__init__()

        self.cna1 = CNA(in_nc, nc)
        self.cna2 = CNA(nc, nc)

        unet_block = None
        for i in range(num_downs - 3):
            unet_block = UnetBlock(8 * nc, 8 * nc, 8 * nc, unet_block)
        unet_block = UnetBlock(4 * nc, 8 * nc, 4 * nc, unet_block)
        unet_block = UnetBlock(2 * nc, 4 * nc, 2 * nc, unet_block)
        self.unet_block = UnetBlock(nc, 2 * nc, nc, unet_block)

        self.cna3 = CNA(nc, nc)

        self.conv_last = nn.Conv2d(nc, out_nc, 3, padding=1)

    def forward(self, x):
        out = self.cna1(x)
        out = self.cna2(out)
        out = self.unet_block(out)
        out = self.cna3(out)
        out = self.conv_last(out)

        return out


def augmentation(mean, std):
    transform_train = A.Compose([
        A.Normalize(mean, std, max_pixel_value=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Resize(height=256, width=256),
    ])

    transform_val = A.Compose([
        A.Normalize(mean, std, max_pixel_value=1),
        A.Resize(height=256, width=256),
    ])
    return transform_train, transform_val
