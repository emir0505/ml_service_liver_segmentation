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


def conv_plus_conv(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),

        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
    )


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        base_channels = 32

        self.down1 = conv_plus_conv(1, base_channels)
        self.down2 = conv_plus_conv(base_channels, base_channels * 2)
        self.down3 = conv_plus_conv(base_channels * 2, base_channels * 4)
        self.down4 = conv_plus_conv(base_channels * 4, base_channels * 8)

        self.up1 = conv_plus_conv(base_channels * 2, base_channels)
        self.up2 = conv_plus_conv(base_channels * 4, base_channels)
        self.up3 = conv_plus_conv(base_channels * 8, base_channels * 2)
        self.up4 = conv_plus_conv(base_channels * 16, base_channels * 4)

        self.bottleneck = conv_plus_conv(base_channels * 8, base_channels * 8)

        self.out = nn.Conv2d(in_channels=base_channels, out_channels=3, kernel_size=1)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual1 = self.down1(x)
        x = self.downsample(residual1)

        residual2 = self.down2(x)
        x = self.downsample(residual2)

        residual3 = self.down3(x)
        x = self.downsample(residual3)

        residual4 = self.down4(x)
        x = self.downsample(residual4)

        x = self.bottleneck(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual4), dim=1)
        x = self.up4(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual3), dim=1)
        x = self.up3(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual2), dim=1)
        x = self.up2(x)

        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, residual1), dim=1)
        x = self.up1(x)

        return self.out(x)


def augmentation():
    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.Resize(height=256, width=256),
        ToTensorV2(),
    ])

    transform_val = A.Compose([
        A.Resize(height=256, width=256),
        ToTensorV2(),
    ])
    return transform_train, transform_val

# нужна модель для сохранения фотографий из этого
# cnt = 0
# for x,y in df_train:
#     cnt += 1
#     if cnt < 50:
#         continue
#     plt.imshow(x.squeeze(0), cmap='Greys')
#     plt.show()
#     plt.imshow(y.squeeze(0), cmap='Greys')
# сохраняем в статикфайлс, в папку reading_images, во вьюхе вызову все объекты модели, и отдам в контекст
# разумеется, чтобы бд не была перегружена, нужно при каждом заходе их удалять


