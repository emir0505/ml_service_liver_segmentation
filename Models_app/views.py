import torch
from django.shortcuts import render

from Models_app.models import augmentation, LiverDataset


def main_page():
    pass


def download_dataset():
    pass


def reading_dataset():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mean = -644.2137072615333
    std = 673.311976351113
    tr_tr, tr_val = augmentation(mean, std)
    cnt = 3
    df_train = LiverDataset(tr_tr, cnt)
    df_test = LiverDataset(tr_val, cnt)
    pass


def train():
    pass


def results():
    pass