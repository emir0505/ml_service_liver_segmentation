import torch
from django.shortcuts import render, redirect
from matplotlib import pyplot as plt

from Models_app.models import augmentation, LiverDataset


def main_page():
    # Здесь тупо что-то вроде описания работы, а также переход на загрузку картинок, download_dataset
    pass


def download_dataset():
    # Здесь, просто грузим на сервер выбранные файлы, с редиректом на страничку со чтением, также надо бы сделать полосу загрузки
    pass


def watching_photos(df_train):
    # получаем все фото после чтения из модели
    # и выводим
    # в шаблоне этого урла, надо кнопку с выбором модели, а также запуск обучения
    # return render('watching.html', context=...)
    pass


def reading_dataset():
    # Считываем датасет, также надо какой-то traceback, что-то по типу как было в коллабе или в кегле
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mean = -644.2137072615333
    std = 673.311976351113
    tr_tr, tr_val = augmentation(mean, std)
    cnt = 3
    df_train = LiverDataset(tr_tr, cnt)
    df_test = LiverDataset(tr_val, cnt)
    return redirect('watching_photos')  # надо как-то передать туда df_train


def train():
    # тренируем выбранную модель. Затем редирект на страницу результатов
    pass


def results():
    # показываем графики метрик, индекс жакара, можно также картинки(ориг изображение, метку и предикт)
    pass