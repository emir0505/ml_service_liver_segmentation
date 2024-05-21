from django import forms
from .models import UploadedImage


class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Загрузить изображение')
