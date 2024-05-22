from django import forms
from .models import UploadedImage


class UploadForm(forms.Form):
    image_file = forms.FileField(label='Select an image file', required=False)
    nifti_file = forms.FileField(label='Select a NIfTI file', required=False)
