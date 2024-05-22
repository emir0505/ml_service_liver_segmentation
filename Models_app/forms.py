from django import forms
from .models import UploadedImage


class UploadForm(forms.Form):
    nifti_file = forms.FileField(label='Select a NIfTI file', required=False)
