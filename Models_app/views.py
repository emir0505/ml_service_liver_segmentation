from PIL import Image
import numpy as np
import torch
from django.shortcuts import redirect, render
import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib
from Models_app.forms import UploadForm
from Models_app.models import augmentation, UNET
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main_page(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image_file']
            nifti_file = form.cleaned_data['nifti_file']

            if image_file:
                file_path = 'staticfiles/input/uploaded_image.png'
                with open(file_path, 'wb') as f:
                    for chunk in image_file.chunks():
                        f.write(chunk)
                return redirect('watching_photos')
            if nifti_file:
                file_path = 'staticfiles/input/uploaded_image.nii'
                with open(file_path, 'wb') as f:
                    for chunk in nifti_file.chunks():
                        f.write(chunk)
                img_3d = nib.load(file_path).get_fdata().transpose(2, 0, 1)
                mean = -644.2137072615333
                std = 673.311976351113
                trans = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
                for idx, img_slice in enumerate(img_3d):
                    img_feature = np.clip(img_slice, 0, 1)
                    transformed = trans(image=img_feature)
                    img_feature_ = transformed['image'].squeeze(0).float().cpu().numpy()
                    img_feature_ = (255*(img_feature_ - img_feature_.min()) / (img_feature_.max() - img_feature_.min())).astype(np.uint8)
                    output_dir = 'staticfiles/input/'
                    output_image_pil = Image.fromarray(img_feature_)
                    output_image_path = os.path.join(output_dir, f'uploaded_image.png')
                    output_image_pil.save(output_image_path)
                return redirect('watching_photos')
    else:
        form = UploadForm()
    return render(request, 'mainpage.html', {'form': form})


def watching_photos(request):
    image_path = 'input/uploaded_image.png'
    return render(request, 'watching.html', context={'image_path': image_path})


def load_model(device):
    model = UNET()
    model_path = os.path.join(BASE_DIR, 'models', 'unet_tumor_08.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def predict(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    model.eval()
    model.to(device)
    _, transform_val = augmentation()
    image_path = 'staticfiles/input/uploaded_image.png'
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transformed = transform_val(image=image_np)
    image_tensor = transformed['image']
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)

    output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save('staticfiles/outputs/pred.png')
    return redirect('results')


def results(request):
    # показываем графики метрик, индекс жакара, можно также картинки(ориг изображение, метку и предикт)
    pass