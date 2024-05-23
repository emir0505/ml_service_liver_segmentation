from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchmetrics import JaccardIndex
from tqdm import tqdm
from django.shortcuts import redirect, render
import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib
from Models_app.forms import UploadForm
from Models_app.models import UNET
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main_page(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            nifti_file = form.cleaned_data['nifti_file']
            if nifti_file:
                file_path = 'staticfiles/input/uploaded_image.nii'
                with open(file_path, 'wb') as f:
                    for chunk in nifti_file.chunks():
                        f.write(chunk)
                img_3d = nib.load(file_path).get_fdata().transpose(2, 0, 1)
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


def load_data_nib(path_features):
    features = []

    trans_standart = A.Compose([
        A.Resize(height=256, width=256),
    ])

    mean = -644.2137072615333
    std = 673.311976351113

    img_3d_feature = nib.load(path_features).get_fdata().transpose(2, 0, 1)

    for img_feature in img_3d_feature:
        img_feature = (img_feature - mean) / std
        trans = trans_standart(image=img_feature)
        img_feature_ = np.array(trans['image'])

        features.append(img_feature_)

    return torch.tensor(np.array(features)).unsqueeze(1).float()


def predict(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    model.eval()
    image_path = 'staticfiles/input/uploaded_image.nii'
    data = load_data_nib(image_path)
    preds = []
    cnt = 0
    for x in tqdm(data):
        cnt += 1
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).max(dim=0)[1].cpu()
        preds.append(pred)

    cnt = 0
    for x, y in zip(data, preds):
        cnt += 1
        if cnt < 60:
            continue
        x = x.squeeze(0).cpu()
        y = y.squeeze(0).cpu()
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(x)
        ax[1].imshow(y)
        plt.savefig('staticfiles/outputs/pred.png')
    return redirect('results')


def results(request):
    output_image_path = 'outputs/pred.png'
    return render(request, 'result.html', context={'output_image_path': output_image_path})

