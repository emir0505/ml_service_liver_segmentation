from PIL import Image
import numpy as np
import torch
from PIL.Image import Image
from django.shortcuts import redirect, render
from Models_app.models import augmentation, UNET


def main_page(request):
    return render(request, 'mainpage.html')


def download_image(request):
    return render(request, '')


def watching_photos(request):
    image_path = '' # куда загрузили
    image = Image.open(image_path).convert("RGB")
    return render(request, 'watching.html', context={'images': image})


def predict(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET()
    model.load_state_dict(torch.load('unet_tumor_08', map_location=device))
    model.eval()
    model.to(device)
    _, transform_val = augmentation()
    image_path = 'your_image.jpg'
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