from PIL import Image
import numpy as np
import torch
from django.shortcuts import redirect, render
from Models_app.forms import ImageUploadForm
from Models_app.models import augmentation, UNET


def main_page(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            with open('staticfiles/input/uploaded_image.png', 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)
            return redirect('watching_photos')
    else:
        form = ImageUploadForm()
    return render(request, 'mainpage.html', {'form': form})


def watching_photos(request):
    image_path = 'staticfiles/input/uploaded_image.png'
    image = Image.open(image_path).convert("RGB")
    return render(request, 'watching.html', context={'images': image})


def predict(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET()
    model.load_state_dict(torch.load('unet_tumor_08', map_location=device))
    model.eval()
    model.to(device)
    _, transform_val = augmentation()
    image_path = '/input/uploaded_image.png'
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