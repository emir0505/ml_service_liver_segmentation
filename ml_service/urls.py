from django.contrib import admin
from django.urls import path
from Models_app.views import main_page, download_dataset, train, results


urlpatterns = [
    path('admin/', admin.site.urls),
    path('main/', main_page),
    path('download/', download_dataset),
    path('training/', train),
    path('show_res/', results)
]
