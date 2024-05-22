from django.contrib import admin
from django.urls import path
from Models_app.views import main_page, watching_photos, predict, results


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', main_page),
    path('watch/', watching_photos, name='watching_photos'),
    path('predict/', predict, name='predict'),
    path('results/', results, name='results')
]
