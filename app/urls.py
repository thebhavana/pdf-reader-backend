from django.urls import path
from . import views

urlpatterns = [
    path('upload-pdf/', views.UploadPDF.as_view(), name='upload-pdf'),
    path('query/', views.QueryView.as_view(), name='query'),
]
