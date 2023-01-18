from django.urls import path
from nlp_app import views

urlpatterns = [
    path('', views.home, name="home"),
    path('home/', views.home, name="home"),
    path('analyze/', views.getPrediction, name="analyze"),
]
