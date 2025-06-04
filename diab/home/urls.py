from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_diabetes, name='predict'),
    path('features/', views.feature_importance, name='features'),
    path('health/', views.health_check, name='health'),
]