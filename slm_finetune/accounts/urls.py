from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('me/', views.me_view, name='me'),
    path('update-keys/', views.update_api_keys, name='update-keys'),
    path('groq-health/', views.groq_health_check, name='groq-health'),
    path('jina-health/', views.jina_health_check, name='jina-health'),
]
