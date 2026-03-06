from django.contrib.auth.models import AbstractUser
from django.db import models
from encrypted_model_fields.fields import EncryptedCharField

class CustomUser(AbstractUser):
    ## Abstractuser already has: username, email etc
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True)
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to = "avatars/", null=True, blank=True)
    
    ## api keys 
    jina_api_key = EncryptedCharField(max_length=500, blank=True)
    groq_api_key = EncryptedCharField(max_length=500, blank=True)

    ## The email as login pattern
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
