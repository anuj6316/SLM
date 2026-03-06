from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        ('API Keys', {'fields': ('jina_api_key', 'groq_api_key')}),
    )

admin.site.register(CustomUser, CustomUserAdmin)