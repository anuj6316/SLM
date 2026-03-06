from django.contrib.auth import get_user_model
from rest_framework import serializers

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'jina_api_key', 'groq_api_key')
        extra_kwargs = {
            'jina_api_key': {'required': False, 'allow_blank': True},
            'groq_api_key': {'required': False, 'allow_blank': True},
        }

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data['email'],
            username=validated_data.get('username', validated_data['email']),
            password=validated_data['password'],
            jina_api_key=validated_data.get('jina_api_key', ''),
            groq_api_key=validated_data.get('groq_api_key', '')
        )
        return user

