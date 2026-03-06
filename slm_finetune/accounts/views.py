from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.contrib.auth import authenticate
from .serializers import UserSerializer
from groq import Groq
import logging
import requests

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([AllowAny])
def signup_view(request):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "user": serializer.data})
    return Response(serializer.errors, status=400)

@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get('username') or request.data.get('email')
    password = request.data.get('password')
    
    if not username or not password:
        return Response({"error": "Please provide both username/email and password"}, status=400)
        
    user = authenticate(username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        serializer = UserSerializer(user)
        return Response({
            "token": token.key,
            "user": serializer.data
        })
    return Response({"error": "Invalid Credentials"}, status=400)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def me_view(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)

@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def update_api_keys(request):
    serializer = UserSerializer(request.user, data=request.data, partial=True)

    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=400)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def groq_health_check(request):
    """
    Validates the Groq API key by attempting a lightweight API call (listing models).
    """
    api_key = request.data.get("groq_api_key") or request.user.groq_api_key
    if not api_key:
        return Response({
            "isActive": False,
            "error": "API key is missing.",
        }, status=400)
    
    try:
        # Initialize the base Groq client
        client = Groq(api_key=api_key)
        
        # Use models.list() as it's free and doesn't consume tokens/invoke models
        client.models.list()
        
        return Response({
            "isActive": True
        })
    except Exception as e:
        logger.error(f"Groq Health Check failed for user {request.user.email}: {e}")
        return Response({
            "isActive": False,
            "error": str(e),
        }, status=200) # Return 200 so the check itself is considered 'successful' by the frontend

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def jina_health_check(request):
    """
    Validates the Jina API key by attempting to scrape a sample URL.
    """
    api_key = request.data.get("jina_api_key") or request.user.jina_api_key
    if not api_key:
        return Response({
            "isActive": False,
            "error": "API key is missing"
        }, status=400)

    try:
        url = "https://r.jina.ai/https://www.example.com"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Added timeout to prevent the view from hanging
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return Response({
                "isActive": True
            })
        
        # Handle non-200 responses (401, 403, etc.)
        return Response({
            "isActive": False,
            "error": f"Jina returned status {response.status_code}"
        }, status=200)
            
    except Exception as e:
        logger.error(f"Jina Health Check failed for user {request.user.email}: {e}")
        return Response({
            "isActive": False,
            "error": str(e)
        }, status=200)
