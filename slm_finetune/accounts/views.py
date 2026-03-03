from rest_framework.decorators import api_view ## Marks fucntions as DRF views, restricts HTTP methods.
from rest_framework.response import Response ## used to return JSON responses from views
from rest_framework.authtoken.models import Token ## used to create or retrieve an auth token for a user
from django.contrib.auth import authenticate ## used to verify a user's credentials during login
from .serializers import UserSerializer ## used to validate and create new users during registration

@api_view(['POST'])
def signup_view(request):
    serializer = UserSerializer(data = request.data)
    if serializer.is_valid():
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "user": serializer.data})
    return Response(serializer.errors, status=400)

@api_view(['POST'])
def login_view(request):
    username = request.data.get('username') or request.data.get('email')
    user = authenticate(username=username, password=request.data['password'])
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key})
    return Response({"error": "Invalid Credentials"}, status=400)