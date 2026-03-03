# 📘 The SLM Pipeline: Auth Master Reference

> **Quick Summary**: This project uses **Django Rest Framework (DRF)** for the backend and **React + LocalStorage** for the frontend. Security is handled via **PBKDF2 Hashing** and **Token-based Authentication**.

---

### 🚀 The Best Practice Flow (Integration Steps)

Follow this flow whenever you are setting up or fixing the Auth system.

| Step | Action | Best Practice |
| :--- | :--- | :--- |
| **1. Infrastructure** | Add `rest_framework.authtoken` & `corsheaders` | Always put `CorsMiddleware` at the **top** of the list. |
| **2. Logic** | Define `serializers.py` & `views.py` | Never return the `password` field in a GET response. |
| **3. Connectivity** | Set `CORS_ALLOWED_ORIGINS` | Use specific URLs (localhost) instead of `*` for security. |
| **4. Persistence** | Save Token in `localStorage` | Check for the token inside `useEffect` on app startup. |

---

### 💻 Essential Code Snippets

#### **A. The Backend Bridge (`views.py`)**
The logic that handles the "Security Badge" (Token) creation.
```python
@api_view(['POST'])
def login_view(request):
    # Standard: Use email or username fallback
    username = request.data.get('email') or request.data.get('username')
    user = authenticate(username=username, password=request.data['password'])
    
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key})
    return Response({"error": "Invalid Credentials"}, status=400)
```

#### **B. The Frontend Keeper (`App.tsx`)**
How React remembers the user after a refresh.
```tsx
useEffect(() => {
  const token = localStorage.getItem('token');
  if (token) {
    setIsAuthenticated(true);
    // Best Practice: Load initial dashboard data here
  }
}, []);
```

---

### 🛠️ Customizing & Extending

#### **How to add Custom Fields (Phone, Bio, etc.)**

**Step 1: The Model (AbstractUser Approach)**
```python
# accounts/models.py
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    phone = models.CharField(max_length=15, blank=True)
```

**Step 2: The Settings**
```python
# settings.py
AUTH_USER_MODEL = 'accounts.CustomUser'
```

**Step 3: The Serializer (Required to see it in React!)**
```python
# accounts/serializers.py
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User # or CustomUser if defined
        fields = ['id', 'email', 'phone'] # MUST add field name here
```

---

### 🚨 Quick Troubleshooting (Skim This!)

*   **"UndefinedTable" Error**: You forgot to migrate. 
    *   `python manage.py migrate`
*   **"CORS Error" in Browser**: Check `settings.py`. 
    *   Is `CorsMiddleware` at the very top?
*   **"Invalid Token"**: The token in `localStorage` might be old.
    *   Clear browser cache or use `localStorage.removeItem('token')`.
*   **"404 Not Found"**: Check your trailing slashes. 
    *   Django usually expects `/account/login/` (with a slash).

---

### 🔗 Useful References
*   [Django Auth Docs](https://docs.djangoproject.com/en/5.1/topics/auth/)
*   [DRF Token Guide](https://www.django-rest-framework.org/api-guide/authentication/#tokenauthentication)
*   [Django Password Management](https://docs.djangoproject.com/en/5.1/topics/auth/passwords/)
