# 📘 The SLM Pipeline: Django Auth Master Reference

> **Quick Summary**: This project uses **Django Rest Framework (DRF)** for the backend and **React + LocalStorage** for the frontend. Security is handled via **PBKDF2 Hashing** and **Token-based Authentication**.

---

## 📑 Table of Contents

1. [🗺️ Developer Implementation Flow](#developer-implementation-flow)
2. [🧠 The Big Picture — What IS a User in Django?](#the-big-picture)
3. [🏛️ Django's Built-in User — The Free Gift](#djangos-built-in-user)
4. [🔐 The Password — Django's Most Important Secret](#the-password)
5. [🎫 The Token System — The Security Badge](#the-token-system)
6. [🔬 Dissecting YOUR Code Line by Line](#dissecting-your-code)
7. [⚙️ The `settings.py` — The Control Panel](#the-settings-control-panel)
8. [🏗️ AbstractUser — The Upgrade Path](#abstractuser-the-upgrade-path)
9. [🛡️ Protecting Views — The Bouncer Pattern](#protecting-views)
10. [🗺️ The Complete Auth Flow — End to End](#the-complete-auth-flow)
11. [🚀 The Best Practice Integration Table](#best-practice-integration-table)
12. [💻 Essential Code Snippets](#essential-code-snippets)
13. [🛠️ Customizing & Extending](#customizing-and-extending)
14. [🚨 Quick Troubleshooting](#quick-troubleshooting)
15. [🔗 Useful References](#useful-references)

---

## 🗺️ Developer Implementation Flow <a name="developer-implementation-flow"></a>

> **Read this first.** Every time you set up auth in a new Django project, follow these steps in order. Skipping steps or doing them out of order is the #1 cause of auth bugs.

```
STEP 1: MODEL          STEP 2: SETTINGS       STEP 3: SERIALIZER
─────────────          ────────────────        ──────────────────
accounts/models.py  →  settings.py          →  accounts/serializers.py
Define CustomUser      Register apps           Define what fields
(AbstractUser)         Set AUTH_USER_MODEL      go in / come out
                       Add REST_FRAMEWORK       Set write_only on password
                       Set CORS origins

        ↓
STEP 4: VIEWS          STEP 5: URLS           STEP 6: MIGRATIONS
─────────────          ────────────            ──────────────────
accounts/views.py   →  accounts/urls.py    →  python manage.py
Write signup_view       Wire views to paths    makemigrations
Write login_view        Include in             python manage.py
Write logout_view       project urls.py        migrate

        ↓
STEP 7: TEST (Postman / curl)
─────────────────────────────
POST /account/signup/   →  Get token back
POST /account/login/    →  Get token back
GET  /account/me/       →  Send token in header, get user back
POST /account/logout/   →  Token deleted from DB
```

### The File-by-File Dependency Map

```
settings.py
    ├── INSTALLED_APPS        ← must have authtoken, corsheaders, accounts
    ├── AUTH_USER_MODEL       ← must point to accounts.CustomUser
    ├── REST_FRAMEWORK        ← sets default auth + permission classes
    └── CORS_ALLOWED_ORIGINS  ← must list your React dev server port

accounts/models.py            ← define CustomUser (do this BEFORE first migrate)
    └── depends on: settings.AUTH_USER_MODEL

accounts/serializers.py       ← validate + create users
    └── depends on: models.CustomUser

accounts/views.py             ← handle requests, issue tokens
    └── depends on: serializers, Token model, authenticate()

accounts/urls.py              ← map paths to views
    └── depends on: views

slm_finetune/urls.py          ← include accounts.urls
    └── depends on: accounts/urls.py
```

---

## 🧠 The Big Picture — What IS a User in Django? <a name="the-big-picture"></a>

Before any code, build the mental model. When someone visits your app, Django asks one fundamental question every single time:

```
"Who are you, and what are you allowed to do?"
```

This splits into two concepts that developers constantly confuse:

```
Authentication  →  "Who are you?"      (Proving identity: Login)
Authorization   →  "What can you do?"  (Checking permissions: Access Control)
```

Your current code **only handles Authentication**. Authorization comes later (with permissions, roles, etc). Don't mix them up.

---

## 🏛️ Django's Built-in User — The Free Gift <a name="djangos-built-in-user"></a>

Django ships with a complete `User` model out of the box. Look at what you're already using in your `serializers.py`:

```python
from django.contrib.auth.models import User
```

This single import gives you a database table with these columns **already created for free:**

```
id            → Auto-incrementing primary key (1, 2, 3...)
username      → Must be unique. "john_doe"
email         → Not unique by default (a common gotcha!)
password      → NEVER stored as plain text. More on this below.
first_name    → Optional
last_name     → Optional
is_active     → Boolean. False = "soft deleted" / banned user
is_staff      → Can this user log into /admin/?
is_superuser  → God mode. Bypasses ALL permission checks.
date_joined   → Auto-set timestamp
last_login    → Auto-updated on every login
```

> **🔴 Production Insight:** The default `User` uses `username` as the login identifier. Most modern apps want `email` as the login. This is why the guide mentions `AbstractUser` — see [Chapter: AbstractUser](#abstractuser-the-upgrade-path).

---

## 🔐 The Password — Django's Most Important Secret <a name="the-password"></a>

This is the #1 thing juniors get wrong.

**What actually gets stored in your database:**

```
# You receive from user:
password = "mypassword123"

# What Django stores:
"pbkdf2_sha256$720000$randomSalt$hashedOutput="
```

It's **4 parts** separated by `$`:

```
Algorithm  → pbkdf2_sha256  (the hashing function used)
Iterations → 720000         (how many times it's scrambled)
Salt       → randomSalt     (a random string to prevent rainbow table attacks)
Hash       → hashedOutput=  (the final scrambled result)
```

**Why does this matter in production?**

```python
# ❌ WRONG - Never do this (plain text comparison)
if user.password == "mypassword123":
    ...

# ✅ CORRECT - Django handles the hashing internally
from django.contrib.auth import authenticate
user = authenticate(username="john", password="mypassword123")
```

`authenticate()` internally calls `check_password()` which re-hashes the input and compares. You **never** touch the raw password after creation.

**The `create_user` vs `create` difference is critical:**

```python
# ❌ WRONG - Stores "mypassword123" as plain text in DB
User.objects.create(username="john", password="mypassword123")

# ✅ CORRECT - Stores "pbkdf2_sha256$720000$salt$hash" in DB
User.objects.create_user(username="john", password="mypassword123")
```

---

## 🎫 The Token System — The Security Badge <a name="the-token-system"></a>

```
Traditional Web (Sessions)          Your App (Token-based / API)
─────────────────────────           ─────────────────────────────
1. Login → Server creates           1. Login → Server creates
   a session in DB                     a Token in DB
2. Browser gets a Cookie            2. Client gets a Token string
3. Cookie sent automatically        3. Client sends manually in
   on every request                    every request Header
4. Works only in browsers           4. Works anywhere (mobile,
                                       Postman, React, etc.)
```

**What the `Token` table looks like in your DB:**

```
token_key              user_id    created
─────────────────────────────────────────────
9944b09199c62bcf...    1          2026-03-01
a5b2c9d8e7f6a5b4...    2          2026-03-01
```

One token per user. When a user logs in again, `get_or_create` either returns the existing token or makes a new one.

**How the frontend uses it (the Authorization Header pattern):**

```javascript
// Every protected API call must look like this:
fetch('/api/some-protected-endpoint/', {
  headers: {
    'Authorization': `Token 9944b09199c62bcf...`  // ← This exact format
  }
})
```

Django reads this header, looks up the token in the DB, finds the user, and attaches them to `request.user`. This is how you know WHO is making the request in any view.

---

## 🔬 Dissecting YOUR Code Line by Line <a name="dissecting-your-code"></a>

### `serializers.py` — The Validator + Creator

```python
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "password", "email", "first_name", "last_name"]
        extra_kwargs = {"password": {"write_only": True}}
        #               ↑ This is critical. password comes IN but never goes OUT.
        #               Without this, your API would return the hashed password
        #               in responses. Security disaster.

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)
        #                  ↑ NOT .create() — This is the most important line.
        #                  .create() stores plain text password.
        #                  .create_user() hashes it first. Never mix these up.
```

### `views.py` — The Traffic Controller

```python
@api_view(['POST'])
# ↑ This decorator does 3 things:
#   1. Wraps the function as a DRF view
#   2. Enforces only POST requests (returns 405 for GET, PUT, etc.)
#   3. Gives you the DRF Request object (with .data instead of .POST)

def signup_view(request):
    serializer = UserSerializer(data=request.data)
    # ↑ Pass incoming JSON to serializer. Not validated yet.

    if serializer.is_valid():
    # ↑ NOW it validates: Are required fields present?
    #   Is username unique? Is password strong enough?

        user = serializer.save()
        # ↑ Calls our custom .create() method above.

        token, _ = Token.objects.get_or_create(user=user)
        # ↑ The _ means "I don't care about the second return value"
        #   get_or_create returns (object, created_boolean)

        return Response({"token": token.key, "user": serializer.data})
        # ↑ serializer.data is safe because password is write_only

    return Response(serializer.errors, status=400)
    # ↑ Returns exactly what went wrong e.g.:
    #   {"username": ["A user with that username already exists."]}
```

---

## ⚙️ The `settings.py` — The Control Panel <a name="the-settings-control-panel"></a>

```python
INSTALLED_APPS = [
    'django.contrib.auth',        # ← The entire auth system lives here
    'rest_framework.authtoken',   # ← Creates the Token table in your DB
    'corsheaders',                # ← Allows your React app to talk to Django
    'accounts',                   # ← Your custom app
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # ← MUST be first. Handles CORS headers
    # ...
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    # ↑ Runs on EVERY request. Reads the Authorization header,
    #   looks up the token, sets request.user automatically.
    #   If no valid token: request.user = AnonymousUser (not None!)
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",   # ← Create React App port
    "http://localhost:5173"    # ← Vite port
]
# Without this, browsers BLOCK all requests from React to Django.
# This is browser enforcement, not Django.

# ✅ Add this block — you're currently missing it:
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
        # Makes ALL views require auth by default.
        # Override specific public views with AllowAny.
    ]
}
```

---

## 🏗️ AbstractUser — The Upgrade Path <a name="abstractuser-the-upgrade-path"></a>

Here's the fork in the road every Django developer faces:

```
django.contrib.auth.models.User          →  Good for quick prototypes
                                             Hard to extend later

django.contrib.auth.models.AbstractUser  →  The production standard
                                             Extend BEFORE first migration
                                             Add any fields you want
```

> **Golden Rule:** If there's ANY chance you'll add custom fields, use AbstractUser from Day 1. Changing the User model after migrations exist is extremely painful.

```python
# accounts/models.py — Production Pattern
from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # AbstractUser already has: username, email, password,
    # first_name, last_name, is_active, is_staff, is_superuser

    # You just ADD what you need:
    phone = models.CharField(max_length=15, blank=True)
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)

    # The email-as-login pattern (very common in production):
    USERNAME_FIELD = 'email'        # ← "Login with email instead of username"
    REQUIRED_FIELDS = ['username']  # ← Still required for createsuperuser

# settings.py — Tell Django to use YOUR model, not the default
AUTH_USER_MODEL = 'accounts.CustomUser'
```

**Migration checklist for this upgrade:**

```bash
# 1. Update models.py with AbstractUser
# 2. Update serializers.py to import CustomUser
# 3. Add AUTH_USER_MODEL = 'accounts.CustomUser' to settings.py
# 4. Delete old migration files in accounts/migrations/ (keep __init__.py)
# 5. Run fresh migrations:
python manage.py makemigrations accounts
python manage.py migrate
```

---

## 🛡️ Protecting Views — The Bouncer Pattern <a name="protecting-views"></a>

Right now all your views are public. In production, most views need to be protected. Here are the three patterns:

```python
# Pattern 1: Per-view decorator (most common for mixed-auth apps)
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_my_profile(request):
    # request.user is guaranteed to be a real user here
    # If no valid token → automatic 401 Unauthorized response
    return Response({"email": request.user.email})


# Pattern 2: Global default (set in settings.py — shown above)
# All views require auth. Override with AllowAny for public endpoints.


# Pattern 3: AllowAny for public endpoints (login, signup)
from rest_framework.permissions import AllowAny

@api_view(['POST'])
@permission_classes([AllowAny])  # ← Explicitly public
def login_view(request):
    ...
```

**The logout view you're currently missing:**

```python
# accounts/views.py
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    request.user.auth_token.delete()
    return Response({"message": "Logged out successfully"})

# accounts/urls.py — add this:
path('logout/', views.logout_view, name='logout'),
```

---

## 🗺️ The Complete Auth Flow — End to End <a name="the-complete-auth-flow"></a>

```
SIGNUP FLOW:
─────────────────────────────────────────────────────────────
React                          Django
──────                         ──────
POST /account/signup/    →     1. UserSerializer validates data
{                              2. create_user() hashes password
  username: "john",            3. User row created in DB
  password: "pass123",         4. Token row created in DB
  email: "j@j.com"             5. Returns token + user data
}
                         ←     {"token": "9944b09...", "user": {...}}

React: localStorage.setItem('token', '9944b09...')


LOGIN FLOW:
─────────────────────────────────────────────────────────────
React                          Django
──────                         ──────
POST /account/login/     →     1. authenticate() checks credentials
{                              2. Looks up user by username
  username: "john",            3. Hashes input password
  password: "pass123"          4. Compares with stored hash
}                              5. get_or_create token
                         ←     {"token": "9944b09..."}

React: localStorage.setItem('token', '9944b09...')


PROTECTED REQUEST FLOW:
─────────────────────────────────────────────────────────────
React                          Django
──────                         ──────
GET /api/scrape-jobs/    →     1. AuthenticationMiddleware reads header
Headers: {                     2. Looks up token in DB → finds user
  Authorization:               3. Sets request.user = john
  "Token 9944b09..."           4. Permission check: IsAuthenticated? ✓
}                              5. View executes with request.user available
                         ←     {scrape jobs data...}


LOGOUT FLOW:
─────────────────────────────────────────────────────────────
React                          Django
──────                         ──────
POST /account/logout/    →     Token.objects.filter(user=request.user).delete()
Headers: {                     ↑ Deletes the token from DB
  Authorization:               ↑ The old token string is now useless
  "Token 9944b09..."
}                         ←    {"message": "Logged out"}

React: localStorage.removeItem('token')
```

---

## 🚀 The Best Practice Integration Table <a name="best-practice-integration-table"></a>

Follow this flow whenever you are setting up or fixing the Auth system.

| Step | File | Action | Best Practice |
| :--- | :--- | :--- | :--- |
| **1. Model** | `accounts/models.py` | Define `CustomUser` with `AbstractUser` | Do this **before** first migration — changing later is painful |
| **2. Settings** | `settings.py` | Register apps, set `AUTH_USER_MODEL`, `REST_FRAMEWORK`, `CORS` | `CorsMiddleware` must be **first** in middleware list |
| **3. Serializer** | `accounts/serializers.py` | Define fields, set `write_only` on password | Never return the `password` field in a GET response |
| **4. Views** | `accounts/views.py` | Write signup, login, logout views | Use `create_user()` not `create()`. Always use `authenticate()` |
| **5. URLs** | `accounts/urls.py` + `slm_finetune/urls.py` | Wire views to paths, include in project | Django expects trailing slashes: `/account/login/` |
| **6. Migrate** | Terminal | `makemigrations` then `migrate` | Check for `UndefinedTable` errors — always migrate after model changes |
| **7. Frontend** | `App.tsx` | Store token in `localStorage`, check on startup | Always check for the token inside `useEffect` |

---

## 💻 Essential Code Snippets <a name="essential-code-snippets"></a>

### A. The Backend Bridge (`views.py`)

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

### B. The Logout View (Missing from your code — add this)

```python
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    request.user.auth_token.delete()
    return Response({"message": "Logged out successfully"})
```

### C. A Protected "Get My Profile" View

```python
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def me_view(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)
```

### D. The Frontend Keeper (`App.tsx`)

```tsx
useEffect(() => {
  const token = localStorage.getItem('token');
  if (token) {
    setIsAuthenticated(true);
    // Best Practice: Load initial dashboard data here
  }
}, []);
```

### E. The Frontend Request Helper

```typescript
// utils/api.ts — a reusable helper for all protected requests
const authFetch = (url: string, options: RequestInit = {}) => {
  const token = localStorage.getItem('token');
  return fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Token ${token}`,
      ...options.headers,
    },
  });
};

// Usage in any component:
const response = await authFetch('/api/scrape-jobs/');
```

---

## 🛠️ Customizing & Extending <a name="customizing-and-extending"></a>

### How to Add Custom Fields (Phone, Bio, etc.)

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
        model = CustomUser  # Switch from User to CustomUser
        fields = ['id', 'email', 'phone']  # MUST add field name here
```

### The Three Functions You Will Use in Every Django Project

| Function | Where Used | What it Does |
|---|---|---|
| `create_user()` | Signup view | Creates user with **hashed** password |
| `authenticate()` | Login view | Verifies credentials, returns User or None |
| `Token.objects.get_or_create()` | Login + Signup | Gets or creates the auth token |

---

## 🚨 Quick Troubleshooting <a name="quick-troubleshooting"></a>

- **"UndefinedTable" Error**: You forgot to migrate.
    - `python manage.py migrate`
- **"CORS Error" in Browser**: Check `settings.py`.
    - Is `CorsMiddleware` at the very top of `MIDDLEWARE`?
- **"Invalid Token"**: The token in `localStorage` might be old.
    - Clear browser cache or use `localStorage.removeItem('token')`.
- **"404 Not Found"**: Check your trailing slashes.
    - Django usually expects `/account/login/` (with a slash).
- **Password stored as plain text**: You used `.create()` instead of `.create_user()`.
    - Delete the user, fix the view, re-register.
- **`request.user` is `AnonymousUser`**: You forgot the `Authorization` header in the frontend request.
    - Check your fetch calls — `Authorization: Token <key>` must be present.
- **`AUTH_USER_MODEL` error after switching to CustomUser**: You ran migrations before adding `AUTH_USER_MODEL` to settings.
    - Delete migrations, add setting, re-run `makemigrations` + `migrate`.

---

## 🔗 Useful References <a name="useful-references"></a>

- [Django Auth Docs](https://docs.djangoproject.com/en/5.1/topics/auth/)
- [DRF Token Guide](https://www.django-rest-framework.org/api-guide/authentication/#tokenauthentication)
- [Django Password Management](https://docs.djangoproject.com/en/5.1/topics/auth/passwords/)
- [AbstractUser Docs](https://docs.djangoproject.com/en/5.1/topics/auth/customizing/#substituting-a-custom-user-model)
- [DRF Permissions](https://www.django-rest-framework.org/api-guide/permissions/)