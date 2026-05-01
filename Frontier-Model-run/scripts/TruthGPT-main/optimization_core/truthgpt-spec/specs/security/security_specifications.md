# TruthGPT Security Specifications

## Overview

This document outlines the comprehensive security specifications for TruthGPT, covering authentication, authorization, data protection, network security, and compliance requirements.

## Authentication & Authorization

### JWT Authentication

```python
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"

@dataclass
class TokenPayload:
    """JWT token payload structure."""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    jti: str  # JWT ID

class TruthGPTSecurity:
    """TruthGPT security manager."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist = set()
        self.rate_limits = {}
    
    def generate_token(self, user_data: Dict[str, Any], 
                      token_type: TokenType = TokenType.ACCESS,
                      expires_in: int = 3600) -> str:
        """Generate JWT token."""
        now = datetime.utcnow()
        jti = secrets.token_urlsafe(32)
        
        payload = {
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'email': user_data['email'],
            'roles': user_data.get('roles', []),
            'permissions': user_data.get('permissions', []),
            'token_type': token_type.value,
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'jti': jti,
            'iss': 'truthgpt.ai',
            'aud': 'truthgpt-api'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Verify JWT token."""
        try:
            # Check if token is blacklisted
            if token in self.token_blacklist:
                return None
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Create token payload object
            token_payload = TokenPayload(
                user_id=payload['user_id'],
                username=payload['username'],
                email=payload['email'],
                roles=payload['roles'],
                permissions=payload['permissions'],
                token_type=TokenType(payload['token_type']),
                issued_at=datetime.fromtimestamp(payload['iat']),
                expires_at=datetime.fromtimestamp(payload['exp']),
                jti=payload['jti']
            )
            
            return token_payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def blacklist_token(self, token: str):
        """Add token to blacklist."""
        self.token_blacklist.add(token)
    
    def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """Check rate limit for user and endpoint."""
        key = f"{user_id}:{endpoint}"
        now = datetime.utcnow()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old entries
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if now - timestamp < timedelta(minutes=1)
        ]
        
        # Check if under limit
        if len(self.rate_limits[key]) >= 100:  # 100 requests per minute
            return False
        
        # Add current request
        self.rate_limits[key].append(now)
        return True
```

### OAuth 2.0 Integration

```python
from authlib.integrations.flask_client import OAuth
from authlib.integrations.requests_client import OAuth2Session
import requests

class OAuth2Provider:
    """OAuth 2.0 provider integration."""
    
    def __init__(self, client_id: str, client_secret: str, 
                 redirect_uri: str, provider: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.provider = provider
        self.session = OAuth2Session(client_id, client_secret)
    
    def get_authorization_url(self, state: str) -> str:
        """Get authorization URL."""
        if self.provider == "google":
            return self._get_google_auth_url(state)
        elif self.provider == "github":
            return self._get_github_auth_url(state)
        elif self.provider == "microsoft":
            return self._get_microsoft_auth_url(state)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_google_auth_url(self, state: str) -> str:
        """Get Google OAuth authorization URL."""
        auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid email profile',
            'response_type': 'code',
            'state': state,
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{auth_url}?{query_string}"
    
    def _get_github_auth_url(self, state: str) -> str:
        """Get GitHub OAuth authorization URL."""
        auth_url = "https://github.com/login/oauth/authorize"
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'user:email',
            'state': state
        }
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{auth_url}?{query_string}"
    
    def _get_microsoft_auth_url(self, state: str) -> str:
        """Get Microsoft OAuth authorization URL."""
        auth_url = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid email profile',
            'response_type': 'code',
            'state': state,
            'response_mode': 'query'
        }
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{auth_url}?{query_string}"
    
    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        if self.provider == "google":
            return self._exchange_google_code(code)
        elif self.provider == "github":
            return self._exchange_github_code(code)
        elif self.provider == "microsoft":
            return self._exchange_microsoft_code(code)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _exchange_google_code(self, code: str) -> Dict[str, Any]:
        """Exchange Google authorization code."""
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()
    
    def _exchange_github_code(self, code: str) -> Dict[str, Any]:
        """Exchange GitHub authorization code."""
        token_url = "https://github.com/login/oauth/access_token"
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()
    
    def _exchange_microsoft_code(self, code: str) -> Dict[str, Any]:
        """Exchange Microsoft authorization code."""
        token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from provider."""
        if self.provider == "google":
            return self._get_google_user_info(access_token)
        elif self.provider == "github":
            return self._get_github_user_info(access_token)
        elif self.provider == "microsoft":
            return self._get_microsoft_user_info(access_token)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_google_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get Google user information."""
        user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {'Authorization': f'Bearer {access_token}'}
        
        response = requests.get(user_info_url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def _get_github_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get GitHub user information."""
        user_info_url = "https://api.github.com/user"
        headers = {'Authorization': f'token {access_token}'}
        
        response = requests.get(user_info_url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def _get_microsoft_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get Microsoft user information."""
        user_info_url = "https://graph.microsoft.com/v1.0/me"
        headers = {'Authorization': f'Bearer {access_token}'}
        
        response = requests.get(user_info_url, headers=headers)
        response.raise_for_status()
        return response.json()
```

### Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import Set, List, Dict, Any
from dataclasses import dataclass

class Permission(Enum):
    """System permissions."""
    # Model permissions
    MODELS_READ = "models:read"
    MODELS_WRITE = "models:write"
    MODELS_DELETE = "models:delete"
    
    # Optimization permissions
    OPTIMIZATIONS_READ = "optimizations:read"
    OPTIMIZATIONS_WRITE = "optimizations:write"
    OPTIMIZATIONS_DELETE = "optimizations:delete"
    
    # Inference permissions
    INFERENCE_READ = "inference:read"
    INFERENCE_WRITE = "inference:write"
    
    # Admin permissions
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    ADMIN_DELETE = "admin:delete"
    
    # User management
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"

class Role(Enum):
    """System roles."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"

@dataclass
class User:
    """User entity."""
    user_id: str
    username: str
    email: str
    roles: Set[Role]
    permissions: Set[Permission]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None

class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self.role_permissions = {
            Role.ADMIN: {
                Permission.MODELS_READ,
                Permission.MODELS_WRITE,
                Permission.MODELS_DELETE,
                Permission.OPTIMIZATIONS_READ,
                Permission.OPTIMIZATIONS_WRITE,
                Permission.OPTIMIZATIONS_DELETE,
                Permission.INFERENCE_READ,
                Permission.INFERENCE_WRITE,
                Permission.ADMIN_READ,
                Permission.ADMIN_WRITE,
                Permission.ADMIN_DELETE,
                Permission.USERS_READ,
                Permission.USERS_WRITE,
                Permission.USERS_DELETE
            },
            Role.MODERATOR: {
                Permission.MODELS_READ,
                Permission.MODELS_WRITE,
                Permission.OPTIMIZATIONS_READ,
                Permission.OPTIMIZATIONS_WRITE,
                Permission.INFERENCE_READ,
                Permission.INFERENCE_WRITE,
                Permission.USERS_READ
            },
            Role.USER: {
                Permission.MODELS_READ,
                Permission.OPTIMIZATIONS_READ,
                Permission.INFERENCE_READ,
                Permission.INFERENCE_WRITE
            },
            Role.GUEST: {
                Permission.MODELS_READ,
                Permission.INFERENCE_READ
            }
        }
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for a user."""
        permissions = set()
        
        for role in user.roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
        
        # Add explicit permissions
        permissions.update(user.permissions)
        
        return permissions
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        if not user.is_active:
            return False
        
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def has_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        if not user.is_active:
            return False
        
        user_permissions = self.get_user_permissions(user)
        return any(permission in user_permissions for permission in permissions)
    
    def has_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        if not user.is_active:
            return False
        
        user_permissions = self.get_user_permissions(user)
        return all(permission in user_permissions for permission in permissions)
    
    def can_access_resource(self, user: User, resource_type: str, 
                           action: str) -> bool:
        """Check if user can access a specific resource."""
        permission_name = f"{resource_type}:{action}"
        
        try:
            permission = Permission(permission_name)
            return self.has_permission(user, permission)
        except ValueError:
            return False
    
    def get_accessible_resources(self, user: User, resource_type: str) -> List[str]:
        """Get list of accessible resources for a user."""
        if not user.is_active:
            return []
        
        user_permissions = self.get_user_permissions(user)
        accessible_resources = []
        
        for permission in user_permissions:
            if permission.value.startswith(f"{resource_type}:"):
                action = permission.value.split(":", 1)[1]
                accessible_resources.append(action)
        
        return accessible_resources
```

## Data Protection

### Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import os
import secrets

class DataEncryption:
    """Data encryption for TruthGPT."""
    
    def __init__(self, master_key: str = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using Fernet."""
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_file(self, input_path: str, output_path: str):
        """Encrypt file."""
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt_data(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
    
    def decrypt_file(self, input_path: str, output_path: str):
        """Decrypt file."""
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        data = self.decrypt_data(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(data)
    
    def encrypt_string(self, text: str) -> str:
        """Encrypt string and return base64 encoded result."""
        data = text.encode('utf-8')
        encrypted_data = self.encrypt_data(data)
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt base64 encoded string."""
        encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
        data = self.decrypt_data(encrypted_data)
        return data.decode('utf-8')

class AESEncryption:
    """AES encryption for sensitive data."""
    
    def __init__(self, key: bytes = None):
        if key:
            self.key = key
        else:
            self.key = secrets.token_bytes(32)  # 256-bit key
    
    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes]:
        """Encrypt data using AES-256-GCM."""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return ciphertext, iv + encryptor.tag
    
    def decrypt(self, ciphertext: bytes, iv_and_tag: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        iv = iv_and_tag[:12]
        tag = iv_and_tag[12:]
        
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext

class PasswordHashing:
    """Password hashing utilities."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using PBKDF2."""
        salt = secrets.token_bytes(32)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.b64encode(kdf.derive(password.encode()))
        return base64.b64encode(salt + key).decode()
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            decoded = base64.b64decode(hashed_password.encode())
            salt = decoded[:32]
            stored_key = decoded[32:]
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.b64encode(kdf.derive(password.encode()))
            
            return key == stored_key
        except Exception:
            return False
```

### Data Anonymization

```python
import hashlib
import random
import string
from typing import Dict, Any, List
import re

class DataAnonymizer:
    """Data anonymization for TruthGPT."""
    
    def __init__(self):
        self.anonymization_rules = {
            'email': self._anonymize_email,
            'phone': self._anonymize_phone,
            'ssn': self._anonymize_ssn,
            'credit_card': self._anonymize_credit_card,
            'ip_address': self._anonymize_ip_address,
            'name': self._anonymize_name,
            'address': self._anonymize_address
        }
    
    def anonymize_data(self, data: Dict[str, Any], 
                      fields_to_anonymize: List[str]) -> Dict[str, Any]:
        """Anonymize specified fields in data."""
        anonymized_data = data.copy()
        
        for field in fields_to_anonymize:
            if field in anonymized_data:
                anonymized_data[field] = self._anonymize_field(
                    field, anonymized_data[field]
                )
        
        return anonymized_data
    
    def _anonymize_field(self, field_type: str, value: Any) -> Any:
        """Anonymize a specific field."""
        if field_type in self.anonymization_rules:
            return self.anonymization_rules[field_type](value)
        else:
            return self._anonymize_generic(value)
    
    def _anonymize_email(self, email: str) -> str:
        """Anonymize email address."""
        if '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        anonymized_local = local[0] + '*' * (len(local) - 1)
        return f"{anonymized_local}@{domain}"
    
    def _anonymize_phone(self, phone: str) -> str:
        """Anonymize phone number."""
        # Remove non-digits
        digits = re.sub(r'\D', '', phone)
        
        if len(digits) >= 10:
            return f"***-***-{digits[-4:]}"
        else:
            return "***-***-****"
    
    def _anonymize_ssn(self, ssn: str) -> str:
        """Anonymize Social Security Number."""
        # Remove non-digits
        digits = re.sub(r'\D', '', ssn)
        
        if len(digits) == 9:
            return f"***-**-{digits[-4:]}"
        else:
            return "***-**-****"
    
    def _anonymize_credit_card(self, card: str) -> str:
        """Anonymize credit card number."""
        # Remove non-digits
        digits = re.sub(r'\D', '', card)
        
        if len(digits) >= 4:
            return f"****-****-****-{digits[-4:]}"
        else:
            return "****-****-****-****"
    
    def _anonymize_ip_address(self, ip: str) -> str:
        """Anonymize IP address."""
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
        else:
            return "xxx.xxx.xxx.xxx"
    
    def _anonymize_name(self, name: str) -> str:
        """Anonymize name."""
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}*** {parts[-1][0]}***"
        else:
            return f"{name[0]}***"
    
    def _anonymize_address(self, address: str) -> str:
        """Anonymize address."""
        # Simple anonymization - replace with generic address
        return "123 Main St, Anytown, ST 12345"
    
    def _anonymize_generic(self, value: Any) -> str:
        """Generic anonymization."""
        if isinstance(value, str):
            if len(value) <= 3:
                return "*" * len(value)
            else:
                return value[0] + "*" * (len(value) - 2) + value[-1]
        else:
            return "***"

class DataMasking:
    """Data masking for sensitive information."""
    
    def __init__(self):
        self.masking_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
    
    def mask_text(self, text: str, mask_char: str = '*') -> str:
        """Mask sensitive information in text."""
        masked_text = text
        
        for pattern_name, pattern in self.masking_patterns.items():
            masked_text = re.sub(pattern, mask_char * 10, masked_text)
        
        return masked_text
    
    def mask_specific_pattern(self, text: str, pattern: str, 
                             mask_char: str = '*') -> str:
        """Mask specific pattern in text."""
        return re.sub(pattern, mask_char * 10, text)
```

## Network Security

### Firewall Rules

```yaml
# firewall-rules.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: truthgpt-firewall
  namespace: truthgpt
spec:
  podSelector:
    matchLabels:
      app: truthgpt
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Allow HTTP traffic from ingress controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8001
  
  # Allow gRPC traffic from internal services
  - from:
    - podSelector:
        matchLabels:
          app: truthgpt-grpc
    ports:
    - protocol: TCP
      port: 50051
  
  # Allow health checks
  - from:
    - podSelector:
        matchLabels:
          app: truthgpt
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8001
  
  egress:
  # Allow database connections
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  
  # Allow Redis connections
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  
  # Allow DNS resolution
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  
  # Allow HTTPS traffic for external APIs
  - to: []
    ports:
    - protocol: TCP
      port: 443
```

### SSL/TLS Configuration

```yaml
# tls-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: truthgpt-tls
  namespace: truthgpt
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t...
  tls.key: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t...
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: truthgpt-ingress
  namespace: truthgpt
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
    nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384"
    nginx.ingress.kubernetes.io/ssl-prefer-server-ciphers: "true"
    nginx.ingress.kubernetes.io/ssl-session-cache: "shared:SSL:10m"
    nginx.ingress.kubernetes.io/ssl-session-timeout: "10m"
spec:
  tls:
  - hosts:
    - api.truthgpt.ai
    - grpc.truthgpt.ai
    secretName: truthgpt-tls
  rules:
  - host: api.truthgpt.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: truthgpt-api-service
            port:
              number: 8000
  - host: grpc.truthgpt.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: truthgpt-grpc-service
            port:
              number: 50051
```

### VPN Configuration

```yaml
# vpn-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vpn-config
  namespace: truthgpt
data:
  vpn.conf: |
    client
    dev tun
    proto udp
    remote vpn.truthgpt.ai 1194
    resolv-retry infinite
    nobind
    persist-key
    persist-tun
    ca ca.crt
    cert client.crt
    key client.key
    cipher AES-256-GCM
    auth SHA256
    tls-auth ta.key 1
    comp-lzo
    verb 3
---
apiVersion: v1
kind: Secret
metadata:
  name: vpn-credentials
  namespace: truthgpt
type: Opaque
data:
  ca.crt: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t...
  client.crt: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t...
  client.key: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t...
  ta.key: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t...
```

## Compliance

### GDPR Compliance

```python
class GDPRCompliance:
    """GDPR compliance utilities."""
    
    def __init__(self):
        self.data_retention_period = 365  # days
        self.consent_required = True
    
    def process_data_request(self, user_id: str, request_type: str) -> Dict[str, Any]:
        """Process GDPR data request."""
        if request_type == "access":
            return self._process_access_request(user_id)
        elif request_type == "portability":
            return self._process_portability_request(user_id)
        elif request_type == "erasure":
            return self._process_erasure_request(user_id)
        elif request_type == "rectification":
            return self._process_rectification_request(user_id)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def _process_access_request(self, user_id: str) -> Dict[str, Any]:
        """Process data access request."""
        # Collect all user data
        user_data = {
            'personal_info': self._get_personal_info(user_id),
            'usage_data': self._get_usage_data(user_id),
            'model_data': self._get_model_data(user_id),
            'optimization_data': self._get_optimization_data(user_id)
        }
        
        return {
            'status': 'completed',
            'data': user_data,
            'requested_at': datetime.now().isoformat(),
            'processed_at': datetime.now().isoformat()
        }
    
    def _process_portability_request(self, user_id: str) -> Dict[str, Any]:
        """Process data portability request."""
        # Export user data in portable format
        export_data = {
            'user_id': user_id,
            'exported_at': datetime.now().isoformat(),
            'data': self._get_all_user_data(user_id),
            'format': 'JSON',
            'version': '1.0'
        }
        
        return {
            'status': 'completed',
            'export_data': export_data,
            'download_url': self._generate_download_url(export_data)
        }
    
    def _process_erasure_request(self, user_id: str) -> Dict[str, Any]:
        """Process data erasure request."""
        # Delete all user data
        deleted_items = {
            'personal_info': self._delete_personal_info(user_id),
            'usage_data': self._delete_usage_data(user_id),
            'model_data': self._delete_model_data(user_id),
            'optimization_data': self._delete_optimization_data(user_id)
        }
        
        return {
            'status': 'completed',
            'deleted_items': deleted_items,
            'erased_at': datetime.now().isoformat()
        }
    
    def _process_rectification_request(self, user_id: str) -> Dict[str, Any]:
        """Process data rectification request."""
        # Update user data
        return {
            'status': 'completed',
            'updated_at': datetime.now().isoformat()
        }
    
    def check_consent(self, user_id: str) -> bool:
        """Check if user has given consent."""
        # Implementation to check consent
        pass
    
    def record_consent(self, user_id: str, consent_data: Dict[str, Any]):
        """Record user consent."""
        # Implementation to record consent
        pass
    
    def get_data_retention_status(self, user_id: str) -> Dict[str, Any]:
        """Get data retention status for user."""
        return {
            'user_id': user_id,
            'retention_period': self.data_retention_period,
            'data_created_at': self._get_data_created_at(user_id),
            'expires_at': self._get_data_expires_at(user_id),
            'auto_delete': True
        }

class DataAudit:
    """Data audit utilities."""
    
    def __init__(self):
        self.audit_log = []
    
    def log_data_access(self, user_id: str, resource: str, action: str):
        """Log data access."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        }
        
        self.audit_log.append(audit_entry)
        self._store_audit_entry(audit_entry)
    
    def log_data_modification(self, user_id: str, resource: str, 
                            old_value: Any, new_value: Any):
        """Log data modification."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': 'modify',
            'old_value': old_value,
            'new_value': new_value,
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        }
        
        self.audit_log.append(audit_entry)
        self._store_audit_entry(audit_entry)
    
    def get_audit_trail(self, user_id: str = None, 
                       start_date: datetime = None, 
                       end_date: datetime = None) -> List[Dict[str, Any]]:
        """Get audit trail."""
        filtered_log = self.audit_log
        
        if user_id:
            filtered_log = [entry for entry in filtered_log 
                           if entry['user_id'] == user_id]
        
        if start_date:
            filtered_log = [entry for entry in filtered_log 
                           if datetime.fromisoformat(entry['timestamp']) >= start_date]
        
        if end_date:
            filtered_log = [entry for entry in filtered_log 
                           if datetime.fromisoformat(entry['timestamp']) <= end_date]
        
        return filtered_log
```

### HIPAA Compliance

```python
class HIPAACompliance:
    """HIPAA compliance utilities."""
    
    def __init__(self):
        self.phi_fields = [
            'patient_id', 'name', 'date_of_birth', 'ssn', 
            'medical_record_number', 'diagnosis', 'treatment'
        ]
        self.encryption_required = True
        self.audit_required = True
    
    def identify_phi(self, data: Dict[str, Any]) -> List[str]:
        """Identify PHI fields in data."""
        phi_fields_found = []
        
        for field in self.phi_fields:
            if field in data:
                phi_fields_found.append(field)
        
        return phi_fields_found
    
    def encrypt_phi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt PHI fields."""
        encrypted_data = data.copy()
        
        for field in self.phi_fields:
            if field in encrypted_data:
                encrypted_data[field] = self._encrypt_field(encrypted_data[field])
        
        return encrypted_data
    
    def de_identify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """De-identify PHI data."""
        de_identified_data = data.copy()
        
        for field in self.phi_fields:
            if field in de_identified_data:
                de_identified_data[field] = self._de_identify_field(field, de_identified_data[field])
        
        return de_identified_data
    
    def _encrypt_field(self, value: Any) -> str:
        """Encrypt individual field."""
        # Implementation for field encryption
        pass
    
    def _de_identify_field(self, field_name: str, value: Any) -> str:
        """De-identify individual field."""
        # Implementation for field de-identification
        pass
```

## Security Monitoring

### Intrusion Detection

```python
class IntrusionDetection:
    """Intrusion detection system."""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'SELECT.*FROM.*WHERE.*1=1',
            r'UNION.*SELECT',
            r'DROP.*TABLE',
            r'INSERT.*INTO',
            r'UPDATE.*SET',
            r'DELETE.*FROM',
            r'<script.*>',
            r'javascript:',
            r'eval\(',
            r'exec\('
        ]
        self.rate_limits = {
            'login_attempts': 5,
            'api_requests': 100,
            'file_uploads': 10
        }
        self.blocked_ips = set()
        self.suspicious_activities = []
    
    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request for suspicious activity."""
        analysis_result = {
            'is_suspicious': False,
            'threat_level': 'low',
            'threats_detected': [],
            'recommended_action': 'allow'
        }
        
        # Check for SQL injection
        if self._check_sql_injection(request_data):
            analysis_result['is_suspicious'] = True
            analysis_result['threat_level'] = 'high'
            analysis_result['threats_detected'].append('sql_injection')
            analysis_result['recommended_action'] = 'block'
        
        # Check for XSS
        if self._check_xss(request_data):
            analysis_result['is_suspicious'] = True
            analysis_result['threat_level'] = 'high'
            analysis_result['threats_detected'].append('xss')
            analysis_result['recommended_action'] = 'block'
        
        # Check for rate limiting
        if self._check_rate_limit(request_data):
            analysis_result['is_suspicious'] = True
            analysis_result['threat_level'] = 'medium'
            analysis_result['threats_detected'].append('rate_limit_exceeded')
            analysis_result['recommended_action'] = 'throttle'
        
        # Check for suspicious patterns
        if self._check_suspicious_patterns(request_data):
            analysis_result['is_suspicious'] = True
            analysis_result['threat_level'] = 'medium'
            analysis_result['threats_detected'].append('suspicious_pattern')
            analysis_result['recommended_action'] = 'monitor'
        
        return analysis_result
    
    def _check_sql_injection(self, request_data: Dict[str, Any]) -> bool:
        """Check for SQL injection patterns."""
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in self.suspicious_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return True
        return False
    
    def _check_xss(self, request_data: Dict[str, Any]) -> bool:
        """Check for XSS patterns."""
        xss_patterns = [
            r'<script.*>',
            r'javascript:',
            r'onload=',
            r'onerror=',
            r'onclick='
        ]
        
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in xss_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return True
        return False
    
    def _check_rate_limit(self, request_data: Dict[str, Any]) -> bool:
        """Check rate limiting."""
        client_ip = request_data.get('client_ip')
        endpoint = request_data.get('endpoint')
        
        if not client_ip or not endpoint:
            return False
        
        # Implementation for rate limiting check
        return False
    
    def _check_suspicious_patterns(self, request_data: Dict[str, Any]) -> bool:
        """Check for suspicious patterns."""
        # Implementation for suspicious pattern detection
        return False
    
    def block_ip(self, ip_address: str, reason: str):
        """Block IP address."""
        self.blocked_ips.add(ip_address)
        self._log_security_event('ip_blocked', {
            'ip_address': ip_address,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def unblock_ip(self, ip_address: str):
        """Unblock IP address."""
        self.blocked_ips.discard(ip_address)
        self._log_security_event('ip_unblocked', {
            'ip_address': ip_address,
            'timestamp': datetime.now().isoformat()
        })
    
    def _log_security_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log security event."""
        security_event = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': event_data
        }
        
        self.suspicious_activities.append(security_event)
        # Store in security log
        self._store_security_event(security_event)
    
    def _store_security_event(self, event: Dict[str, Any]):
        """Store security event."""
        # Implementation for storing security events
        pass
```

### Security Alerts

```python
class SecurityAlerts:
    """Security alerting system."""
    
    def __init__(self):
        self.alert_rules = {
            'failed_login_attempts': {
                'threshold': 5,
                'time_window': 300,  # 5 minutes
                'severity': 'high'
            },
            'suspicious_api_usage': {
                'threshold': 100,
                'time_window': 60,  # 1 minute
                'severity': 'medium'
            },
            'data_breach_attempt': {
                'threshold': 1,
                'time_window': 60,
                'severity': 'critical'
            }
        }
        self.active_alerts = []
    
    def check_alerts(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for security alerts."""
        triggered_alerts = []
        
        for rule_name, rule_config in self.alert_rules.items():
            if self._should_trigger_alert(rule_name, event_data):
                alert = self._create_alert(rule_name, event_data)
                triggered_alerts.append(alert)
                self.active_alerts.append(alert)
        
        return triggered_alerts
    
    def _should_trigger_alert(self, rule_name: str, event_data: Dict[str, Any]) -> bool:
        """Check if alert should be triggered."""
        rule_config = self.alert_rules[rule_name]
        
        if rule_name == 'failed_login_attempts':
            return self._check_failed_login_attempts(event_data, rule_config)
        elif rule_name == 'suspicious_api_usage':
            return self._check_suspicious_api_usage(event_data, rule_config)
        elif rule_name == 'data_breach_attempt':
            return self._check_data_breach_attempt(event_data, rule_config)
        
        return False
    
    def _create_alert(self, rule_name: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create security alert."""
        rule_config = self.alert_rules[rule_name]
        
        alert = {
            'alert_id': f"alert_{datetime.now().timestamp()}",
            'rule_name': rule_name,
            'severity': rule_config['severity'],
            'timestamp': datetime.now().isoformat(),
            'event_data': event_data,
            'status': 'active',
            'acknowledged': False
        }
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, user_id: str):
        """Acknowledge security alert."""
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_by'] = user_id
                alert['acknowledged_at'] = datetime.now().isoformat()
                break
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_notes: str):
        """Resolve security alert."""
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_by'] = user_id
                alert['resolved_at'] = datetime.now().isoformat()
                alert['resolution_notes'] = resolution_notes
                break
```

## Future Security Enhancements

### Planned Security Features

1. **Zero Trust Architecture**: Implement zero trust security model
2. **Quantum Cryptography**: Post-quantum cryptographic algorithms
3. **AI-Powered Threat Detection**: Machine learning-based threat detection
4. **Blockchain Security**: Decentralized security mechanisms
5. **Biometric Authentication**: Multi-factor biometric authentication

### Research Security Areas

1. **Homomorphic Encryption**: Compute on encrypted data
2. **Secure Multi-Party Computation**: Privacy-preserving computations
3. **Differential Privacy**: Statistical privacy protection
4. **Federated Learning Security**: Secure distributed learning
5. **Edge Security**: Security for edge computing deployments

---

*This security specification provides a comprehensive framework for securing TruthGPT across all deployment scenarios, ensuring data protection, compliance, and threat mitigation.*




