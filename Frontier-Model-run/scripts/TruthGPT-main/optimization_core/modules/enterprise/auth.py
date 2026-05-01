"""
Enterprise TruthGPT Authentication and Authorization
Advanced auth system with RBAC, OAuth2, and JWT
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets

class AuthMethod(Enum):
    """Authentication method enum."""
    OAUTH2 = "oauth2"
    JWT = "jwt"
    API_KEY = "api_key"
    LDAP = "ldap"
    SAML = "saml"

class Permission(Enum):
    """Permission enum."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

@dataclass
class User:
    """User dataclass."""
    username: str
    email: str
    password_hash: str
    roles: List[str] = None
    permissions: List[Permission] = None
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = ["user"]
        if self.permissions is None:
            self.permissions = [Permission.READ]
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Role:
    """Role dataclass."""
    name: str
    permissions: List[Permission]
    description: str = ""

class EnterpriseAuth:
    """Enterprise authentication and authorization system."""
    
    def __init__(
        self,
        method: AuthMethod = AuthMethod.OAUTH2,
        secret_key: Optional[str] = None
    ):
        self.method = method
        self.secret_key = secret_key or secrets.token_hex(32)
        
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.tokens: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default roles
        self._init_default_roles()
    
    def _init_default_roles(self):
        """Initialize default roles."""
        self.add_role("admin", [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN])
        self.add_role("user", [Permission.READ, Permission.WRITE])
        self.add_role("viewer", [Permission.READ])
    
    def add_role(self, name: str, permissions: List[Permission], description: str = ""):
        """Add role."""
        self.roles[name] = Role(name, permissions, description)
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None
    ) -> User:
        """Create user."""
        password_hash = self._hash_password(password)
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ["user"]
        )
        
        # Set permissions based on roles
        user.permissions = self._get_permissions_for_roles(user.roles)
        
        self.users[username] = user
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return token."""
        if username not in self.users:
            return None
        
        user = self.users[username]
        if not user.is_active:
            return None
        
        password_hash = self._hash_password(password)
        if password_hash != user.password_hash:
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        # Generate token
        token = self._generate_token(user)
        return token
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify token and return user."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            username = payload.get("username")
            
            if username not in self.users:
                return None
            
            user = self.users[username]
            if not user.is_active:
                return None
            
            return user
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission."""
        return permission in user.permissions
    
    def check_authorization(self, user: User, required_permission: Permission) -> bool:
        """Check authorization."""
        return self.has_permission(user, required_permission)
    
    def _hash_password(self, password: str) -> str:
        """Hash password."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self, user: User) -> str:
        """Generate JWT token."""
        payload = {
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # Store token
        self.tokens[token] = {
            "username": user.username,
            "created_at": datetime.now(),
            "expires_at": payload["exp"]
        }
        
        return token
    
    def _get_permissions_for_roles(self, roles: List[str]) -> List[Permission]:
        """Get permissions for roles."""
        permissions = []
        for role_name in roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                permissions.extend(role.permissions)
        return list(set(permissions))  # Remove duplicates
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        active_users = sum(1 for user in self.users.values() if user.is_active)
        total_users = len(self.users)
        total_tokens = len(self.tokens)
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_tokens": total_tokens,
            "roles": list(self.roles.keys())
        }

# Global auth instance
_auth: Optional[EnterpriseAuth] = None

def get_auth() -> EnterpriseAuth:
    """Get or create enterprise auth."""
    global _auth
    if _auth is None:
        _auth = EnterpriseAuth()
    return _auth

# Example usage
if __name__ == "__main__":
    auth = get_auth()
    
    # Create users
    auth.create_user("admin", "admin@truthgpt.com", "admin123", roles=["admin"])
    auth.create_user("user", "user@truthgpt.com", "user123", roles=["user"])
    auth.create_user("viewer", "viewer@truthgpt.com", "viewer123", roles=["viewer"])
    
    # Authenticate
    token = auth.authenticate("admin", "admin123")
    if token:
        print(f"Authentication successful. Token: {token[:20]}...")
    
    # Verify token
    user = auth.verify_token(token)
    if user:
        print(f"Token verified. User: {user.username}")
        print(f"Roles: {user.roles}")
        print(f"Permissions: {[p.value for p in user.permissions]}")
    
    # Check authorization
    can_delete = auth.check_authorization(user, Permission.DELETE)
    print(f"Can delete: {can_delete}")
    
    # Get stats
    stats = auth.get_user_stats()
    print("\nAuth Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")








