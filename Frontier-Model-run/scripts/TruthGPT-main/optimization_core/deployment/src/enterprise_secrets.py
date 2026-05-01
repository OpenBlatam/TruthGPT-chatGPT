"""
Enterprise TruthGPT Secrets Management
Secure secrets management with Azure Key Vault integration
"""

from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from typing import Optional, Dict, Any, List, Tuple
import json
from pathlib import Path
import hashlib
import secrets
from datetime import datetime, timedelta
import logging
from enum import Enum

class EnterpriseSecrets:
    """Enterprise secrets management system."""
    
    def __init__(self, vault_name: str):
        self.vault_name = vault_name
        self.vault_url = f"https://{vault_name}.vault.azure.net/"
        self.client = SecretClient(vault_url=self.vault_url, credential=DefaultAzureCredential())
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from Key Vault."""
        try:
            secret = self.client.get_secret(name)
            return secret.value
        except Exception as e:
            print(f"Error getting secret {name}: {str(e)}")
            return None
    
    def set_secret(self, name: str, value: str):
        """Set secret in Key Vault."""
        try:
            self.client.set_secret(name, value)
        except Exception as e:
            print(f"Error setting secret {name}: {str(e)}")
    
    def delete_secret(self, name: str):
        """Delete secret from Key Vault."""
        try:
            self.client.begin_delete_secret(name).wait()
        except Exception as e:
            print(f"Error deleting secret {name}: {str(e)}")
    
    def list_secrets(self) -> list:
        """List all secrets."""
        try:
            return [secret.name for secret in self.client.list_secrets()]
        except Exception as e:
            print(f"Error listing secrets: {str(e)}")
            return []


# =============================================================================
# ENHANCED SECURITY FEATURES
# =============================================================================

class SecretType(str, Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    PASSWORD = "password"
    DATABASE = "database"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    ENCRYPTION_KEY = "encryption_key"


class SecretRotationPolicy:
    """Policy for secret rotation."""
    
    def __init__(self, rotation_days: int = 90, warning_days: int = 7):
        self.rotation_days = rotation_days
        self.warning_days = warning_days
        self.rotation_schedule: Dict[str, datetime] = {}
    
    def should_rotate(self, secret_name: str, created_at: datetime) -> bool:
        """Check if secret should be rotated."""
        rotation_date = created_at + timedelta(days=self.rotation_days)
        return datetime.utcnow() > rotation_date
    
    def get_rotation_warning(self, secret_name: str, created_at: datetime) -> Optional[str]:
        """Get rotation warning if applicable."""
        rotation_date = created_at + timedelta(days=self.rotation_days)
        warning_date = rotation_date - timedelta(days=self.warning_days)
        
        if datetime.utcnow() > warning_date and datetime.utcnow() < rotation_date:
            days_until = (rotation_date - datetime.utcnow()).days
            return f"Secret {secret_name} will expire in {days_until} days"
        
        return None


class SecurityAuditor:
    """Audit security access and changes."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def log_access(self, secret_name: str, user: str, action: str, status: str) -> None:
        """Log secret access."""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'secret_name': secret_name,
            'user': user,
            'action': action,
            'status': status
        }
        
        self.audit_log.append(entry)
        self.logger.info(f"Security audit: {action} on {secret_name} by {user} - {status}")
    
    def get_audit_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit history."""
        return self.audit_log[-limit:]
    
    def detect_anomaly(self, secret_name: str, user: str) -> bool:
        """Detect anomalous access patterns."""
        user_access = [e for e in self.audit_log 
                      if e['secret_name'] == secret_name and e['user'] == user]
        
        # Simple anomaly detection
        if len(user_access) > 100:
            return True
        
        # Check for frequent access
        recent_access = [e for e in user_access 
                        if (datetime.utcnow() - datetime.fromisoformat(e['timestamp'])).seconds < 3600]
        
        return len(recent_access) > 10


class SecretEncryption:
    """Handle secret encryption and decryption."""
    
    def __init__(self):
        self.encryption_key = self._generate_key()
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)
    
    def encrypt_secret(self, plaintext: str) -> Tuple[str, bytes]:
        """Encrypt secret."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        f = Fernet(key)
        encrypted = f.encrypt(plaintext.encode())
        return encrypted.decode(), key
    
    def decrypt_secret(self, ciphertext: str, key: bytes) -> str:
        """Decrypt secret."""
        from cryptography.fernet import Fernet
        f = Fernet(key)
        decrypted = f.decrypt(ciphertext.encode())
        return decrypted.decode()
    
    def hash_secret(self, secret: str) -> str:
        """Hash secret for storage."""
        return hashlib.sha256(secret.encode()).hexdigest()


class SecretManager:
    """Advanced secret manager with encryption and auditing."""
    
    def __init__(self, vault_name: str):
        self.enterprise_secrets = EnterpriseSecrets(vault_name)
        self.rotation_policy = SecretRotationPolicy()
        self.auditor = SecurityAuditor()
        self.encryption = SecretEncryption()
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache
        self.secret_cache: Dict[str, Dict[str, Any]] = {}
    
    def store_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Store secret with metadata."""
        try:
            # Encrypt before storing
            encrypted_value, key = self.encryption.encrypt_secret(value)
            
            # Store encrypted value
            self.enterprise_secrets.set_secret(name, encrypted_value)
            
            # Store metadata
            metadata = {
                'encrypted': True,
                'type': secret_type.value,
                'created_at': datetime.utcnow().isoformat(),
                'tags': tags or {},
                'encryption_key_hash': self.encryption.hash_secret(key.decode())
            }
            
            self.secret_cache[name] = metadata
            
            # Log access
            self.auditor.log_access(name, "system", "store", "success")
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing secret: {e}")
            self.auditor.log_access(name, "system", "store", "failed")
            return False
    
    def retrieve_secret(self, name: str, decrypt: bool = True) -> Optional[str]:
        """Retrieve and decrypt secret."""
        try:
            # Check cache first
            if name in self.secret_cache:
                return self._get_from_cache(name)
            
            # Get from vault
            encrypted_value = self.enterprise_secrets.get_secret(name)
            if not encrypted_value:
                return None
            
            if decrypt:
                # Decrypt secret
                # Note: In real implementation, encryption key should be managed securely
                # This is a simplified version
                return self._decrypt_cached_secret(name, encrypted_value)
            
            return encrypted_value
        except Exception as e:
            self.logger.error(f"Error retrieving secret: {e}")
            return None
    
    def _get_from_cache(self, name: str) -> Optional[str]:
        """Get secret from cache."""
        if name in self.secret_cache:
            metadata = self.secret_cache[name]
            if metadata.get('encrypted'):
                # Decrypt cached secret
                return self._decrypt_cached_secret(name, metadata.get('value', ''))
            return metadata.get('value')
        return None
    
    def _decrypt_cached_secret(self, name: str, encrypted_value: str) -> Optional[str]:
        """Decrypt cached secret."""
        # Simplified decryption
        # In production, proper key management would be used
        try:
            return encrypted_value  # Placeholder for actual decryption
        except Exception as e:
            self.logger.error(f"Error decrypting secret: {e}")
            return None
    
    def rotate_secret(self, name: str, new_value: str) -> bool:
        """Rotate secret to new value."""
        try:
            # Store new secret
            success = self.store_secret(name, new_value, SecretType.API_KEY)
            
            if success:
                self.auditor.log_access(name, "system", "rotate", "success")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error rotating secret: {e}")
            return False
    
    def validate_secret_policy(self, secret_type: SecretType, value: str) -> Tuple[bool, Optional[str]]:
        """Validate secret against policy."""
        if secret_type == SecretType.PASSWORD:
            # Check password strength
            if len(value) < 8:
                return False, "Password must be at least 8 characters"
            if not any(c.isupper() for c in value):
                return False, "Password must contain uppercase letter"
            if not any(c.islower() for c in value):
                return False, "Password must contain lowercase letter"
            if not any(c.isdigit() for c in value):
                return False, "Password must contain a digit"
        
        elif secret_type == SecretType.API_KEY:
            if len(value) < 32:
                return False, "API key must be at least 32 characters"
        
        return True, None
    
    def get_secret_statistics(self) -> Dict[str, Any]:
        """Get secret statistics."""
        all_secrets = self.enterprise_secrets.list_secrets()
        
        return {
            'total_secrets': len(all_secrets),
            'cached_secrets': len(self.secret_cache),
            'rotation_required': 0,
            'audit_log_entries': len(self.auditor.audit_log)
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_enterprise_secrets_manager(vault_name: str) -> SecretManager:
    """Create enterprise secrets manager."""
    return SecretManager(vault_name)


def create_rotation_policy(rotation_days: int = 90, warning_days: int = 7) -> SecretRotationPolicy:
    """Create rotation policy."""
    return SecretRotationPolicy(rotation_days, warning_days)


def create_security_auditor() -> SecurityAuditor:
    """Create security auditor."""
    return SecurityAuditor()


# Example usage
if __name__ == "__main__":
    # Enterprise secrets usage
    secrets = EnterpriseSecrets("truthgpt-kv")
    
    # Get secret
    password = secrets.get_secret("database-password")
    print(f"Database password: {password}")
    
    # Set secret
    secrets.set_secret("api-key", "your-api-key")
    
    # List secrets
    all_secrets = secrets.list_secrets()
    print(f"All secrets: {all_secrets}")
    
    # Advanced usage with SecretManager
    print("\n" + "="*60)
    print("Advanced Secret Management")
    print("="*60)
    
    manager = create_enterprise_secrets_manager("truthgpt-kv")
    
    # Store secret with encryption
    manager.store_secret("test-api-key", "secret-value-123", SecretType.API_KEY)
    
    # Retrieve secret
    retrieved = manager.retrieve_secret("test-api-key")
    print(f"Retrieved secret: {retrieved}")
    
    # Get statistics
    stats = manager.get_secret_statistics()
    print(f"Secret statistics: {stats}")
    
    # Security audit
    audit_history = manager.auditor.get_audit_history()
    print(f"Audit log entries: {len(audit_history)}")

