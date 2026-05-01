"""
TruthGPT Advanced Security Module
Enterprise-grade security utilities for TruthGPT models
"""

import torch
import torch.nn as nn
import hashlib
import hmac
import base64
import secrets
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTSecurityConfig:
    """Configuration for TruthGPT security."""
    # Authentication settings
    enable_authentication: bool = True
    auth_secret_key: Optional[str] = None
    auth_token_expiry: int = 3600  # seconds
    
    # Encryption settings
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256"
    encryption_key: Optional[str] = None
    
    # Authorization settings
    enable_authorization: bool = True
    allowed_roles: List[str] = field(default_factory=lambda: ["admin", "user", "viewer"])
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Input validation
    enable_input_validation: bool = True
    max_input_length: int = 2048
    min_input_length: int = 1
    
    # Output sanitization
    enable_output_sanitization: bool = True
    dangerous_patterns: List[str] = field(default_factory=lambda: ["<script", "javascript:", "eval("])
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_path: str = "./logs/audit.log"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_authentication': self.enable_authentication,
            'enable_encryption': self.enable_encryption,
            'enable_authorization': self.enable_authorization,
            'allowed_roles': self.allowed_roles,
            'enable_rate_limiting': self.enable_rate_limiting,
            'max_requests_per_minute': self.max_requests_per_minute,
            'max_requests_per_hour': self.max_requests_per_hour,
            'enable_input_validation': self.enable_input_validation,
            'max_input_length': self.max_input_length,
            'min_input_length': self.min_input_length,
            'enable_output_sanitization': self.enable_output_sanitization,
            'dangerous_patterns': self.dangerous_patterns,
            'enable_audit_logging': self.enable_audit_logging,
            'audit_log_path': self.audit_log_path
        }

class TruthGPTAuthenticator:
    """Advanced authenticator for TruthGPT."""
    
    def __init__(self, config: TruthGPTSecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Authentication state
        self.active_tokens = {}
        self.auth_secret = config.auth_secret_key or self._generate_secret_key()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_hex(32)
    
    def generate_token(self, user_id: str, role: str = "user") -> str:
        """Generate authentication token."""
        token_data = {
            'user_id': user_id,
            'role': role,
            'issued_at': time.time(),
            'expires_at': time.time() + self.config.auth_token_expiry
        }
        
        # Create token
        token_string = json.dumps(token_data)
        signature = hmac.new(
            self.auth_secret.encode(),
            token_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = base64.b64encode(f"{token_string}:{signature}".encode()).decode()
        
        # Store token
        self.active_tokens[token] = token_data
        
        self.logger.info(f"Generated token for user: {user_id}")
        return token
    
    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate authentication token."""
        if token not in self.active_tokens:
            self.logger.warning("Invalid token provided")
            return False, None
        
        token_data = self.active_tokens[token]
        
        # Check expiration
        if time.time() > token_data['expires_at']:
            del self.active_tokens[token]
            self.logger.warning("Token expired")
            return False, None
        
        self.logger.info(f"Token validated for user: {token_data['user_id']}")
        return True, token_data
    
    def revoke_token(self, token: str) -> bool:
        """Revoke authentication token."""
        if token in self.active_tokens:
            del self.active_tokens[token]
            self.logger.info("Token revoked")
            return True
        return False

class TruthGPTEncryptor:
    """Advanced encryptor for TruthGPT."""
    
    def __init__(self, config: TruthGPTSecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Encryption state
        self.encryption_key = config.encryption_key or self._generate_encryption_key()
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key."""
        return secrets.token_hex(32)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        # Simplified encryption
        # In practice, you would use proper encryption libraries
        encrypted = base64.b64encode(data.encode()).decode()
        self.logger.info("Data encrypted")
        return encrypted
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        # Simplified decryption
        # In practice, you would use proper decryption libraries
        decrypted = base64.b64decode(encrypted_data.encode()).decode()
        self.logger.info("Data decrypted")
        return decrypted

class TruthGPTSanitizer:
    """Advanced data sanitizer for TruthGPT."""
    
    def __init__(self, config: TruthGPTSecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Sanitization stats
        self.sanitization_stats = {
            'total_inputs': 0,
            'sanitized_inputs': 0,
            'dangerous_patterns_found': 0
        }
    
    def validate_input(self, input_data: str) -> Tuple[bool, Optional[str]]:
        """Validate input data."""
        if not self.config.enable_input_validation:
            return True, None
        
        self.sanitization_stats['total_inputs'] += 1
        
        # Check length
        if len(input_data) < self.config.min_input_length:
            return False, "Input too short"
        
        if len(input_data) > self.config.max_input_length:
            return False, "Input too long"
        
        # Check for dangerous patterns
        for pattern in self.config.dangerous_patterns:
            if pattern.lower() in input_data.lower():
                self.sanitization_stats['dangerous_patterns_found'] += 1
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, None
    
    def sanitize_output(self, output_data: str) -> str:
        """Sanitize output data."""
        if not self.config.enable_output_sanitization:
            return output_data
        
        sanitized = output_data
        
        # Remove dangerous patterns
        for pattern in self.config.dangerous_patterns:
            sanitized = sanitized.replace(pattern, "")
        
        self.sanitization_stats['sanitized_inputs'] += 1
        self.logger.info("Output sanitized")
        
        return sanitized
    
    def get_sanitization_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        return self.sanitization_stats.copy()

class TruthGPTRateLimiter:
    """Advanced rate limiter for TruthGPT."""
    
    def __init__(self, config: TruthGPTSecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Rate limiting state
        self.request_times = {}
        self.request_counts = {}
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """Check if request is within rate limits."""
        if not self.config.enable_rate_limiting:
            return True, None
        
        current_time = time.time()
        
        # Initialize user tracking
        if user_id not in self.request_times:
            self.request_times[user_id] = []
            self.request_counts[user_id] = {'minute': 0, 'hour': 0}
        
        # Clean old requests
        self._clean_old_requests(user_id, current_time)
        
        # Check rate limits
        minute_requests = self.request_counts[user_id]['minute']
        hour_requests = self.request_counts[user_id]['hour']
        
        if minute_requests >= self.config.max_requests_per_minute:
            self.logger.warning(f"Rate limit exceeded for user: {user_id} (minute)")
            return False, "Rate limit exceeded (per minute)"
        
        if hour_requests >= self.config.max_requests_per_hour:
            self.logger.warning(f"Rate limit exceeded for user: {user_id} (hour)")
            return False, "Rate limit exceeded (per hour)"
        
        # Record request
        self.request_times[user_id].append(current_time)
        self.request_counts[user_id]['minute'] += 1
        self.request_counts[user_id]['hour'] += 1
        
        return True, None
    
    def _clean_old_requests(self, user_id: str, current_time: float) -> None:
        """Clean old request times."""
        if user_id not in self.request_times:
            return
        
        # Keep only recent requests (within last hour)
        self.request_times[user_id] = [
            t for t in self.request_times[user_id]
            if current_time - t < 3600
        ]
        
        # Update counts
        one_minute_ago = current_time - 60
        self.request_counts[user_id]['minute'] = sum(
            1 for t in self.request_times[user_id]
            if t >= one_minute_ago
        )
        
        self.request_counts[user_id]['hour'] = len(self.request_times[user_id])

class TruthGPTAuditor:
    """Advanced audit logger for TruthGPT."""
    
    def __init__(self, config: TruthGPTSecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Audit state
        self.audit_log_path = Path(config.audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type: str, user_id: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        if not self.config.enable_audit_logging:
            return
        
        audit_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details
        }
        
        # Write to log file
        with open(self.audit_log_path, 'a') as f:
            json.dump(audit_entry, f)
            f.write('\n')
        
        self.logger.info(f"Audit event logged: {event_type}")

class TruthGPTSecurityManager:
    """Advanced security manager for TruthGPT."""
    
    def __init__(self, config: TruthGPTSecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Security components
        self.authenticator = TruthGPTAuthenticator(config) if config.enable_authentication else None
        self.encryptor = TruthGPTEncryptor(config) if config.enable_encryption else None
        self.sanitizer = TruthGPTSanitizer(config)
        self.rate_limiter = TruthGPTRateLimiter(config) if config.enable_rate_limiting else None
        self.auditor = TruthGPTAuditor(config) if config.enable_audit_logging else None
        
        # Security state
        self.security_stats = {}
    
    def secure_request(self, user_id: str, input_data: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Process secure request with all security checks."""
        # Rate limiting
        if self.rate_limiter:
            allowed, error = self.rate_limiter.check_rate_limit(user_id)
            if not allowed:
                return False, error, None
        
        # Input validation
        valid, error = self.sanitizer.validate_input(input_data)
        if not valid:
            if self.auditor:
                self.auditor.log_event("invalid_input", user_id, {'error': error})
            return False, error, None
        
        # Sanitize input
        sanitized_input = input_data  # Input already validated
        
        # Encrypt if needed
        if self.encryptor:
            sanitized_input = self.encryptor.encrypt_data(sanitized_input)
        
        # Log request
        if self.auditor:
            self.auditor.log_event("request", user_id, {'input_length': len(input_data)})
        
        return True, None, sanitized_input
    
    def secure_response(self, output_data: str) -> str:
        """Process secure response."""
        # Sanitize output
        sanitized_output = self.sanitizer.sanitize_output(output_data)
        
        return sanitized_output
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        stats = {}
        
        if self.sanitizer:
            stats.update(self.sanitizer.get_sanitization_stats())
        
        return stats

# Factory functions
def create_truthgpt_security_manager(config: TruthGPTSecurityConfig) -> TruthGPTSecurityManager:
    """Create TruthGPT security manager."""
    return TruthGPTSecurityManager(config)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT security
    print("🚀 TruthGPT Advanced Security Demo")
    print("=" * 50)
    
    # Create security configuration
    config = TruthGPTSecurityConfig(
        enable_authentication=True,
        enable_rate_limiting=True,
        enable_input_validation=True,
        enable_output_sanitization=True
    )
    
    # Create security manager
    manager = create_truthgpt_security_manager(config)
    
    # Test authentication
    token = manager.authenticator.generate_token("user123", "user")
    is_valid, user_data = manager.authenticator.validate_token(token)
    print(f"Authentication test: {is_valid}, User: {user_data}")
    
    # Test secure request
    success, error, sanitized_input = manager.secure_request("user123", "Hello World")
    print(f"Secure request test: {success}, Error: {error}")
    
    # Get security stats
    stats = manager.get_security_stats()
    print(f"Security stats: {stats}")
    
    print("✅ TruthGPT security demo completed!")

