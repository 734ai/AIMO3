"""
security.py - Security and Compliance Module

Provides secure credential management, input validation,
API key management, and audit logging.

Features:
- Secure credential handling
- Input validation and sanitization
- Rate limiting
- Request signing and validation
- Audit logging
"""

import logging
import os
import re
import hmac
import hashlib
import secrets
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import json


logger = logging.getLogger(__name__)


class CredentialManager:
    """Manages sensitive credentials securely."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize credential manager.
        
        Args:
            env_file: Path to .env file with credentials
        """
        self.credentials: Dict[str, str] = {}
        self.env_file = env_file
        
        if env_file:
            self._load_from_env_file(env_file)
        
        # Load from environment variables
        self._load_from_environment()
    
    def _load_from_env_file(self, env_file: str) -> None:
        """Load credentials from .env file."""
        env_path = Path(env_file)
        
        if not env_path.exists():
            logger.warning(f"Environment file not found: {env_file}")
            return
        
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            self.credentials[key.strip()] = value.strip()
            
            logger.info(f"Loaded credentials from {env_file}")
        except Exception as e:
            logger.error(f"Failed to load credentials: {str(e)}")
    
    def _load_from_environment(self) -> None:
        """Load sensitive credentials from environment variables."""
        sensitive_keys = [
            'KAGGLE_API_KEY',
            'KAGGLE_USERNAME',
            'HF_API_KEY',
            'OPENAI_API_KEY',
            'API_KEY',
            'SECRET_KEY'
        ]
        
        for key in sensitive_keys:
            if key in os.environ:
                self.credentials[key] = os.environ[key]
    
    def get_credential(
        self,
        key: str,
        required: bool = True
    ) -> Optional[str]:
        """
        Get a credential securely.
        
        Args:
            key: Credential key
            required: Raise error if not found
            
        Returns:
            Credential value or None
            
        Raises:
            ValueError: If required credential not found
        """
        if key in self.credentials:
            return self.credentials[key]
        
        if required:
            raise ValueError(f"Required credential not found: {key}")
        
        return None
    
    def set_credential(self, key: str, value: str) -> None:
        """
        Set a credential.
        
        Args:
            key: Credential key
            value: Credential value
        """
        self.credentials[key] = value
        logger.debug(f"Credential set: {key}")
    
    def validate_credentials(self, required_keys: List[str]) -> bool:
        """
        Validate that all required credentials are present.
        
        Args:
            required_keys: List of required credential keys
            
        Returns:
            True if all present
            
        Raises:
            ValueError: If any required credential is missing
        """
        missing = [k for k in required_keys if k not in self.credentials]
        
        if missing:
            raise ValueError(f"Missing required credentials: {', '.join(missing)}")
        
        return True


class InputValidator:
    """Validates and sanitizes input data."""
    
    @staticmethod
    def validate_string(
        value: str,
        min_length: int = 0,
        max_length: int = 10000,
        pattern: Optional[str] = None
    ) -> str:
        """
        Validate string input.
        
        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length
            pattern: Optional regex pattern to match
            
        Returns:
            Validated string
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, str):
            raise ValueError("Expected string")
        
        if len(value) < min_length:
            raise ValueError(f"String too short (min: {min_length})")
        
        if len(value) > max_length:
            raise ValueError(f"String too long (max: {max_length})")
        
        if pattern and not re.match(pattern, value):
            raise ValueError(f"String does not match pattern: {pattern}")
        
        return value
    
    @staticmethod
    def validate_integer(
        value: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """
        Validate integer input.
        
        Args:
            value: Integer to validate
            min_value: Minimum value
            max_value: Maximum value
            
        Returns:
            Validated integer
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, int):
            raise ValueError("Expected integer")
        
        if min_value is not None and value < min_value:
            raise ValueError(f"Value below minimum ({min_value})")
        
        if max_value is not None and value > max_value:
            raise ValueError(f"Value above maximum ({max_value})")
        
        return value
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove path separators and traversal sequences
        sanitized = filename.replace('/', '').replace('\\', '')
        sanitized = sanitized.replace('..', '')
        
        # Remove special characters except dots and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', sanitized)
        
        return sanitized
    
    @staticmethod
    def validate_problem_id(problem_id: str) -> str:
        """
        Validate problem ID format.
        
        Args:
            problem_id: Problem ID to validate
            
        Returns:
            Validated problem ID
            
        Raises:
            ValueError: If validation fails
        """
        if not re.match(r'^[a-zA-Z0-9_-]+$', problem_id):
            raise ValueError("Invalid problem ID format")
        
        if len(problem_id) > 100:
            raise ValueError("Problem ID too long")
        
        return problem_id


class RateLimiter:
    """
    Rate limiting implementation.
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: int
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[datetime] = []
    
    def allow_request(self) -> bool:
        """
        Check if request is allowed under rate limit.
        
        Returns:
            True if request is allowed
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        # Remove old requests outside window
        self.requests = [r for r in self.requests if r > cutoff]
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def get_retry_after(self) -> int:
        """
        Get recommended retry time in seconds.
        
        Returns:
            Seconds until next request allowed
        """
        if not self.requests:
            return 0
        
        oldest_request = min(self.requests)
        retry_after = (oldest_request + timedelta(seconds=self.window_seconds) - datetime.now()).total_seconds()
        
        return max(0, int(retry_after) + 1)


class RequestSigner:
    """Signs and validates API requests."""
    
    @staticmethod
    def sign_request(
        data: Dict[str, Any],
        secret: str,
        algorithm: str = "sha256"
    ) -> str:
        """
        Sign request data.
        
        Args:
            data: Request data to sign
            secret: Secret key for signing
            algorithm: Hash algorithm
            
        Returns:
            Signature
        """
        # Serialize data deterministically
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        # Create HMAC
        signature = hmac.new(
            secret.encode(),
            data_str.encode(),
            getattr(hashlib, algorithm)
        ).hexdigest()
        
        return signature
    
    @staticmethod
    def verify_signature(
        data: Dict[str, Any],
        signature: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Verify request signature.
        
        Args:
            data: Request data
            signature: Signature to verify
            secret: Secret key used for signing
            algorithm: Hash algorithm
            
        Returns:
            True if signature is valid
        """
        expected_signature = RequestSigner.sign_request(data, secret, algorithm)
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected_signature)


class AuditLogger:
    """Logs security-relevant events."""
    
    def __init__(self, log_file: str = "audit.log"):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(
        self,
        event_type: str,
        user: str,
        action: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            user: User performing action
            action: Action performed
            resource: Resource affected
            result: Result of action
            details: Additional details
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user": user,
            "action": action,
            "resource": resource,
            "result": result,
            "details": details or {}
        }
        
        # Log to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {str(e)}")
        
        # Also log to standard logger
        logger.info(f"AUDIT: {event_type} - {action} on {resource} - {result}")
    
    def log_authentication(
        self,
        user: str,
        success: bool,
        ip_address: Optional[str] = None
    ) -> None:
        """Log authentication attempt."""
        self.log_event(
            "AUTHENTICATION",
            user,
            "login",
            "user",
            "success" if success else "failed",
            {"ip_address": ip_address}
        )
    
    def log_api_call(
        self,
        user: str,
        endpoint: str,
        method: str,
        status_code: int
    ) -> None:
        """Log API call."""
        self.log_event(
            "API_CALL",
            user,
            method,
            endpoint,
            f"status_{status_code}"
        )


def require_authentication(func):
    """Decorator to require valid credentials."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for credential in kwargs or environment
        credential = kwargs.get('credential') or os.environ.get('API_KEY')
        
        if not credential:
            raise ValueError("Authentication required")
        
        return func(*args, **kwargs)
    
    return wrapper
