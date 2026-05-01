"""
Base configuration classes for optimization_core.

Provides common configuration patterns to reduce duplication.
"""
import logging
from abc import ABC
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Any, Optional, Set, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass


@dataclass
class BaseConfig(ABC):
    """
    Base configuration class with common functionality.
    
    Provides:
    - Automatic validation
    - Dictionary serialization/deserialization
    - Type checking
    - Field validation
    
    Example:
        @dataclass
        class MyConfig(BaseConfig):
            name: str
            value: int = 10
            
            def __post_init__(self):
                super().__post_init__()
                if self.value < 0:
                    raise ConfigValidationError("value must be non-negative")
    """
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate configuration.
        
        Override in subclasses for custom validation.
        """
        pass
    
    def to_dict(self, exclude_none: bool = False, exclude_defaults: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Args:
            exclude_none: Exclude fields with None values
            exclude_defaults: Exclude fields with default values
        
        Returns:
            Dictionary representation
        """
        result = asdict(self)
        
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        
        if exclude_defaults:
            # Get default values from dataclass fields
            field_defaults = {
                f.name: f.default
                for f in fields(self)
                if f.default is not field.MISSING
            }
            result = {
                k: v for k, v in result.items()
                if k not in field_defaults or v != field_defaults[k]
            }
        
        return result
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any], strict: bool = True) -> T:
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            strict: Raise error on unknown fields
        
        Returns:
            Configuration instance
        
        Raises:
            ConfigValidationError: If validation fails
        """
        if strict:
            # Check for unknown fields
            valid_fields = {f.name for f in fields(cls)}
            unknown = set(config_dict.keys()) - valid_fields
            if unknown:
                raise ConfigValidationError(
                    f"Unknown fields in {cls.__name__}: {unknown}. "
                    f"Valid fields: {valid_fields}"
                )
        
        # Filter to only valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        try:
            return cls(**filtered_dict)
        except Exception as e:
            raise ConfigValidationError(
                f"Failed to create {cls.__name__} from dict: {e}"
            ) from e
    
    def update(self, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Field values to update
        
        Raises:
            ConfigValidationError: If validation fails after update
        """
        valid_fields = {f.name for f in fields(self)}
        unknown = set(kwargs.keys()) - valid_fields
        
        if unknown:
            raise ConfigValidationError(
                f"Unknown fields: {unknown}. Valid fields: {valid_fields}"
            )
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Re-validate after update
        self._validate()
    
    def copy(self: T) -> T:
        """
        Create a copy of the configuration.
        
        Returns:
            New configuration instance with same values
        """
        return self.__class__(**self.to_dict())
    
    def merge(self, other: T, overwrite: bool = True) -> T:
        """
        Merge another configuration into this one.
        
        Args:
            other: Configuration to merge
            overwrite: Overwrite existing values
        
        Returns:
            New merged configuration
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        if overwrite:
            merged = {**self_dict, **other_dict}
        else:
            merged = {**other_dict, **self_dict}
        
        return self.__class__.from_dict(merged)


@dataclass
class ValidatedConfig(BaseConfig):
    """
    Configuration with built-in validation helpers.
    
    Provides common validation patterns.
    """
    
    def validate_positive_int(
        self,
        value: int,
        field_name: str,
        min_value: int = 1
    ) -> None:
        """
        Validate positive integer.
        
        Args:
            value: Value to validate
            field_name: Name of field (for error messages)
            min_value: Minimum allowed value
        
        Raises:
            ConfigValidationError: If validation fails
        """
        if not isinstance(value, int):
            raise ConfigValidationError(f"{field_name} must be an integer")
        if value < min_value:
            raise ConfigValidationError(
                f"{field_name} must be >= {min_value}, got {value}"
            )
    
    def validate_positive_float(
        self,
        value: float,
        field_name: str,
        min_value: float = 0.0,
        max_value: Optional[float] = None
    ) -> None:
        """
        Validate positive float in range.
        
        Args:
            value: Value to validate
            field_name: Name of field
            min_value: Minimum allowed value
            max_value: Maximum allowed value (None for no limit)
        
        Raises:
            ConfigValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(f"{field_name} must be a number")
        if value < min_value:
            raise ConfigValidationError(
                f"{field_name} must be >= {min_value}, got {value}"
            )
        if max_value is not None and value > max_value:
            raise ConfigValidationError(
                f"{field_name} must be <= {max_value}, got {value}"
            )
    
    def validate_string(
        self,
        value: str,
        field_name: str,
        min_length: int = 1,
        allowed_values: Optional[Set[str]] = None
    ) -> None:
        """
        Validate string field.
        
        Args:
            value: Value to validate
            field_name: Name of field
            min_length: Minimum length
            allowed_values: Set of allowed values (None for any)
        
        Raises:
            ConfigValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ConfigValidationError(f"{field_name} must be a string")
        if len(value) < min_length:
            raise ConfigValidationError(
                f"{field_name} must have length >= {min_length}"
            )
        if allowed_values is not None and value not in allowed_values:
            raise ConfigValidationError(
                f"{field_name} must be one of {allowed_values}, got {value}"
            )
    
    def validate_optional_positive(
        self,
        value: Optional[int],
        field_name: str,
        min_value: int = 1
    ) -> None:
        """
        Validate optional positive integer.
        
        Args:
            value: Value to validate (can be None)
            field_name: Name of field
            min_value: Minimum allowed value
        
        Raises:
            ConfigValidationError: If validation fails
        """
        if value is not None:
            self.validate_positive_int(value, field_name, min_value)













