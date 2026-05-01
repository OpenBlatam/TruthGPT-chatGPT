"""
Base validation framework.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class Validator(ABC):
    """Base validator class."""
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate data.
        
        Args:
            data: Data to validate
            **kwargs: Additional validation parameters
        
        Returns:
            ValidationResult
        """
        pass
    
    def validate_and_raise(self, data: Any, **kwargs) -> None:
        """
        Validate and raise exception if invalid.
        
        Args:
            data: Data to validate
            **kwargs: Additional validation parameters
        
        Raises:
            ValueError: If validation fails
        """
        result = self.validate(data, **kwargs)
        if not result.valid:
            error_msg = "; ".join(result.errors)
            raise ValueError(f"Validation failed: {error_msg}")
    
    def _create_result(
        self,
        errors: list[str],
        warnings: Optional[list[str]] = None
    ) -> ValidationResult:
        """
        Create a ValidationResult with consistent pattern.
        Centralizes the repetitive pattern of creating ValidationResult.
        
        Args:
            errors: List of error messages
            warnings: Optional list of warning messages
        
        Returns:
            ValidationResult with valid=True if no errors, False otherwise
        """
        warnings = warnings or []
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class CompositeValidator(Validator):
    """
    Validator that combines multiple validators.
    """
    
    def __init__(self, validators: List[Validator]):
        """
        Initialize composite validator.
        
        Args:
            validators: List of validators to combine
        """
        self.validators = validators
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate using all validators."""
        all_errors = []
        all_warnings = []
        
        for validator in self.validators:
            result = validator.validate(data, **kwargs)
            if not result.valid:
                all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return self._create_result(all_errors, all_warnings)



