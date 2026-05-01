"""
Data validation utilities.
"""
import logging
from typing import Any, List

from .validator import Validator, ValidationResult

logger = logging.getLogger(__name__)


class DataValidator(Validator):
    """Validator for data."""
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate data.
        
        Args:
            data: Data to validate (list of texts)
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        if not isinstance(data, list):
            errors.append("Data must be a list")
            return self._create_result(errors)
        
        if len(data) == 0:
            errors.append("Data is empty")
            return self._create_result(errors)
        
        # Check for empty texts
        empty_count = sum(1 for text in data if not text or not text.strip())
        if empty_count > 0:
            warnings.append(f"{empty_count} empty texts found")
        
        # Check for very long texts
        max_length = kwargs.get("max_length", 10000)
        long_count = sum(1 for text in data if len(text) > max_length)
        if long_count > 0:
            warnings.append(f"{long_count} texts exceed max_length ({max_length})")
        
        # Check data type
        non_string_count = sum(1 for text in data if not isinstance(text, str))
        if non_string_count > 0:
            errors.append(f"{non_string_count} non-string items found")
        
        return self._create_result(errors, warnings)
    
    def validate_split(
        self,
        train_data: List[str],
        val_data: List[str],
        min_train_size: int = 10,
        min_val_size: int = 1,
    ) -> ValidationResult:
        """
        Validate train/validation split.
        
        Args:
            train_data: Training data
            val_data: Validation data
            min_train_size: Minimum training samples
            min_val_size: Minimum validation samples
        
        Returns:
            ValidationResult
        """
        errors = []
        
        if len(train_data) < min_train_size:
            errors.append(f"Training data too small: {len(train_data)} < {min_train_size}")
        
        if len(val_data) < min_val_size:
            errors.append(f"Validation data too small: {len(val_data)} < {min_val_size}")
        
        return self._create_result(errors)



