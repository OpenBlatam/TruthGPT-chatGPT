"""
Schema validation utilities for optimization_core.

Provides utilities for validating data structures against schemas.
"""
import logging
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import fields
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Validation error."""
    pass


from .base import BaseOptimizationModel


class FieldSchema(BaseOptimizationModel):
    """Schema for a field."""

    type: Type
    required: bool = True
    default: Any = None
    validator: Optional[Callable] = None
    description: Optional[str] = None


class SchemaValidator:
    """Validator for data structures."""
    
    def __init__(self, schema: Dict[str, FieldSchema]):
        """
        Initialize schema validator.
        
        Args:
            schema: Schema definition
        """
        self.schema = schema
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
        
        Returns:
            Validated data
        
        Raises:
            ValidationError: If validation fails
        """
        validated = {}
        errors = []
        
        # Check required fields
        for field_name, field_schema in self.schema.items():
            if field_name not in data:
                if field_schema.required:
                    errors.append(f"Missing required field: {field_name}")
                else:
                    validated[field_name] = field_schema.default
                continue
            
            value = data[field_name]
            
            # Type check
            if not isinstance(value, field_schema.type):
                errors.append(
                    f"Field '{field_name}' must be of type {field_schema.type.__name__}, "
                    f"got {type(value).__name__}"
                )
                continue
            
            # Custom validator
            if field_schema.validator:
                try:
                    if not field_schema.validator(value):
                        errors.append(f"Field '{field_name}' failed custom validation")
                        continue
                except Exception as e:
                    errors.append(f"Field '{field_name}' validation error: {e}")
                    continue
            
            validated[field_name] = value
        
        # Check for extra fields
        for field_name in data:
            if field_name not in self.schema:
                errors.append(f"Unknown field: {field_name}")
        
        if errors:
            raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        return validated
    
    def validate_partial(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate partial data (only validate provided fields).
        
        Args:
            data: Partial data to validate
        
        Returns:
            Validated data
        """
        validated = {}
        errors = []
        
        for field_name, value in data.items():
            if field_name not in self.schema:
                errors.append(f"Unknown field: {field_name}")
                continue
            
            field_schema = self.schema[field_name]
            
            # Type check
            if not isinstance(value, field_schema.type):
                errors.append(
                    f"Field '{field_name}' must be of type {field_schema.type.__name__}, "
                    f"got {type(value).__name__}"
                )
                continue
            
            # Custom validator
            if field_schema.validator:
                try:
                    if not field_schema.validator(value):
                        errors.append(f"Field '{field_name}' failed custom validation")
                        continue
                except Exception as e:
                    errors.append(f"Field '{field_name}' validation error: {e}")
                    continue
            
            validated[field_name] = value
        
        if errors:
            raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        return validated


def validate_dataclass(
    data: Dict[str, Any],
    dataclass_type: Type
) -> Dict[str, Any]:
    """
    Validate data against a dataclass.
    
    Args:
        data: Data to validate
        dataclass_type: Dataclass type
    
    Returns:
        Validated data
    """
    schema = {}
    
    for field in fields(dataclass_type):
        schema[field.name] = FieldSchema(
            type=field.type,
            required=field.default is None and field.default_factory is None,
            default=field.default if field.default is not None else field.default_factory() if field.default_factory else None
        )
    
    validator = SchemaValidator(schema)
    return validator.validate(data)













