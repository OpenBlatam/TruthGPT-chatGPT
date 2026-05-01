"""
Data transformation utilities for optimization_core.

Provides utilities for transforming and processing data.
"""
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


from pydantic import BaseModel, ConfigDict


class Transformation(BaseModel):
    """Data transformation definition."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    func: Callable
    input_type: type
    output_type: type
    description: Optional[str] = None


class DataTransformer:
    """Transformer for data processing."""
    
    def __init__(self):
        """Initialize data transformer."""
        self.transformations: Dict[str, Transformation] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        input_type: type,
        output_type: type,
        description: Optional[str] = None
    ):
        """
        Register a transformation.
        
        Args:
            name: Transformation name
            func: Transformation function
            input_type: Input type
            output_type: Output type
            description: Optional description
        """
        self.transformations[name] = Transformation(
            name=name,
            func=func,
            input_type=input_type,
            output_type=output_type,
            description=description
        )
        logger.debug(f"Registered transformation: {name}")
    
    def transform(
        self,
        data: Any,
        transformation_name: str
    ) -> Any:
        """
        Apply a transformation.
        
        Args:
            data: Data to transform
            transformation_name: Name of transformation
        
        Returns:
            Transformed data
        """
        if transformation_name not in self.transformations:
            raise ValueError(f"Transformation not found: {transformation_name}")
        
        transformation = self.transformations[transformation_name]
        
        # Type check
        if not isinstance(data, transformation.input_type):
            raise TypeError(
                f"Expected {transformation.input_type.__name__}, "
                f"got {type(data).__name__}"
            )
        
        try:
            result = transformation.func(data)
            
            # Verify output type
            if not isinstance(result, transformation.output_type):
                logger.warning(
                    f"Transformation '{transformation_name}' returned "
                    f"{type(result).__name__}, expected {transformation.output_type.__name__}"
                )
            
            return result
        except Exception as e:
            logger.error(f"Transformation '{transformation_name}' failed: {e}", exc_info=True)
            raise
    
    def pipeline(
        self,
        data: Any,
        transformations: List[str]
    ) -> Any:
        """
        Apply a pipeline of transformations.
        
        Args:
            data: Initial data
            transformations: List of transformation names
        
        Returns:
            Transformed data
        """
        result = data
        
        for transformation_name in transformations:
            result = self.transform(result, transformation_name)
        
        return result
    
    def list_transformations(self) -> List[str]:
        """
        List all registered transformations.
        
        Returns:
            List of transformation names
        """
        return list(self.transformations.keys())


def create_transformer() -> DataTransformer:
    """
    Create a data transformer.
    
    Returns:
        Data transformer
    """
    return DataTransformer()













