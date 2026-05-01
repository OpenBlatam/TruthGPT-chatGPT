"""
Serialization utilities for polyglot_core.

Provides serialization and deserialization for cache, models, and configurations.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import json
import pickle
import numpy as np


class Serializer:
    """
    Serializer for polyglot_core objects.
    
    Supports JSON, Pickle, and NumPy formats.
    """
    
    @staticmethod
    def serialize_cache_entry(key: Any, value: Any, format: str = "pickle") -> bytes:
        """
        Serialize cache entry.
        
        Args:
            key: Cache key
            value: Cache value
            format: Serialization format ("pickle", "json", "numpy")
            
        Returns:
            Serialized bytes
        """
        if format == "pickle":
            return pickle.dumps({'key': key, 'value': value})
        elif format == "json":
            # Convert numpy arrays to lists
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            data = {'key': convert(key), 'value': convert(value)}
            return json.dumps(data).encode('utf-8')
        elif format == "numpy":
            # Save as numpy format
            import io
            buffer = io.BytesIO()
            np.savez(buffer, key=key, value=value)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def deserialize_cache_entry(data: bytes, format: str = "pickle") -> tuple:
        """
        Deserialize cache entry.
        
        Args:
            data: Serialized bytes
            format: Serialization format
            
        Returns:
            Tuple of (key, value)
        """
        if format == "pickle":
            obj = pickle.loads(data)
            return obj['key'], obj['value']
        elif format == "json":
            obj = json.loads(data.decode('utf-8'))
            # Convert lists back to numpy arrays if needed
            def convert(obj):
                if isinstance(obj, list):
                    try:
                        return np.array(obj)
                    except:
                        return obj
                return obj
            return convert(obj['key']), convert(obj['value'])
        elif format == "numpy":
            import io
            buffer = io.BytesIO(data)
            loaded = np.load(buffer)
            return loaded['key'], loaded['value']
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: Union[str, Path], format: str = "json"):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            filepath: Output file path
            format: File format ("json", "yaml")
        """
        path = Path(filepath)
        
        if format == "json":
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        elif format == "yaml":
            import yaml
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def load_config(filepath: Union[str, Path], format: str = "json") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            filepath: Input file path
            format: File format ("json", "yaml")
            
        Returns:
            Configuration dictionary
        """
        path = Path(filepath)
        
        if format == "json":
            with open(path, 'r') as f:
                return json.load(f)
        elif format == "yaml":
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def save_tensor(tensor: np.ndarray, filepath: Union[str, Path], format: str = "numpy"):
        """
        Save tensor to file.
        
        Args:
            tensor: NumPy array
            filepath: Output file path
            format: File format ("numpy", "json")
        """
        path = Path(filepath)
        
        if format == "numpy":
            np.save(path, tensor)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(tensor.tolist(), f)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def load_tensor(filepath: Union[str, Path], format: str = "numpy") -> np.ndarray:
        """
        Load tensor from file.
        
        Args:
            filepath: Input file path
            format: File format ("numpy", "json")
            
        Returns:
            NumPy array
        """
        path = Path(filepath)
        
        if format == "numpy":
            return np.load(path)
        elif format == "json":
            with open(path, 'r') as f:
                data = json.load(f)
            return np.array(data)
        else:
            raise ValueError(f"Unknown format: {format}")


def serialize_cache_entry(key: Any, value: Any, format: str = "pickle") -> bytes:
    """Convenience function to serialize cache entry."""
    return Serializer.serialize_cache_entry(key, value, format)


def deserialize_cache_entry(data: bytes, format: str = "pickle") -> tuple:
    """Convenience function to deserialize cache entry."""
    return Serializer.deserialize_cache_entry(data, format)













