"""
Configuration Manager for TruthGPT Optimization Core
Handles loading, validation, and management of configuration from multiple sources
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigLoadError(Exception):
    """Raised when configuration loading fails."""
    pass

@dataclass
class ConfigMetadata:
    """Metadata for configuration files."""
    source: str
    format: str
    version: str
    last_modified: str
    checksum: str
    validation_status: str = "pending"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ConfigManager:
    """
    Centralized configuration manager for TruthGPT optimization core.
    
    Supports loading from:
    - YAML files
    - JSON files  
    - Environment variables
    - Default configurations
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.configs: Dict[str, Any] = {}
        self.metadata: Dict[str, ConfigMetadata] = {}
        self.validation_rules: Dict[str, Dict] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load default validation schemas
        self._load_default_schemas()
    
    def load_config_from_file(
        self, 
        file_path: Union[str, Path], 
        config_name: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            config_name: Name for the configuration (defaults to filename)
            validate: Whether to validate the configuration
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigLoadError: If file cannot be loaded
            ConfigValidationError: If validation fails
        """
        file_path = Path(file_path)
        config_name = config_name or file_path.stem
        
        try:
            # Load file based on extension
            if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ConfigLoadError(f"Unsupported file format: {file_path.suffix}")
            
            # Store metadata
            self.metadata[config_name] = ConfigMetadata(
                source=str(file_path),
                format=file_path.suffix.lower(),
                version=config.get('version', '1.0.0'),
                last_modified=str(file_path.stat().st_mtime),
                checksum=self._calculate_checksum(file_path)
            )
            
            # Validate if requested
            if validate:
                self.validate_config(config, config_name)
                self.metadata[config_name].validation_status = "passed"
            else:
                self.metadata[config_name].validation_status = "skipped"
            
            # Store configuration
            self.configs[config_name] = config
            self.logger.info(f"Loaded configuration '{config_name}' from {file_path}")
            
            return config
            
        except Exception as e:
            error_msg = f"Failed to load configuration from {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ConfigLoadError(error_msg) from e
    
    def load_config_from_env(
        self, 
        prefix: str = "TRUTHGPT_",
        config_name: str = "environment"
    ) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables
            config_name: Name for the configuration
            
        Returns:
            Configuration dictionary from environment variables
        """
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested structure
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys (e.g., TRUTHGPT_MODEL_HIDDEN_SIZE -> model.hidden_size)
                if '_' in config_key:
                    parts = config_key.split('_')
                    current = config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    config[config_key] = self._parse_env_value(value)
        
        # Store configuration
        self.configs[config_name] = config
        self.metadata[config_name] = ConfigMetadata(
            source="environment",
            format="env",
            version="1.0.0",
            last_modified="",
            checksum=""
        )
        
        self.logger.info(f"Loaded configuration '{config_name}' from environment variables")
        return config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary
            
        Raises:
            KeyError: If configuration not found
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration '{config_name}' not found")
        
        return self.configs[config_name]
    
    def merge_configs(self, *config_names: str, target_name: str = "merged") -> Dict[str, Any]:
        """
        Merge multiple configurations.
        
        Args:
            *config_names: Names of configurations to merge
            target_name: Name for the merged configuration
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config_name in config_names:
            if config_name in self.configs:
                self._deep_merge(merged, self.configs[config_name])
        
        self.configs[target_name] = merged
        self.logger.info(f"Merged configurations {config_names} into '{target_name}'")
        
        return merged
    
    def validate_config(self, config: Dict[str, Any], config_name: str) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            config_name: Name of the configuration
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigValidationError: If validation fails
        """
        if config_name not in self.validation_rules:
            self.logger.warning(f"No validation rules found for '{config_name}'")
            return True
        
        schema = self.validation_rules[config_name]
        
        try:
            validate(instance=config, schema=schema)
            self.logger.info(f"Configuration '{config_name}' validation passed")
            return True
            
        except ValidationError as e:
            error_msg = f"Configuration validation failed for '{config_name}': {e.message}"
            self.logger.error(error_msg)
            raise ConfigValidationError(error_msg) from e
    
    def add_validation_rule(self, config_name: str, schema: Dict[str, Any]) -> None:
        """
        Add validation rule for configuration.
        
        Args:
            config_name: Name of the configuration
            schema: JSON schema for validation
        """
        self.validation_rules[config_name] = schema
        self.logger.info(f"Added validation rule for '{config_name}'")
    
    def save_config(self, config: Dict[str, Any], file_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            file_path: Path to save the configuration
            format: File format ('yaml' or 'json')
        """
        file_path = Path(file_path)
        
        try:
            if format.lower() == 'yaml':
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved configuration to {file_path}")
            
        except Exception as e:
            error_msg = f"Failed to save configuration to {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise ConfigLoadError(error_msg) from e
    
    def _load_default_schemas(self) -> None:
        """Load default validation schemas."""
        # Transformer configuration schema
        transformer_schema = {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "d_model": {"type": "integer", "minimum": 1},
                        "n_heads": {"type": "integer", "minimum": 1},
                        "n_layers": {"type": "integer", "minimum": 1},
                        "d_ff": {"type": "integer", "minimum": 1},
                        "vocab_size": {"type": "integer", "minimum": 1},
                        "max_seq_length": {"type": "integer", "minimum": 1}
                    },
                    "required": ["d_model", "n_heads", "n_layers", "d_ff", "vocab_size"]
                },
                "optimization": {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"type": "number", "minimum": 0},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "epochs": {"type": "integer", "minimum": 1}
                    }
                }
            },
            "required": ["model"]
        }
        
        self.add_validation_rule("transformer", transformer_schema)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value
        return value
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dictionary into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate checksum for file."""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

# Factory functions
def create_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """Create a new configuration manager."""
    return ConfigManager(config_dir)

def load_config_from_file(
    file_path: Union[str, Path], 
    config_name: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """Load configuration from file using default manager."""
    manager = ConfigManager()
    return manager.load_config_from_file(file_path, config_name, validate)

def load_config_from_env(
    prefix: str = "TRUTHGPT_",
    config_name: str = "environment"
) -> Dict[str, Any]:
    """Load configuration from environment variables using default manager."""
    manager = ConfigManager()
    return manager.load_config_from_env(prefix, config_name)

def validate_config(config: Dict[str, Any], config_name: str) -> bool:
    """Validate configuration using default manager."""
    manager = ConfigManager()
    return manager.validate_config(config, config_name)





