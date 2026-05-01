"""
Production Configuration System - Enterprise-grade configuration management
Provides validation, environment-specific configs, and dynamic configuration updates
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading
import time
from contextlib import contextmanager
import jsonschema
from jsonschema import validate, ValidationError
import copy

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ConfigSource(Enum):
    """Configuration sources."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    API = "api"
    DEFAULT = "default"

@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_path: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True

@dataclass
class ConfigMetadata:
    """Configuration metadata."""
    source: ConfigSource
    timestamp: float
    version: str
    environment: Environment
    checksum: str = ""

class ProductionConfig:
    """Production-grade configuration management system."""
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 environment: Environment = Environment.DEVELOPMENT,
                 enable_validation: bool = True,
                 enable_hot_reload: bool = False):
        
        self.environment = environment
        self.enable_validation = enable_validation
        self.enable_hot_reload = enable_hot_reload
        
        # Configuration state
        self.config_data = {}
        self.config_metadata = {}
        self.validation_rules = []
        self.config_schema = None
        
        # Hot reload
        self.file_watcher = None
        self.config_lock = threading.RLock()
        self.update_callbacks = []
        
        # Load initial configuration
        if config_file:
            self.load_from_file(config_file)
        else:
            self._load_default_config()
        
        # Setup hot reload if enabled
        if self.enable_hot_reload and config_file:
            self._setup_hot_reload(config_file)
        
        logger.info(f"🔧 Production Config initialized for {environment.value} environment")
    
    def _load_default_config(self):
        """Load default configuration."""
        self.config_data = {
            'optimization': {
                'level': 'standard',
                'enable_quantization': True,
                'enable_pruning': True,
                'enable_mixed_precision': True,
                'max_memory_gb': 16.0,
                'max_cpu_cores': 8
            },
            'monitoring': {
                'enable_profiling': True,
                'profiling_interval': 100,
                'enable_metrics_collection': True,
                'log_level': 'INFO'
            },
            'performance': {
                'batch_size': 32,
                'max_workers': 4,
                'enable_gpu_acceleration': True,
                'gpu_memory_fraction': 0.8
            },
            'reliability': {
                'max_retry_attempts': 3,
                'retry_delay': 1.0,
                'enable_circuit_breaker': True,
                'circuit_breaker_threshold': 5
            },
            'caching': {
                'enable_result_caching': True,
                'cache_size_mb': 1024,
                'cache_ttl_seconds': 3600
            }
        }
        
        self._update_metadata(ConfigSource.DEFAULT)
    
    def load_from_file(self, filepath: str):
        """Load configuration from file."""
        try:
            file_path = Path(filepath)
            
            if not file_path.exists():
                logger.warning(f"Config file {filepath} not found, using defaults")
                self._load_default_config()
                return
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            # Merge with existing config
            self._merge_config(config)
            self._update_metadata(ConfigSource.FILE, filepath)
            
            logger.info(f"📁 Configuration loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
            self._load_default_config()
    
    def load_from_environment(self, prefix: str = "OPTIMIZATION_"):
        """Load configuration from environment variables."""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested structure
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys (e.g., OPTIMIZATION_LEVEL -> optimization.level)
                if '_' in config_key:
                    parts = config_key.split('_')
                    current = env_config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    env_config[config_key] = self._parse_env_value(value)
        
        if env_config:
            self._merge_config(env_config)
            self._update_metadata(ConfigSource.ENVIRONMENT)
            logger.info(f"🌍 Configuration loaded from environment variables")
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool, List]:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # List values (comma-separated)
        if ',' in value:
            return [self._parse_env_value(item.strip()) for item in value.split(',')]
        
        # String value
        return value
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing."""
        with self.config_lock:
            self._deep_merge(self.config_data, new_config)
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _update_metadata(self, source: ConfigSource, filepath: str = None):
        """Update configuration metadata."""
        self.config_metadata = ConfigMetadata(
            source=source,
            timestamp=time.time(),
            version="1.0.0",
            environment=self.environment,
            checksum=self._calculate_checksum()
        )
    
    def _calculate_checksum(self) -> str:
        """Calculate configuration checksum."""
        import hashlib
        config_str = json.dumps(self.config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path."""
        with self.config_lock:
            keys = key_path.split('.')
            value = self.config_data
            
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value by dot-separated key path."""
        with self.config_lock:
            keys = key_path.split('.')
            config = self.config_data
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            self._update_metadata(ConfigSource.API)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]):
        """Update entire configuration section."""
        with self.config_lock:
            if section not in self.config_data:
                self.config_data[section] = {}
            
            self._deep_merge(self.config_data[section], updates)
            self._update_metadata(ConfigSource.API)
    
    def add_validation_rule(self, rule: ConfigValidationRule):
        """Add configuration validation rule."""
        self.validation_rules.append(rule)
    
    def validate_config(self) -> List[str]:
        """Validate current configuration against rules."""
        errors = []
        
        for rule in self.validation_rules:
            try:
                value = self.get(rule.field_path)
                
                if rule.required and value is None:
                    errors.append(f"Required field '{rule.field_path}' is missing")
                    continue
                
                if value is not None and not rule.validator(value):
                    errors.append(f"Validation failed for '{rule.field_path}': {rule.error_message}")
                    
            except Exception as e:
                errors.append(f"Error validating '{rule.field_path}': {e}")
        
        return errors
    
    def set_schema(self, schema: Dict[str, Any]):
        """Set JSON schema for configuration validation."""
        self.config_schema = schema
    
    def validate_with_schema(self) -> List[str]:
        """Validate configuration against JSON schema."""
        if not self.config_schema:
            return []
        
        try:
            validate(instance=self.config_data, schema=self.config_schema)
            return []
        except ValidationError as e:
            return [f"Schema validation error: {e.message}"]
        except Exception as e:
            return [f"Schema validation failed: {e}"]
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        env_config = copy.deepcopy(self.config_data)
        
        # Apply environment-specific overrides
        if self.environment == Environment.PRODUCTION:
            # Production optimizations
            env_config['optimization']['level'] = 'aggressive'
            env_config['monitoring']['log_level'] = 'WARNING'
            env_config['reliability']['max_retry_attempts'] = 5
            
        elif self.environment == Environment.DEVELOPMENT:
            # Development settings
            env_config['monitoring']['log_level'] = 'DEBUG'
            env_config['optimization']['enable_quantization'] = False
            
        elif self.environment == Environment.TESTING:
            # Testing settings
            env_config['optimization']['max_memory_gb'] = 4.0
            env_config['performance']['batch_size'] = 8
        
        return env_config
    
    def export_config(self, filepath: str, format: str = 'json'):
        """Export current configuration to file."""
        try:
            with open(filepath, 'w') as f:
                if format.lower() == 'json':
                    json.dump(self.config_data, f, indent=2)
                elif format.lower() in ['yaml', 'yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"📤 Configuration exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
    
    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for configuration updates."""
        self.update_callbacks.append(callback)
    
    def _notify_update_callbacks(self):
        """Notify all update callbacks."""
        for callback in self.update_callbacks:
            try:
                callback(self.config_data)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    def _setup_hot_reload(self, config_file: str):
        """Setup hot reload for configuration file."""
        try:
            import watchdog
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class ConfigFileHandler(FileSystemEventHandler):
                def __init__(self, config_manager):
                    self.config_manager = config_manager
                
                def on_modified(self, event):
                    if event.src_path == config_file:
                        self.config_manager.load_from_file(config_file)
                        self.config_manager._notify_update_callbacks()
            
            self.file_watcher = Observer()
            self.file_watcher.schedule(ConfigFileHandler(self), Path(config_file).parent)
            self.file_watcher.start()
            
            logger.info(f"🔄 Hot reload enabled for {config_file}")
            
        except ImportError:
            logger.warning("watchdog not available, hot reload disabled")
        except Exception as e:
            logger.error(f"Failed to setup hot reload: {e}")
    
    def cleanup(self):
        """Cleanup configuration resources."""
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.join()
        
        logger.info("🧹 Configuration cleanup completed")

# Factory functions
def create_production_config(config_file: Optional[str] = None,
                           environment: Environment = Environment.DEVELOPMENT,
                           **kwargs) -> ProductionConfig:
    """Create a production configuration instance."""
    return ProductionConfig(config_file, environment, **kwargs)

def load_config_from_file(filepath: str, environment: Environment = Environment.DEVELOPMENT) -> ProductionConfig:
    """Load configuration from file."""
    return ProductionConfig(config_file=filepath, environment=environment)

def create_environment_config(environment: Environment) -> ProductionConfig:
    """Create environment-specific configuration."""
    config = ProductionConfig(environment=environment)
    return config

# Context manager
@contextmanager
def production_config_context(config_file: Optional[str] = None,
                             environment: Environment = Environment.DEVELOPMENT,
                             **kwargs):
    """Context manager for production configuration."""
    config = create_production_config(config_file, environment, **kwargs)
    try:
        yield config
    finally:
        config.cleanup()

# Validation utilities
def create_optimization_validation_rules() -> List[ConfigValidationRule]:
    """Create validation rules for optimization configuration."""
    return [
        ConfigValidationRule(
            field_path="optimization.max_memory_gb",
            validator=lambda x: isinstance(x, (int, float)) and x > 0,
            error_message="max_memory_gb must be a positive number"
        ),
        ConfigValidationRule(
            field_path="optimization.max_cpu_cores",
            validator=lambda x: isinstance(x, int) and x > 0,
            error_message="max_cpu_cores must be a positive integer"
        ),
        ConfigValidationRule(
            field_path="optimization.level",
            validator=lambda x: x in ['minimal', 'standard', 'aggressive', 'maximum'],
            error_message="optimization.level must be one of: minimal, standard, aggressive, maximum"
        )
    ]

def create_monitoring_validation_rules() -> List[ConfigValidationRule]:
    """Create validation rules for monitoring configuration."""
    return [
        ConfigValidationRule(
            field_path="monitoring.profiling_interval",
            validator=lambda x: isinstance(x, int) and x > 0,
            error_message="profiling_interval must be a positive integer"
        ),
        ConfigValidationRule(
            field_path="monitoring.log_level",
            validator=lambda x: x in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            error_message="log_level must be a valid logging level"
        )
    ]

if __name__ == "__main__":
    print("🔧 Production Configuration System")
    print("=" * 40)
    
    # Example usage
    with production_config_context(environment=Environment.DEVELOPMENT) as config:
        print("✅ Production config created")
        
        # Add validation rules
        config.validation_rules.extend(create_optimization_validation_rules())
        config.validation_rules.extend(create_monitoring_validation_rules())
        
        # Test configuration
        config.set("optimization.level", "aggressive")
        config.set("monitoring.log_level", "INFO")
        
        # Validate configuration
        errors = config.validate_config()
        if errors:
            print(f"❌ Configuration errors: {errors}")
        else:
            print("✅ Configuration is valid")
        
        # Get environment-specific config
        env_config = config.get_environment_config()
        print(f"🌍 Environment config: {env_config['optimization']['level']}")
        
        # Export configuration
        config.export_config("example_config.json")
        print("📤 Configuration exported")
        
        print("✅ Production configuration example completed")

