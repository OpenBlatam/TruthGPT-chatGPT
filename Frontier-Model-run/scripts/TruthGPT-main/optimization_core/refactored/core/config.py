"""
Unified Configuration Management
===============================

Advanced configuration system with:
- Multiple config sources (YAML, JSON, environment variables)
- Configuration validation and schema
- Hot reloading
- Environment-specific configs
- Secret management
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ConfigSource(Enum):
    """Configuration source types"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DEFAULTS = "defaults"
    OVERRIDE = "override"


@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    key: str
    type: Type
    required: bool = False
    default: Any = None
    description: str = ""
    validation: Optional[callable] = None


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for config hot reloading"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            self.logger.info(f"Config file changed: {event.src_path}")
            self.config_manager.reload_config()


class UnifiedConfig:
    """
    Unified configuration management system.
    
    Features:
    - Multiple configuration sources
    - Schema validation
    - Hot reloading
    - Environment-specific configs
    - Secret management
    - Type safety
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else None
        self.config_data: Dict[str, Any] = {}
        self.schema: Dict[str, ConfigSchema] = {}
        self.sources: Dict[str, ConfigSource] = {}
        self.lock = threading.RLock()
        self.observers = []
        
        # Default configuration
        self._setup_default_schema()
        self._load_defaults()
        
        # Load configuration
        if self.config_path:
            self._load_config_file()
            self._setup_file_watcher()
        
        # Load environment variables
        self._load_environment_variables()
    
    def _setup_default_schema(self):
        """Setup default configuration schema"""
        default_schema = [
            ConfigSchema('framework.name', str, default='TruthGPT Optimization Framework'),
            ConfigSchema('framework.version', str, default='3.0.0'),
            ConfigSchema('framework.debug', bool, default=False),
            
            ConfigSchema('optimization.max_workers', int, default=4),
            ConfigSchema('optimization.max_processes', int, default=2),
            ConfigSchema('optimization.timeout', float, default=300.0),
            ConfigSchema('optimization.retry_attempts', int, default=3),
            
            ConfigSchema('cache.enabled', bool, default=True),
            ConfigSchema('cache.max_size', int, default=1000),
            ConfigSchema('cache.ttl', int, default=3600),
            ConfigSchema('cache.backend', str, default='memory'),
            
            ConfigSchema('monitoring.enabled', bool, default=True),
            ConfigSchema('monitoring.metrics_interval', int, default=60),
            ConfigSchema('monitoring.log_level', str, default='INFO'),
            
            ConfigSchema('api.enabled', bool, default=True),
            ConfigSchema('api.host', str, default='localhost'),
            ConfigSchema('api.port', int, default=8000),
            ConfigSchema('api.cors_enabled', bool, default=True),
            
            ConfigSchema('plugins.enabled', bool, default=True),
            ConfigSchema('plugins.path', str, default='plugins'),
            ConfigSchema('plugins.auto_load', bool, default=True),
            
            ConfigSchema('security.secret_key', str, required=True),
            ConfigSchema('security.encryption_enabled', bool, default=True),
            ConfigSchema('security.max_request_size', int, default=10485760),  # 10MB
        ]
        
        for schema in default_schema:
            self.schema[schema.key] = schema
    
    def _load_defaults(self):
        """Load default configuration values"""
        with self.lock:
            for key, schema in self.schema.items():
                if schema.default is not None:
                    self.config_data[key] = schema.default
                    self.sources[key] = ConfigSource.DEFAULTS
    
    def _load_config_file(self):
        """Load configuration from file"""
        if not self.config_path or not self.config_path.exists():
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif self.config_path.suffix == '.json':
                    file_config = json.load(f)
                else:
                    self.logger.warning(f"Unsupported config file format: {self.config_path.suffix}")
                    return
            
            self._merge_config(file_config, ConfigSource.FILE)
            self.logger.info(f"Loaded config from file: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config file: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_config = {}
        
        for key in self.schema.keys():
            # Convert key to environment variable name
            env_key = key.upper().replace('.', '_')
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                # Convert to appropriate type
                schema = self.schema[key]
                try:
                    if schema.type == bool:
                        env_config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif schema.type == int:
                        env_config[key] = int(env_value)
                    elif schema.type == float:
                        env_config[key] = float(env_value)
                    else:
                        env_config[key] = env_value
                except ValueError as e:
                    self.logger.warning(f"Invalid environment variable {env_key}: {e}")
                    continue
        
        self._merge_config(env_config, ConfigSource.ENVIRONMENT)
    
    def _merge_config(self, new_config: Dict[str, Any], source: ConfigSource):
        """Merge new configuration data"""
        with self.lock:
            for key, value in new_config.items():
                if key in self.schema:
                    # Validate value
                    if self._validate_value(key, value):
                        self.config_data[key] = value
                        self.sources[key] = source
                    else:
                        self.logger.warning(f"Invalid value for {key}: {value}")
    
    def _validate_value(self, key: str, value: Any) -> bool:
        """Validate configuration value"""
        schema = self.schema.get(key)
        if not schema:
            return False
        
        # Type validation
        if not isinstance(value, schema.type):
            try:
                # Try to convert
                if schema.type == bool:
                    value = str(value).lower() in ('true', '1', 'yes', 'on')
                else:
                    value = schema.type(value)
            except (ValueError, TypeError):
                return False
        
        # Custom validation
        if schema.validation and not schema.validation(value):
            return False
        
        return True
    
    def _setup_file_watcher(self):
        """Setup file system watcher for hot reloading"""
        if not self.config_path:
            return
        
        try:
            observer = Observer()
            handler = ConfigFileHandler(self)
            observer.schedule(handler, str(self.config_path.parent), recursive=False)
            observer.start()
            self.observers.append(observer)
            self.logger.info("Config file watcher started")
        except Exception as e:
            self.logger.warning(f"Failed to setup file watcher: {e}")
    
    def reload_config(self):
        """Reload configuration from sources"""
        self.logger.info("Reloading configuration...")
        
        # Clear current config
        with self.lock:
            self.config_data.clear()
            self.sources.clear()
        
        # Reload
        self._load_defaults()
        if self.config_path:
            self._load_config_file()
        self._load_environment_variables()
        
        self.logger.info("Configuration reloaded")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        with self.lock:
            return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.OVERRIDE):
        """Set configuration value"""
        if key not in self.schema:
            self.logger.warning(f"Unknown configuration key: {key}")
            return
        
        if not self._validate_value(key, value):
            self.logger.error(f"Invalid value for {key}: {value}")
            return
        
        with self.lock:
            self.config_data[key] = value
            self.sources[key] = source
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section"""
        with self.lock:
            section_data = {}
            for key, value in self.config_data.items():
                if key.startswith(f"{section}."):
                    section_key = key[len(f"{section}."):]
                    section_data[section_key] = value
            return section_data
    
    def get_source(self, key: str) -> Optional[ConfigSource]:
        """Get configuration source for key"""
        return self.sources.get(key)
    
    def validate_config(self) -> List[str]:
        """Validate entire configuration"""
        errors = []
        
        with self.lock:
            for key, schema in self.schema.items():
                if schema.required and key not in self.config_data:
                    errors.append(f"Required configuration missing: {key}")
                elif key in self.config_data:
                    if not self._validate_value(key, self.config_data[key]):
                        errors.append(f"Invalid configuration value: {key}")
        
        return errors
    
    def export_config(self, format: str = 'yaml') -> str:
        """Export configuration to string"""
        with self.lock:
            if format.lower() == 'yaml':
                return yaml.dump(self.config_data, default_flow_style=False)
            elif format.lower() == 'json':
                return json.dumps(self.config_data, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def save_config(self, path: str, format: str = 'yaml'):
        """Save configuration to file"""
        config_str = self.export_config(format)
        
        with open(path, 'w') as f:
            f.write(config_str)
        
        self.logger.info(f"Configuration saved to: {path}")
    
    def add_schema(self, schema: ConfigSchema):
        """Add new configuration schema"""
        with self.lock:
            self.schema[schema.key] = schema
            
            # Set default value if provided
            if schema.default is not None:
                self.config_data[schema.key] = schema.default
                self.sources[schema.key] = ConfigSource.DEFAULTS
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information"""
        with self.lock:
            return {
                'total_keys': len(self.config_data),
                'schema_keys': len(self.schema),
                'sources': {key: source.value for key, source in self.sources.items()},
                'validation_errors': self.validate_config()
            }
    
    def cleanup(self):
        """Cleanup resources"""
        for observer in self.observers:
            observer.stop()
            observer.join()
        self.observers.clear()



