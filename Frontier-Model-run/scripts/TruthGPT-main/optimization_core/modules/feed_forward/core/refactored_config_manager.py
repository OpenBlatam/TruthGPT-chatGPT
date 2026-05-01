"""
Refactored Configuration Management System
==========================================

Advanced configuration management with validation, hot-reloading, and environment-specific configs.
This module provides a robust system for managing application configuration from multiple sources.
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import yaml

# Configure logging
logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


@dataclass
class ConfigValidationRule:
    """
    Configuration validation rule.

    Attributes:
        field: Name of the field to validate.
        validator: Function that returns True if value is valid.
        error_message: Message to display on validation failure.
        required: Whether the field is mandatory.
    """
    field: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True


@dataclass
class ConfigSourceInfo:
    """
    Configuration source information.

    Attributes:
        source: Type of configuration source.
        path: Path to file (if applicable).
        format: File format (if applicable).
        priority: Priority of the source (higher wins).
        hot_reload: Whether to monitor for changes.
        checksum: Last known checksum of the source.
        last_modified: Last modified timestamp.
    """
    source: ConfigSource
    path: Optional[str] = None
    format: ConfigFormat = ConfigFormat.JSON
    priority: int = 0
    hot_reload: bool = False
    checksum: Optional[str] = None
    last_modified: Optional[float] = None


class ConfigurationManager:
    """
    Advanced configuration manager with validation and hot-reloading.

    Manages calling multiple configuration sources, merging them by priority,
    validating the result, and monitoring for changes.
    """

    def __init__(self, base_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the ConfigurationManager.

        Args:
            base_config: Initial default configuration.
        """
        self.base_config: Dict[str, Any] = base_config or {}
        self.config: Dict[str, Any] = self.base_config.copy()
        self.sources: List[ConfigSourceInfo] = []
        self.validation_rules: List[ConfigValidationRule] = []
        self.observers: List[Callable[[str, Any], None]] = []
        self.hot_reload_thread: Optional[threading.Thread] = None
        self._stop_hot_reload: bool = False
        self._lock: threading.RLock = threading.RLock()
        self.logger: logging.Logger = logging.getLogger('config_manager')

        # Setup default validation rules
        self._setup_default_validation_rules()

    def _setup_default_validation_rules(self) -> None:
        """Setup default validation rules."""
        # Add common validation rules
        self.add_validation_rule(
            ConfigValidationRule(
                field="production_mode",
                validator=lambda x: x in ["development", "staging", "production", "high_performance", "cost_optimized"],
                error_message="Invalid production mode",
                required=True
            )
        )

        self.add_validation_rule(
            ConfigValidationRule(
                field="hidden_size",
                validator=lambda x: isinstance(x, int) and x > 0,
                error_message="Hidden size must be a positive integer",
                required=True
            )
        )

        self.add_validation_rule(
            ConfigValidationRule(
                field="num_experts",
                validator=lambda x: isinstance(x, int) and x > 0,
                error_message="Number of experts must be a positive integer",
                required=True
            )
        )

    def add_source(self, source_info: ConfigSourceInfo) -> None:
        """
        Add a configuration source.

        Args:
            source_info: Information about the source to add.
        """
        with self._lock:
            self.sources.append(source_info)
            self.sources.sort(key=lambda x: x.priority, reverse=True)

    def add_validation_rule(self, rule: ConfigValidationRule) -> None:
        """
        Add a validation rule.

        Args:
            rule: Validation rule to add.
        """
        self.validation_rules.append(rule)

    def add_observer(self, observer: Callable[[str, Any], None]) -> None:
        """
        Add a configuration change observer.

        Args:
            observer: Function to call when configuration changes.
        """
        self.observers.append(observer)

    def load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from all sources.

        Returns:
            The loaded and merged configuration dictionary.
        """
        with self._lock:
            # Start with base config
            self.config = self.base_config.copy()

            # Load from each source
            for source_info in self.sources:
                try:
                    source_config = self._load_from_source(source_info)
                    if source_config:
                        self.config.update(source_config)
                        self.logger.info(f"Loaded config from {source_info.source.value}")
                except Exception as e:
                    self.logger.error(f"Failed to load config from {source_info.source.value}: {e}")

            # Validate configuration
            self._validate_configuration()

            # Start hot reload if enabled
            self._start_hot_reload()

            return self.config.copy()

    def _load_from_source(self, source_info: ConfigSourceInfo) -> Dict[str, Any]:
        """
        Load configuration from a specific source.

        Args:
            source_info: Source information.

        Returns:
            Dictionary containing configuration from the source.

        Raises:
            ValueError: If the source type is unsupported.
        """
        if source_info.source == ConfigSource.FILE:
            return self._load_from_file(source_info)
        elif source_info.source == ConfigSource.ENVIRONMENT:
            return self._load_from_environment()
        elif source_info.source == ConfigSource.MEMORY:
            return self._load_from_memory()
        else:
            raise ValueError(f"Unsupported config source: {source_info.source}")

    def _load_from_file(self, source_info: ConfigSourceInfo) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            source_info: File source information.

        Returns:
            Configuration dictionary.

        Raises:
            ValueError: If file format is unsupported.
        """
        if not source_info.path or not os.path.exists(source_info.path):
            return {}

        # Check if file has changed
        current_checksum = self._calculate_file_checksum(source_info.path)
        if source_info.checksum == current_checksum:
            return {}

        # Update source info
        source_info.checksum = current_checksum
        source_info.last_modified = os.path.getmtime(source_info.path)

        # Load file based on format
        with open(source_info.path, 'r', encoding='utf-8') as f:
            if source_info.format == ConfigFormat.JSON:
                return json.load(f)
            elif source_info.format == ConfigFormat.YAML:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {source_info.format}")

    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Returns:
            Configuration dictionary.
        """
        config: Dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith('PIMOE_'):
                # Remove prefix and convert to nested dict
                config_key = key[6:].lower().replace('_', '.')
                self._set_nested_value(config, config_key, value)
        return config

    def _load_from_memory(self) -> Dict[str, Any]:
        """
        Load configuration from memory.

        Returns:
            Empty dictionary (placeholder).
        """
        return {}

    def _calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate file checksum for change detection.

        Args:
            file_path: Path to the file.

        Returns:
            MD5 checksum string.
        """
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """
        Set nested dictionary value from dot notation.

        Args:
            config: Target dictionary.
            key: Dot-separated key path.
            value: Value to set.
        """
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def _validate_configuration(self) -> None:
        """
        Validate configuration against rules.

        Raises:
            ValueError: If validation fails.
        """
        for rule in self.validation_rules:
            if rule.required and rule.field not in self.config:
                # Only raise if required and missing
                # But if default exists elsewhere?
                # Assuming config is complete if validations run.
                # However, if partially loaded, this might fail.
                # We assume base_config provides defaults.
                pass
                # The original code raised error. I will keep it consistent.
                # raise ValueError(f"Required field '{rule.field}' not found in configuration")
            
            # Since self.config starts with base_config, commonly required fields should be present.
            if rule.field in self.config:
                if not rule.validator(self.config[rule.field]):
                    raise ValueError(f"Validation failed for '{rule.field}': {rule.error_message}")
            elif rule.required:
                 raise ValueError(f"Required field '{rule.field}' not found in configuration")

    def _start_hot_reload(self) -> None:
        """Start hot reload monitoring."""
        if any(source.hot_reload for source in self.sources):
            self._stop_hot_reload = False
            self.hot_reload_thread = threading.Thread(target=self._hot_reload_loop, daemon=True)
            self.hot_reload_thread.start()

    def _hot_reload_loop(self) -> None:
        """Hot reload monitoring loop."""
        while not self._stop_hot_reload:
            try:
                for source_info in self.sources:
                    if source_info.hot_reload and source_info.source == ConfigSource.FILE:
                        if self._check_file_changed(source_info):
                            self.logger.info(f"File {source_info.path} changed, reloading config")
                            self.load_configuration()
                            break

                time.sleep(1.0)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in hot reload loop: {e}")
                time.sleep(5.0)

    def _check_file_changed(self, source_info: ConfigSourceInfo) -> bool:
        """
        Check if file has changed.

        Args:
            source_info: File source information.

        Returns:
            True if file changed, False otherwise.
        """
        if not source_info.path or not os.path.exists(source_info.path):
            return False

        current_checksum = self._calculate_file_checksum(source_info.path)
        return current_checksum != source_info.checksum

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Dot-separated key path.
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        with self._lock:
            return self._get_nested_value(self.config, key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Dot-separated key path.
            value: Value to set.
        """
        with self._lock:
            self._set_nested_value(self.config, key, value)
            self._notify_observers(key, value)

    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Get nested dictionary value from dot notation.

        Args:
            config: Source dictionary.
            key: Dot-separated key path.
            default: Default value.

        Returns:
            Found value or default.
        """
        keys = key.split('.')
        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def _notify_observers(self, key: str, value: Any) -> None:
        """
        Notify observers of configuration changes.

        Args:
            key: Changed key.
            value: New value.
        """
        for observer in self.observers:
            try:
                observer(key, value)
            except Exception as e:
                self.logger.error(f"Error in config observer: {e}")

    def save_configuration(self, file_path: str, format: ConfigFormat = ConfigFormat.JSON) -> None:
        """
        Save current configuration to file.

        Args:
            file_path: Output file path.
            format: Output format.

        Raises:
            ValueError: If format is unsupported.
        """
        with self._lock:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    json.dump(self.config, f, indent=2)
                elif format == ConfigFormat.YAML:
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported save format: {format}")

    def stop_hot_reload(self) -> None:
        """Stop hot reload monitoring."""
        self._stop_hot_reload = True
        if self.hot_reload_thread:
            self.hot_reload_thread.join()

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.

        Returns:
            Summary dictionary.
        """
        return {
            'sources': [asdict(source) for source in self.sources],
            'validation_rules': len(self.validation_rules),
            'observers': len(self.observers),
            'hot_reload_enabled': any(source.hot_reload for source in self.sources),
            'config_keys': list(self.config.keys())
        }


class EnvironmentConfigBuilder:
    """Builder for environment-specific configurations."""

    def __init__(self) -> None:
        """Initialize EnvironmentConfigBuilder."""
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.current_environment: Optional[str] = None

    def for_environment(self, environment: str) -> 'EnvironmentConfigBuilder':
        """
        Set current environment.

        Args:
            environment: Environment name.

        Returns:
            Self for chaining.
        """
        self.current_environment = environment
        if environment not in self.configs:
            self.configs[environment] = {}
        return self

    def with_base_config(self, config: Dict[str, Any]) -> 'EnvironmentConfigBuilder':
        """
        Set base configuration.

        Args:
            config: Base configuration dict.

        Returns:
            Self for chaining.
        """
        if self.current_environment:
            self.configs[self.current_environment].update(config)
        return self

    def with_override(self, key: str, value: Any) -> 'EnvironmentConfigBuilder':
        """
        Add configuration override.

        Args:
            key: Config key.
            value: Config value.

        Returns:
            Self for chaining.
        """
        if self.current_environment:
            self.configs[self.current_environment][key] = value
        return self

    def with_file_source(self, file_path: str, priority: int = 0) -> 'EnvironmentConfigBuilder':
        """
        Add file source.

        Args:
            file_path: Path to config file.
            priority: Source priority.

        Returns:
            Self for chaining.
        """
        if self.current_environment:
            if 'sources' not in self.configs[self.current_environment]:
                self.configs[self.current_environment]['sources'] = []
            self.configs[self.current_environment]['sources'].append({
                'source': ConfigSource.FILE,
                'path': file_path,
                'priority': priority
            })
        return self

    def with_environment_source(self, priority: int = 0) -> 'EnvironmentConfigBuilder':
        """
        Add environment variable source.

        Args:
            priority: Source priority.

        Returns:
            Self for chaining.
        """
        if self.current_environment:
            if 'sources' not in self.configs[self.current_environment]:
                self.configs[self.current_environment]['sources'] = []
            self.configs[self.current_environment]['sources'].append({
                'source': ConfigSource.ENVIRONMENT,
                'priority': priority
            })
        return self

    def build(self) -> Dict[str, Dict[str, Any]]:
        """
        Build environment configurations.

        Returns:
            Dictionary of environment configurations.
        """
        return self.configs.copy()


class ConfigurationFactory:
    """Factory for creating configuration managers."""

    @staticmethod
    def create_from_file(file_path: str, format: ConfigFormat = ConfigFormat.JSON) -> ConfigurationManager:
        """
        Create configuration manager from file.

        Args:
            file_path: Path to file.
            format: File format.

        Returns:
            New ConfigurationManager instance.
        """
        manager = ConfigurationManager()

        source_info = ConfigSourceInfo(
            source=ConfigSource.FILE,
            path=file_path,
            format=format,
            priority=1,
            hot_reload=True
        )

        manager.add_source(source_info)
        manager.load_configuration()

        return manager

    @staticmethod
    def create_from_environment() -> ConfigurationManager:
        """
        Create configuration manager from environment variables.

        Returns:
            New ConfigurationManager instance.
        """
        manager = ConfigurationManager()

        source_info = ConfigSourceInfo(
            source=ConfigSource.ENVIRONMENT,
            priority=1
        )

        manager.add_source(source_info)
        manager.load_configuration()

        return manager

    @staticmethod
    def create_multi_source(sources: List[ConfigSourceInfo]) -> ConfigurationManager:
        """
        Create configuration manager with multiple sources.

        Args:
            sources: List of source configurations.

        Returns:
            New ConfigurationManager instance.
        """
        manager = ConfigurationManager()

        for source_info in sources:
            manager.add_source(source_info)

        manager.load_configuration()
        return manager

    @staticmethod
    def create_for_environment(
        environment: str,
        configs: Dict[str, Dict[str, Any]]
    ) -> ConfigurationManager:
        """
        Create configuration manager for specific environment.

        Args:
            environment: Environment name.
            configs: Dictionary of environment configurations.

        Returns:
            New ConfigurationManager instance.

        Raises:
            ValueError: If environment not found.
        """
        if environment not in configs:
            raise ValueError(f"Environment '{environment}' not found in configurations")

        manager = ConfigurationManager(configs[environment])

        # Add sources if specified
        if 'sources' in configs[environment]:
            for source_config in configs[environment]['sources']:
                # Need to convert string enum to Enum
                if isinstance(source_config.get('source'), str):
                     try:
                         source_config['source'] = ConfigSource(source_config['source'])
                     except ValueError:
                         pass # Let it fail later or assume standard
                
                # Careful with kwargs unpacking if types mismatch, but dataclass usually handles it if fields match.
                # However, ConfigSourceInfo requires 'source' as ConfigSource enum.
                # Since we are building this inside EnvironmentConfigBuilder using Enums, it should be fine.
                # But if configs came from external JSON, we'd need parsing.
                # Assuming EnvironmentConfigBuilder usage, it's fine.
                
                source_info = ConfigSourceInfo(**source_config)
                manager.add_source(source_info)

        manager.load_configuration()
        return manager


class ConfigValidators:
    """Collection of common configuration validators."""

    @staticmethod
    def is_positive_int(value: Any) -> bool:
        """Check if value is a positive integer."""
        return isinstance(value, int) and value > 0

    @staticmethod
    def is_positive_float(value: Any) -> bool:
        """Check if value is a positive float."""
        return isinstance(value, (int, float)) and value > 0

    @staticmethod
    def is_in_range(value: Any, min_val: float, max_val: float) -> bool:
        """Check if value is in range."""
        return isinstance(value, (int, float)) and min_val <= value <= max_val

    @staticmethod
    def is_one_of(value: Any, options: List[Any]) -> bool:
        """Check if value is one of the options."""
        return value in options

    @staticmethod
    def is_boolean(value: Any) -> bool:
        """Check if value is boolean."""
        return isinstance(value, bool)

    @staticmethod
    def is_string(value: Any) -> bool:
        """Check if value is string."""
        return isinstance(value, str)

    @staticmethod
    def is_dict(value: Any) -> bool:
        """Check if value is dictionary."""
        return isinstance(value, dict)

    @staticmethod
    def is_list(value: Any) -> bool:
        """Check if value is list."""
        return isinstance(value, list)


class ConfigTemplates:
    """Predefined configuration templates."""

    @staticmethod
    def development_config() -> Dict[str, Any]:
        """Development environment configuration."""
        return {
            "production_mode": "development",
            "hidden_size": 256,
            "num_experts": 4,
            "max_batch_size": 8,
            "max_sequence_length": 512,
            "enable_monitoring": True,
            "enable_metrics": True,
            "log_level": "debug",
            "max_concurrent_requests": 10,
            "request_timeout": 30.0,
            "memory_threshold_mb": 2000.0,
            "cpu_threshold_percent": 70.0
        }

    @staticmethod
    def staging_config() -> Dict[str, Any]:
        """Staging environment configuration."""
        return {
            "production_mode": "staging",
            "hidden_size": 512,
            "num_experts": 8,
            "max_batch_size": 16,
            "max_sequence_length": 1024,
            "enable_monitoring": True,
            "enable_metrics": True,
            "log_level": "info",
            "max_concurrent_requests": 50,
            "request_timeout": 30.0,
            "memory_threshold_mb": 4000.0,
            "cpu_threshold_percent": 75.0
        }

    @staticmethod
    def production_config() -> Dict[str, Any]:
        """Production environment configuration."""
        return {
            "production_mode": "production",
            "hidden_size": 512,
            "num_experts": 8,
            "max_batch_size": 32,
            "max_sequence_length": 2048,
            "enable_monitoring": True,
            "enable_metrics": True,
            "log_level": "info",
            "max_concurrent_requests": 100,
            "request_timeout": 30.0,
            "memory_threshold_mb": 8000.0,
            "cpu_threshold_percent": 80.0
        }

    @staticmethod
    def high_performance_config() -> Dict[str, Any]:
        """High performance environment configuration."""
        return {
            "production_mode": "high_performance",
            "hidden_size": 1024,
            "num_experts": 16,
            "max_batch_size": 64,
            "max_sequence_length": 4096,
            "enable_monitoring": True,
            "enable_metrics": True,
            "log_level": "warning",
            "max_concurrent_requests": 200,
            "request_timeout": 60.0,
            "memory_threshold_mb": 16000.0,
            "cpu_threshold_percent": 90.0
        }

