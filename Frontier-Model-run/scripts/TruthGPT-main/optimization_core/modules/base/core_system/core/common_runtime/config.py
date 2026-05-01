"""
Configuration management - Refactored configuration system
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import threading
import time
from contextlib import contextmanager

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
class OptimizationConfig:
    """Optimization configuration."""
    level: str = "standard"
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_mixed_precision: bool = True
    enable_kernel_fusion: bool = True
    max_memory_gb: float = 16.0
    max_cpu_cores: int = 8
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level,
            'enable_quantization': self.enable_quantization,
            'enable_pruning': self.enable_pruning,
            'enable_mixed_precision': self.enable_mixed_precision,
            'enable_kernel_fusion': self.enable_kernel_fusion,
            'max_memory_gb': self.max_memory_gb,
            'max_cpu_cores': self.max_cpu_cores,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'gpu_memory_fraction': self.gpu_memory_fraction
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_profiling: bool = True
    profiling_interval: int = 100
    log_level: str = "INFO"
    enable_metrics_collection: bool = True
    metrics_retention_days: int = 30
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    gpu_memory_threshold: float = 90.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_profiling': self.enable_profiling,
            'profiling_interval': self.profiling_interval,
            'log_level': self.log_level,
            'enable_metrics_collection': self.enable_metrics_collection,
            'metrics_retention_days': self.metrics_retention_days,
            'cpu_threshold': self.cpu_threshold,
            'memory_threshold': self.memory_threshold,
            'gpu_memory_threshold': self.gpu_memory_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    batch_size: int = 32
    max_workers: int = 4
    enable_async_processing: bool = True
    enable_parallel_optimization: bool = True
    optimization_timeout: float = 300.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'enable_async_processing': self.enable_async_processing,
            'enable_parallel_optimization': self.enable_parallel_optimization,
            'optimization_timeout': self.optimization_timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config_data: Dict[str, Any] = {}
        self.config_lock = threading.RLock()
        self.update_callbacks: List[callable] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize with defaults
        self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration."""
        self.config_data = {
            'optimization': OptimizationConfig().to_dict(),
            'monitoring': MonitoringConfig().to_dict(),
            'performance': PerformanceConfig().to_dict()
        }
    
    def load_from_file(self, filepath: str) -> bool:
        """Load configuration from file."""
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                self.logger.warning(f"Config file {filepath} not found")
                return False
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    self.logger.error(f"Unsupported config file format: {file_path.suffix}")
                    return False
            
            with self.config_lock:
                self._merge_config(config)
            
            self.logger.info(f"Configuration loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {filepath}: {e}")
            return False
    
    def load_from_environment(self, prefix: str = "OPTIMIZATION_") -> bool:
        """Load configuration from environment variables."""
        try:
            env_config = {}
            
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower()
                    
                    # Handle nested keys
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
                with self.config_lock:
                    self._merge_config(env_config)
                
                self.logger.info("Configuration loaded from environment variables")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load config from environment: {e}")
            return False
    
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
        def deep_merge(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(self.config_data, new_config)
        self._notify_update_callbacks()
    
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
            self._notify_update_callbacks()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]):
        """Update entire configuration section."""
        with self.config_lock:
            if section not in self.config_data:
                self.config_data[section] = {}
            
            self._merge_config({section: updates})
    
    def get_optimization_config(self) -> OptimizationConfig:
        """Get optimization configuration."""
        data = self.get_section('optimization')
        return OptimizationConfig.from_dict(data)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        data = self.get_section('monitoring')
        return MonitoringConfig.from_dict(data)
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        data = self.get_section('performance')
        return PerformanceConfig.from_dict(data)
    
    def add_update_callback(self, callback: callable):
        """Add callback for configuration updates."""
        self.update_callbacks.append(callback)
    
    def _notify_update_callbacks(self):
        """Notify all update callbacks."""
        for callback in self.update_callbacks:
            try:
                callback(self.config_data)
            except Exception as e:
                self.logger.error(f"Error in update callback: {e}")
    
    def export_config(self, filepath: str, format: str = 'json') -> bool:
        """Export current configuration to file."""
        try:
            with open(filepath, 'w') as f:
                if format.lower() == 'json':
                    json.dump(self.config_data, f, indent=2)
                elif format.lower() in ['yaml', 'yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False)
                else:
                    self.logger.error(f"Unsupported export format: {format}")
                    return False
            
            self.logger.info(f"Configuration exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate current configuration."""
        errors = []
        
        # Validate optimization config
        opt_config = self.get_optimization_config()
        if opt_config.max_memory_gb <= 0:
            errors.append("max_memory_gb must be positive")
        if opt_config.max_cpu_cores <= 0:
            errors.append("max_cpu_cores must be positive")
        if not 0 < opt_config.gpu_memory_fraction <= 1:
            errors.append("gpu_memory_fraction must be between 0 and 1")
        
        # Validate monitoring config
        mon_config = self.get_monitoring_config()
        if mon_config.profiling_interval <= 0:
            errors.append("profiling_interval must be positive")
        if mon_config.cpu_threshold <= 0 or mon_config.cpu_threshold > 100:
            errors.append("cpu_threshold must be between 0 and 100")
        
        # Validate performance config
        perf_config = self.get_performance_config()
        if perf_config.batch_size <= 0:
            errors.append("batch_size must be positive")
        if perf_config.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        return errors

# Factory functions
def create_config_manager(environment: Environment = Environment.DEVELOPMENT) -> ConfigManager:
    """Create a configuration manager."""
    return ConfigManager(environment)

@contextmanager
def config_context(environment: Environment = Environment.DEVELOPMENT):
    """Context manager for configuration."""
    manager = create_config_manager(environment)
    try:
        yield manager
    finally:
        # Cleanup if needed
        pass

