"""
Configuration management for polyglot_core.

Provides centralized configuration with YAML support, environment-specific
settings, and validation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import json
import yaml
import os


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class PolyglotConfig:
    """Main configuration for polyglot_core."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    
    # Backend preferences
    preferred_backends: Dict[str, str] = field(default_factory=lambda: {
        'kv_cache': 'rust',
        'attention': 'cpp',
        'compression': 'rust',
        'inference': 'cpp',
        'tokenization': 'rust',
        'quantization': 'cpp'
    })
    
    # Performance settings
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'enable_profiling': False,
        'enable_metrics': True,
        'enable_benchmarking': False,
        'profiling_interval': 1.0,
        'metrics_retention_hours': 24
    })
    
    # Cache settings
    cache: Dict[str, Any] = field(default_factory=lambda: {
        'default_max_size': 100000,
        'default_memory_gb': 8,
        'enable_compression': True,
        'compression_threshold': 4096
    })
    
    # Attention settings
    attention: Dict[str, Any] = field(default_factory=lambda: {
        'default_d_model': 768,
        'default_n_heads': 12,
        'use_flash': True,
        'block_size': 64
    })
    
    # Compression settings
    compression: Dict[str, Any] = field(default_factory=lambda: {
        'default_algorithm': 'lz4',
        'default_level': 3,
        'chunk_size': 65536
    })
    
    # Inference settings
    inference: Dict[str, Any] = field(default_factory=lambda: {
        'default_seed': 42,
        'default_max_tokens': 100,
        'default_temperature': 1.0,
        'default_top_p': 0.9
    })
    
    # Resource limits
    resources: Dict[str, Any] = field(default_factory=lambda: {
        'max_memory_gb': 32,
        'max_cache_size_gb': 16,
        'max_concurrent_operations': 10
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['environment'] = self.environment.value
        return result
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(filepath)
        with open(path, 'w') as f:
            f.write(self.to_yaml())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolyglotConfig":
        """Create from dictionary."""
        if 'environment' in data:
            data['environment'] = Environment(data['environment'])
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "PolyglotConfig":
        """Load from YAML file."""
        path = Path(filepath)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls) -> "PolyglotConfig":
        """Load from environment variables."""
        config = cls()
        
        # Environment
        env_str = os.getenv('POLYGLOT_ENV', 'development')
        try:
            config.environment = Environment(env_str.lower())
        except ValueError:
            config.environment = Environment.DEVELOPMENT
        
        config.debug = os.getenv('POLYGLOT_DEBUG', 'false').lower() == 'true'
        config.log_level = os.getenv('POLYGLOT_LOG_LEVEL', 'INFO')
        
        # Performance
        config.performance['enable_profiling'] = (
            os.getenv('POLYGLOT_ENABLE_PROFILING', 'false').lower() == 'true'
        )
        config.performance['enable_metrics'] = (
            os.getenv('POLYGLOT_ENABLE_METRICS', 'true').lower() == 'true'
        )
        
        # Resources
        if 'POLYGLOT_MAX_MEMORY_GB' in os.environ:
            config.resources['max_memory_gb'] = int(os.getenv('POLYGLOT_MAX_MEMORY_GB'))
        
        return config
    
    @classmethod
    def default(cls) -> "PolyglotConfig":
        """Get default configuration."""
        return cls()
    
    @classmethod
    def production(cls) -> "PolyglotConfig":
        """Get production configuration."""
        config = cls()
        config.environment = Environment.PRODUCTION
        config.debug = False
        config.log_level = "WARNING"
        config.performance['enable_profiling'] = True
        config.performance['enable_metrics'] = True
        config.resources['max_memory_gb'] = 64
        return config
    
    @classmethod
    def development(cls) -> "PolyglotConfig":
        """Get development configuration."""
        config = cls()
        config.environment = Environment.DEVELOPMENT
        config.debug = True
        config.log_level = "DEBUG"
        config.performance['enable_profiling'] = True
        config.performance['enable_benchmarking'] = True
        return config


class ConfigManager:
    """
    Manages polyglot_core configuration.
    
    Supports loading from YAML files, environment variables, and defaults.
    
    Example:
        >>> manager = ConfigManager()
        >>> config = manager.load_config("config.yaml")
        >>> cache = KVCache(max_size=config.cache['default_max_size'])
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory for config files (default: ~/.polyglot_core)
        """
        if config_dir is None:
            config_dir = Path.home() / ".polyglot_core"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._current_config: Optional[PolyglotConfig] = None
    
    def load_config(
        self,
        filepath: Optional[Union[str, Path]] = None,
        environment: Optional[Environment] = None
    ) -> PolyglotConfig:
        """
        Load configuration.
        
        Args:
            filepath: Path to config file (optional)
            environment: Override environment
            
        Returns:
            PolyglotConfig
        """
        if filepath:
            config = PolyglotConfig.from_yaml(filepath)
        else:
            # Try to load from default location
            default_config = self.config_dir / "config.yaml"
            if default_config.exists():
                config = PolyglotConfig.from_yaml(default_config)
            else:
                # Load from environment or use defaults
                config = PolyglotConfig.from_env()
        
        if environment:
            config.environment = environment
        
        self._current_config = config
        return config
    
    def save_config(self, config: PolyglotConfig, filepath: Optional[Union[str, Path]] = None):
        """
        Save configuration.
        
        Args:
            config: Configuration to save
            filepath: Output path (default: config_dir/config.yaml)
        """
        if filepath is None:
            filepath = self.config_dir / "config.yaml"
        
        config.save(filepath)
        self._current_config = config
    
    def get_config(self) -> Optional[PolyglotConfig]:
        """Get current configuration."""
        if self._current_config is None:
            self.load_config()
        return self._current_config
    
    def update_config(self, **kwargs):
        """Update current configuration."""
        if self._current_config is None:
            self._current_config = PolyglotConfig()
        
        for key, value in kwargs.items():
            if hasattr(self._current_config, key):
                setattr(self._current_config, key, value)
    
    def reset_to_defaults(self):
        """Reset to default configuration."""
        self._current_config = PolyglotConfig.default()


# Global config manager
_global_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get global config manager."""
    return _global_config_manager


def get_config() -> PolyglotConfig:
    """Get current configuration."""
    return _global_config_manager.get_config() or PolyglotConfig.default()


def load_config(filepath: Optional[Union[str, Path]] = None) -> PolyglotConfig:
    """Load configuration from file."""
    return _global_config_manager.load_config(filepath)


def save_config(config: PolyglotConfig, filepath: Optional[Union[str, Path]] = None):
    """Save configuration to file."""
    _global_config_manager.save_config(config, filepath)













