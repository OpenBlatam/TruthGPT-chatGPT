"""
Environment Configuration for TruthGPT Optimization Core
Handles different environment configurations (development, production, testing)
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    STAGING = "staging"

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class EnvironmentConfig:
    """Base environment configuration."""
    
    # Environment identification
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    max_gpu_count: int = 1
    
    # Security
    enable_encryption: bool = False
    enable_authentication: bool = False
    secret_key: Optional[str] = None
    
    def __post_init__(self):
        """Initialize environment-specific settings."""
        self._setup_environment()
        self._create_directories()
    
    def _setup_environment(self) -> None:
        """Setup environment-specific configurations."""
        if self.environment == Environment.DEVELOPMENT:
            self._setup_development()
        elif self.environment == Environment.PRODUCTION:
            self._setup_production()
        elif self.environment == Environment.TESTING:
            self._setup_testing()
        elif self.environment == Environment.STAGING:
            self._setup_staging()
    
    def _setup_development(self) -> None:
        """Setup development environment."""
        self.debug = True
        self.log_level = LogLevel.DEBUG
        self.max_memory_gb = 4.0
        self.max_cpu_cores = 2
        self.max_gpu_count = 0
        self.enable_encryption = False
        self.enable_authentication = False
    
    def _setup_production(self) -> None:
        """Setup production environment."""
        self.debug = False
        self.log_level = LogLevel.INFO
        self.max_memory_gb = 32.0
        self.max_cpu_cores = 16
        self.max_gpu_count = 4
        self.enable_encryption = True
        self.enable_authentication = True
        self.secret_key = os.getenv("TRUTHGPT_SECRET_KEY")
    
    def _setup_testing(self) -> None:
        """Setup testing environment."""
        self.debug = True
        self.log_level = LogLevel.DEBUG
        self.max_memory_gb = 2.0
        self.max_cpu_cores = 1
        self.max_gpu_count = 0
        self.enable_encryption = False
        self.enable_authentication = False
    
    def _setup_staging(self) -> None:
        """Setup staging environment."""
        self.debug = False
        self.log_level = LogLevel.INFO
        self.max_memory_gb = 16.0
        self.max_cpu_cores = 8
        self.max_gpu_count = 2
        self.enable_encryption = True
        self.enable_authentication = True
        self.secret_key = os.getenv("TRUTHGPT_SECRET_KEY")
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.model_dir,
            self.cache_dir,
            self.output_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment.value,
            'debug': self.debug,
            'log_level': self.log_level.value,
            'log_file': self.log_file,
            'log_format': self.log_format,
            'data_dir': self.data_dir,
            'model_dir': self.model_dir,
            'cache_dir': self.cache_dir,
            'output_dir': self.output_dir,
            'max_memory_gb': self.max_memory_gb,
            'max_cpu_cores': self.max_cpu_cores,
            'max_gpu_count': self.max_gpu_count,
            'enable_encryption': self.enable_encryption,
            'enable_authentication': self.enable_authentication,
            'secret_key': self.secret_key
        }

@dataclass
class DevelopmentConfig(EnvironmentConfig):
    """Development environment configuration."""
    
    def __post_init__(self):
        """Initialize development configuration."""
        self.environment = Environment.DEVELOPMENT
        super().__post_init__()
        
        # Development-specific settings
        self.log_level = LogLevel.DEBUG
        self.debug = True
        self.max_memory_gb = 4.0
        self.max_cpu_cores = 2
        self.max_gpu_count = 0

@dataclass
class ProductionConfig(EnvironmentConfig):
    """Production environment configuration."""
    
    # Production-specific settings
    enable_monitoring: bool = True
    enable_metrics: bool = True
    enable_alerting: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    max_batch_size: int = 32
    
    def __post_init__(self):
        """Initialize production configuration."""
        self.environment = Environment.PRODUCTION
        super().__post_init__()
        
        # Production-specific settings
        self.log_level = LogLevel.INFO
        self.debug = False
        self.max_memory_gb = 32.0
        self.max_cpu_cores = 16
        self.max_gpu_count = 4
        self.enable_encryption = True
        self.enable_authentication = True

@dataclass
class TestingConfig(EnvironmentConfig):
    """Testing environment configuration."""
    
    # Testing-specific settings
    use_mock_data: bool = True
    fast_tests: bool = True
    parallel_tests: bool = True
    
    def __post_init__(self):
        """Initialize testing configuration."""
        self.environment = Environment.TESTING
        super().__post_init__()
        
        # Testing-specific settings
        self.log_level = LogLevel.DEBUG
        self.debug = True
        self.max_memory_gb = 2.0
        self.max_cpu_cores = 1
        self.max_gpu_count = 0
        self.enable_encryption = False
        self.enable_authentication = False

# Factory functions
def create_environment_config(
    environment: Optional[Environment] = None,
    **kwargs
) -> EnvironmentConfig:
    """Create environment configuration based on environment type."""
    if environment is None:
        environment = Environment(os.getenv("TRUTHGPT_ENVIRONMENT", "development"))
    
    if environment == Environment.DEVELOPMENT:
        return DevelopmentConfig(**kwargs)
    elif environment == Environment.PRODUCTION:
        return ProductionConfig(**kwargs)
    elif environment == Environment.TESTING:
        return TestingConfig(**kwargs)
    else:
        return EnvironmentConfig(environment=environment, **kwargs)

def load_environment_from_env() -> EnvironmentConfig:
    """Load environment configuration from environment variables."""
    env_vars = {
        'environment': os.getenv("TRUTHGPT_ENVIRONMENT", "development"),
        'debug': os.getenv("TRUTHGPT_DEBUG", "false").lower() == "true",
        'log_level': os.getenv("TRUTHGPT_LOG_LEVEL", "INFO"),
        'log_file': os.getenv("TRUTHGPT_LOG_FILE"),
        'data_dir': os.getenv("TRUTHGPT_DATA_DIR", "./data"),
        'model_dir': os.getenv("TRUTHGPT_MODEL_DIR", "./models"),
        'cache_dir': os.getenv("TRUTHGPT_CACHE_DIR", "./cache"),
        'output_dir': os.getenv("TRUTHGPT_OUTPUT_DIR", "./outputs"),
        'max_memory_gb': float(os.getenv("TRUTHGPT_MAX_MEMORY_GB", "8.0")),
        'max_cpu_cores': int(os.getenv("TRUTHGPT_MAX_CPU_CORES", "4")),
        'max_gpu_count': int(os.getenv("TRUTHGPT_MAX_GPU_COUNT", "1")),
        'enable_encryption': os.getenv("TRUTHGPT_ENABLE_ENCRYPTION", "false").lower() == "true",
        'enable_authentication': os.getenv("TRUTHGPT_ENABLE_AUTHENTICATION", "false").lower() == "true",
        'secret_key': os.getenv("TRUTHGPT_SECRET_KEY")
    }
    
    # Filter out None values
    env_vars = {k: v for k, v in env_vars.items() if v is not None}
    
    return create_environment_config(**env_vars)





