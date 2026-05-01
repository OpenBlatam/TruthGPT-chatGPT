"""
Deployment utilities for optimization_core.

Provides utilities for deployment and environment management.
"""
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EnvironmentConfig(BaseModel):
    """Configuration for deployment environment."""
    name: str
    api_url: Optional[str] = None
    model_path: Optional[str] = None
    cache_dir: Optional[str] = None
    log_level: str = "INFO"
    debug: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeploymentManager:
    """Manager for deployment configurations."""
    
    def __init__(self):
        """Initialize deployment manager."""
        self.environments: Dict[str, EnvironmentConfig] = {}
    
    def register_environment(
        self,
        config: EnvironmentConfig
    ):
        """
        Register an environment.
        
        Args:
            config: Environment configuration
        """
        self.environments[config.name] = config
        logger.info(f"Registered environment: {config.name}")
    
    def get_environment(
        self,
        name: Optional[str] = None
    ) -> Optional[EnvironmentConfig]:
        """
        Get environment configuration.
        
        Args:
            name: Environment name (defaults to ENV environment variable)
        
        Returns:
            Environment configuration
        """
        if name is None:
            name = os.getenv("ENV", "development")
        
        return self.environments.get(name)
    
    def load_from_env(self) -> EnvironmentConfig:
        """
        Load configuration from environment variables.
        
        Returns:
            Environment configuration
        """
        return EnvironmentConfig(
            name=os.getenv("ENV", "development"),
            api_url=os.getenv("API_URL"),
            model_path=os.getenv("MODEL_PATH"),
            cache_dir=os.getenv("CACHE_DIR", "/tmp/cache"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
    
    def setup_environment(
        self,
        env_name: Optional[str] = None
    ) -> EnvironmentConfig:
        """
        Setup environment for deployment.
        
        Args:
            env_name: Environment name
        
        Returns:
            Environment configuration
        """
        config = self.get_environment(env_name) or self.load_from_env()
        
        # Setup logging
        import logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create cache directory
        if config.cache_dir:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Environment '{config.name}' setup complete")
        return config


def get_deployment_config(
    env_name: Optional[str] = None
) -> EnvironmentConfig:
    """
    Get deployment configuration.
    
    Args:
        env_name: Environment name
    
    Returns:
        Environment configuration
    """
    manager = DeploymentManager()
    return manager.setup_environment(env_name)













