"""
Migration utilities for optimization_core.

Provides utilities for migrating code and configurations between versions.
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, ConfigDict
from pathlib import Path

logger = logging.getLogger(__name__)


class Migration(BaseModel):
    """Migration definition."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    from_version: str
    to_version: str
    description: str
    migrate_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    
    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply migration to data.
        
        Args:
            data: Data to migrate
        
        Returns:
            Migrated data
        """
        logger.info(f"Applying migration: {self.description}")
        return self.migrate_func(data)


class MigrationManager:
    """Manager for handling migrations."""
    
    def __init__(self):
        """Initialize migration manager."""
        self.migrations: List[Migration] = []
    
    def register(
        self,
        from_version: str,
        to_version: str,
        description: str,
        migrate_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """
        Register a migration.
        
        Args:
            from_version: Source version
            to_version: Target version
            description: Migration description
            migrate_func: Migration function
        """
        migration = Migration(
            from_version=from_version,
            to_version=to_version,
            description=description,
            migrate_func=migrate_func
        )
        self.migrations.append(migration)
        logger.debug(f"Registered migration: {from_version} -> {to_version}")
    
    def find_migrations(
        self,
        from_version: str,
        to_version: str
    ) -> List[Migration]:
        """
        Find migrations needed to go from one version to another.
        
        Args:
            from_version: Source version
            to_version: Target version
        
        Returns:
            List of migrations to apply
        """
        # Simple implementation - in practice, would use version comparison
        # For now, return all migrations
        return [
            m for m in self.migrations
            if m.from_version == from_version and m.to_version == to_version
        ]
    
    def migrate(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """
        Migrate data from one version to another.
        
        Args:
            data: Data to migrate
            from_version: Source version
            to_version: Target version
        
        Returns:
            Migrated data
        """
        migrations = self.find_migrations(from_version, to_version)
        
        if not migrations:
            logger.warning(f"No migrations found from {from_version} to {to_version}")
            return data
        
        result = data
        for migration in migrations:
            result = migration.apply(result)
        
        return result


def migrate_config(
    config: Dict[str, Any],
    from_version: str,
    to_version: str
) -> Dict[str, Any]:
    """
    Migrate configuration between versions.
    
    Args:
        config: Configuration dictionary
        from_version: Source version
        to_version: Target version
    
    Returns:
        Migrated configuration
    """
    manager = MigrationManager()
    
    # Register common migrations
    # Example: Rename 'model_path' to 'model' in v1.0 -> v1.1
    manager.register(
        "1.0.0",
        "1.1.0",
        "Rename model_path to model",
        lambda data: {
            **{k: v for k, v in data.items() if k != "model_path"},
            "model": data.get("model_path", data.get("model"))
        }
    )
    
    return manager.migrate(config, from_version, to_version)













