"""
Migration utilities for polyglot_core.

Provides version migration and upgrade capabilities.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import json
import yaml


@dataclass
class Migration:
    """Migration definition."""
    from_version: str
    to_version: str
    description: str
    migrate_fn: Callable[[Dict[str, Any]], Dict[str, Any]]


class MigrationManager:
    """
    Migration manager for polyglot_core.
    
    Handles version migrations and upgrades.
    """
    
    def __init__(self):
        self._migrations: List[Migration] = []
    
    def register_migration(
        self,
        from_version: str,
        to_version: str,
        description: str,
        migrate_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """
        Register a migration.
        
        Args:
            from_version: Source version
            to_version: Target version
            description: Migration description
            migrate_fn: Migration function
        """
        migration = Migration(
            from_version=from_version,
            to_version=to_version,
            description=description,
            migrate_fn=migrate_fn
        )
        self._migrations.append(migration)
    
    def find_migration_path(self, from_version: str, to_version: str) -> List[Migration]:
        """
        Find migration path between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            List of migrations to apply
        """
        # Simple implementation: find direct migrations
        # In production, you'd want a more sophisticated path finding
        path = []
        
        for migration in self._migrations:
            if migration.from_version == from_version and migration.to_version == to_version:
                path.append(migration)
                break
        
        return path
    
    def migrate_config(
        self,
        config: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """
        Migrate configuration.
        
        Args:
            config: Configuration dictionary
            from_version: Source version
            to_version: Target version
            
        Returns:
            Migrated configuration
        """
        path = self.find_migration_path(from_version, to_version)
        
        if not path:
            raise ValueError(f"No migration path found from {from_version} to {to_version}")
        
        result = config.copy()
        
        for migration in path:
            result = migration.migrate_fn(result)
            result['version'] = migration.to_version
        
        return result
    
    def migrate_config_file(
        self,
        filepath: Path,
        to_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Migrate configuration file.
        
        Args:
            filepath: Configuration file path
            to_version: Target version (default: latest)
            
        Returns:
            Migrated configuration
        """
        # Load config
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config = json.load(f)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        from_version = config.get('version', '1.0.0')
        
        if to_version is None:
            from .version import get_version
            to_version = get_version()
        
        # Migrate
        migrated = self.migrate_config(config, from_version, to_version)
        
        # Save back
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(migrated, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                yaml.dump(migrated, f, default_flow_style=False)
        
        return migrated


# Global migration manager
_global_migration_manager = MigrationManager()


def get_migration_manager() -> MigrationManager:
    """Get global migration manager."""
    return _global_migration_manager


def register_migration(
    from_version: str,
    to_version: str,
    description: str,
    migrate_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
):
    """Convenience function to register migration."""
    _global_migration_manager.register_migration(from_version, to_version, description, migrate_fn)













