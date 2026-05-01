"""
Backup utilities for optimization_core.

Provides utilities for backup and restore operations.
"""
import logging
import shutil
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

logger = logging.getLogger(__name__)


class BackupInfo(BaseModel):
    """Backup information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    path: Path
    created_at: datetime
    size: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BackupManager:
    """Manager for backup operations."""
    
    def __init__(self, backup_dir: Path):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backups: Dict[str, BackupInfo] = {}
        self._load_backup_index()
    
    def create_backup(
        self,
        source: Path,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BackupInfo:
        """
        Create a backup.
        
        Args:
            source: Source path to backup
            name: Backup name (auto-generated if None)
            metadata: Optional metadata
        
        Returns:
            Backup information
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / name
        
        # Copy source to backup
        if source.is_file():
            shutil.copy2(source, backup_path)
        elif source.is_dir():
            shutil.copytree(source, backup_path, dirs_exist_ok=True)
        else:
            raise ValueError(f"Source path does not exist: {source}")
        
        # Get size
        if backup_path.is_file():
            size = backup_path.stat().st_size
        else:
            size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
        
        backup_info = BackupInfo(
            name=name,
            path=backup_path,
            created_at=datetime.now(),
            size=size,
            metadata=metadata or {}
        )
        
        self.backups[name] = backup_info
        self._save_backup_index()
        
        logger.info(f"Backup created: {name} ({size} bytes)")
        
        return backup_info
    
    def restore_backup(
        self,
        backup_name: str,
        target: Path,
        overwrite: bool = False
    ):
        """
        Restore a backup.
        
        Args:
            backup_name: Name of backup to restore
            target: Target path for restoration
            overwrite: Whether to overwrite existing files
        """
        if backup_name not in self.backups:
            raise ValueError(f"Backup not found: {backup_name}")
        
        backup_info = self.backups[backup_name]
        backup_path = backup_info.path
        
        if not backup_path.exists():
            raise ValueError(f"Backup path does not exist: {backup_path}")
        
        # Check if target exists
        if target.exists() and not overwrite:
            raise ValueError(f"Target path exists: {target}")
        
        # Restore
        if backup_path.is_file():
            shutil.copy2(backup_path, target)
        else:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(backup_path, target)
        
        logger.info(f"Backup restored: {backup_name} -> {target}")
    
    def list_backups(self) -> List[BackupInfo]:
        """
        List all backups.
        
        Returns:
            List of backup information
        """
        return list(self.backups.values())
    
    def delete_backup(self, backup_name: str):
        """
        Delete a backup.
        
        Args:
            backup_name: Name of backup to delete
        """
        if backup_name not in self.backups:
            raise ValueError(f"Backup not found: {backup_name}")
        
        backup_info = self.backups[backup_name]
        
        if backup_info.path.exists():
            if backup_info.path.is_file():
                backup_info.path.unlink()
            else:
                shutil.rmtree(backup_info.path)
        
        del self.backups[backup_name]
        self._save_backup_index()
        
        logger.info(f"Backup deleted: {backup_name}")
    
    def _load_backup_index(self):
        """Load backup index from disk."""
        index_path = self.backup_dir / ".backup_index.json"
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    for name, info in data.items():
                        self.backups[name] = BackupInfo(
                            name=info["name"],
                            path=Path(info["path"]),
                            created_at=datetime.fromisoformat(info["created_at"]),
                            size=info["size"],
                            metadata=info.get("metadata", {})
                        )
            except Exception as e:
                logger.warning(f"Failed to load backup index: {e}")
    
    def _save_backup_index(self):
        """Save backup index to disk."""
        index_path = self.backup_dir / ".backup_index.json"
        
        data = {
            name: {
                "name": info.name,
                "path": str(info.path),
                "created_at": info.created_at.isoformat(),
                "size": info.size,
                "metadata": info.metadata,
            }
            for name, info in self.backups.items()
        }
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)













