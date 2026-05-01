"""
Backup and restore utilities for polyglot_core.

Provides backup and restore capabilities for configurations and data.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import yaml
import shutil
import zipfile
import tempfile


class BackupManager:
    """
    Backup manager for polyglot_core.
    
    Manages backups of configurations and data.
    """
    
    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Backup directory (default: ~/.polyglot_core/backups)
        """
        if backup_dir is None:
            backup_dir = Path.home() / ".polyglot_core" / "backups"
        
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        data_files: Optional[List[Path]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Create backup.
        
        Args:
            name: Backup name
            config: Configuration to backup
            data_files: List of data files to backup
            metadata: Additional metadata
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name}_{timestamp}.zip"
        backup_path = self.backup_dir / backup_name
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Backup configuration
            if config:
                zipf.writestr("config.json", json.dumps(config, indent=2))
            
            # Backup data files
            if data_files:
                for file_path in data_files:
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)
            
            # Backup metadata
            backup_metadata = {
                'name': name,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                **(metadata or {})
            }
            zipf.writestr("metadata.json", json.dumps(backup_metadata, indent=2))
        
        return backup_path
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if "metadata.json" in zipf.namelist():
                        metadata_str = zipf.read("metadata.json").decode('utf-8')
                        metadata = json.loads(metadata_str)
                        metadata['file'] = str(backup_file)
                        metadata['size'] = backup_file.stat().st_size
                        backups.append(metadata)
            except Exception:
                # Skip corrupted backups
                continue
        
        return sorted(backups, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def restore_backup(
        self,
        backup_path: Path,
        restore_config: bool = True,
        restore_data: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Restore backup.
        
        Args:
            backup_path: Path to backup file
            restore_config: Whether to restore configuration
            restore_data: Whether to restore data files
            output_dir: Output directory for restored files
            
        Returns:
            Restore information
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        if output_dir is None:
            output_dir = Path.cwd()
        
        restore_info = {
            'backup_file': str(backup_path),
            'restored_at': datetime.now().isoformat(),
            'restored_files': []
        }
        
        with zipfile.ZipFile(backup_path, 'r') as zipf:
            # Extract metadata
            if "metadata.json" in zipf.namelist():
                metadata_str = zipf.read("metadata.json").decode('utf-8')
                restore_info['metadata'] = json.loads(metadata_str)
            
            # Restore configuration
            if restore_config and "config.json" in zipf.namelist():
                config_str = zipf.read("config.json").decode('utf-8')
                config = json.loads(config_str)
                
                config_file = output_dir / "restored_config.json"
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                restore_info['restored_files'].append(str(config_file))
            
            # Restore data files
            if restore_data:
                for file_name in zipf.namelist():
                    if file_name not in ["config.json", "metadata.json"]:
                        zipf.extract(file_name, output_dir)
                        restore_info['restored_files'].append(str(output_dir / file_name))
        
        return restore_info
    
    def delete_backup(self, backup_path: Path) -> bool:
        """
        Delete backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            backup_path.unlink()
            return True
        except Exception:
            return False
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """
        Cleanup old backups, keeping only the most recent ones.
        
        Args:
            keep_count: Number of backups to keep
        """
        backups = self.list_backups()
        
        if len(backups) > keep_count:
            # Sort by creation time and remove oldest
            backups_to_delete = backups[keep_count:]
            
            for backup in backups_to_delete:
                backup_file = Path(backup['file'])
                self.delete_backup(backup_file)


# Global backup manager
_global_backup_manager = BackupManager()


def get_backup_manager() -> BackupManager:
    """Get global backup manager."""
    return _global_backup_manager


def create_backup(name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Path:
    """Convenience function to create backup."""
    return _global_backup_manager.create_backup(name, config, **kwargs)













