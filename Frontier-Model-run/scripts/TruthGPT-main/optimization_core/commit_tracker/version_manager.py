"""
Version Manager for TruthGPT Optimization Core
Advanced version control with model checkpointing and experiment tracking
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb
import json
import pickle
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import hashlib
import yaml
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class VersionType(Enum):
    """Version type enumeration"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    EXPERIMENTAL = "experimental"
    RELEASE = "release"
    HOTFIX = "hotfix"

class VersionStatus(Enum):
    """Version status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

@dataclass
class VersionInfo:
    """Version information with model metadata"""
    version: str
    version_type: VersionType
    status: VersionStatus
    created_at: datetime
    author: str
    description: str
    
    # Model information
    model_path: Optional[str] = None
    model_size: Optional[int] = None
    model_architecture: Optional[str] = None
    model_hash: Optional[str] = None
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    
    # Training information
    training_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment tracking
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    wandb_run_id: Optional[str] = None
    
    # Dependencies
    parent_versions: List[str] = field(default_factory=list)
    child_versions: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    changelog: List[str] = field(default_factory=list)

class ModelCheckpoint:
    """Model checkpoint with metadata"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epoch: int,
                 loss: float,
                 metrics: Dict[str, float],
                 config: Dict[str, Any]):
        
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.loss = loss
        self.metrics = metrics
        self.config = config
        self.timestamp = datetime.now()
        self.checkpoint_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique checkpoint ID"""
        content = f"{self.epoch}_{self.loss}_{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def save(self, path: Path) -> str:
        """Save checkpoint to disk"""
        checkpoint_path = path / f"checkpoint_{self.checkpoint_id}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'loss': self.loss,
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': self.timestamp.isoformat(),
            'checkpoint_id': self.checkpoint_id
        }, checkpoint_path)
        
        return str(checkpoint_path)
    
    @classmethod
    def load(cls, path: Path, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Load checkpoint from disk"""
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return cls(
            model=model,
            optimizer=optimizer,
            epoch=checkpoint['epoch'],
            loss=checkpoint['loss'],
            metrics=checkpoint['metrics'],
            config=checkpoint['config']
        )

class VersionManager:
    """Advanced version manager with model checkpointing"""
    
    def __init__(self, 
                 base_path: str = "versions",
                 use_wandb: bool = True,
                 use_tensorboard: bool = True,
                 project_name: str = "truthgpt-optimization"):
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.project_name = project_name
        
        # Initialize tracking
        self.versions: Dict[str, VersionInfo] = {}
        self.checkpoints: Dict[str, ModelCheckpoint] = {}
        
        # TensorBoard writer
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.base_path / "tensorboard"))
        else:
            self.tb_writer = None
        
        # Wandb initialization
        if self.use_wandb:
            wandb.init(project=project_name, reinit=True)
        
        # Load existing versions
        self._load_versions()
        
        logger.info(f"VersionManager initialized at {self.base_path}")
        
    def create_version(self,
                      version_type: VersionType,
                      author: str,
                      description: str,
                      model: Optional[nn.Module] = None,
                      performance_metrics: Optional[Dict[str, float]] = None,
                      **kwargs) -> str:
        """Create a new version"""
        
        # Generate version number
        version = self._generate_version_number(version_type)
        
        # Create version info
        version_info = VersionInfo(
            version=version,
            version_type=version_type,
            status=VersionStatus.DRAFT,
            created_at=datetime.now(),
            author=author,
            description=description,
            performance_metrics=performance_metrics or {},
            **kwargs
        )
        
        # Save model if provided
        if model:
            model_path = self._save_model(version, model)
            version_info.model_path = model_path
            version_info.model_size = self._calculate_model_size(model)
            version_info.model_hash = self._calculate_model_hash(model)
        
        # Store version
        self.versions[version] = version_info
        
        # Save to disk
        self._save_version(version_info)
        
        # Log to tracking systems
        self._log_version_creation(version_info)
        
        logger.info(f"Created version {version} by {author}")
        return version
    
    def create_checkpoint(self, 
                        version: str,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        loss: float,
                        metrics: Dict[str, float],
                        config: Dict[str, Any]) -> str:
        """Create a model checkpoint"""
        
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Create checkpoint
        checkpoint = ModelCheckpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            metrics=metrics,
            config=config
        )
        
        # Save checkpoint
        version_path = self.base_path / version
        version_path.mkdir(exist_ok=True)
        checkpoint_path = checkpoint.save(version_path)
        
        # Store checkpoint info
        self.checkpoints[f"{version}_{checkpoint.checkpoint_id}"] = checkpoint
        
        # Log to tracking systems
        self._log_checkpoint(version, checkpoint)
        
        logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for version {version}")
        return checkpoint_path
    
    def load_checkpoint(self, 
                       version: str, 
                       checkpoint_id: str,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer) -> ModelCheckpoint:
        """Load a model checkpoint"""
        
        checkpoint_key = f"{version}_{checkpoint_id}"
        if checkpoint_key not in self.checkpoints:
            # Try to load from disk
            version_path = self.base_path / version
            checkpoint_path = version_path / f"checkpoint_{checkpoint_id}.pth"
            
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint {checkpoint_id} not found for version {version}")
            
            checkpoint = ModelCheckpoint.load(checkpoint_path, model, optimizer)
            self.checkpoints[checkpoint_key] = checkpoint
        
        return self.checkpoints[checkpoint_key]
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get version information"""
        return self.versions.get(version)
    
    def get_version_history(self, limit: int = None) -> List[VersionInfo]:
        """Get version history"""
        versions = list(self.versions.values())
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions"""
        
        if version1 not in self.versions or version2 not in self.versions:
            raise ValueError("One or both versions not found")
        
        v1 = self.versions[version1]
        v2 = self.versions[version2]
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'performance_delta': {},
            'model_differences': {},
            'config_differences': {}
        }
        
        # Compare performance metrics
        for metric in set(v1.performance_metrics.keys()) | set(v2.performance_metrics.keys()):
            val1 = v1.performance_metrics.get(metric, 0)
            val2 = v2.performance_metrics.get(metric, 0)
            comparison['performance_delta'][metric] = val2 - val1
        
        # Compare model information
        comparison['model_differences'] = {
            'size_delta': (v2.model_size or 0) - (v1.model_size or 0),
            'architecture_changed': v1.model_architecture != v2.model_architecture,
            'hash_different': v1.model_hash != v2.model_hash
        }
        
        # Compare configurations
        comparison['config_differences'] = self._compare_configs(
            v1.training_config, v2.training_config
        )
        
        return comparison
    
    def get_best_version(self, metric: str = "accuracy") -> Optional[VersionInfo]:
        """Get the best version based on a metric"""
        
        if not self.versions:
            return None
        
        best_version = None
        best_score = float('-inf')
        
        for version_info in self.versions.values():
            if metric in version_info.performance_metrics:
                score = version_info.performance_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_version = version_info
        
        return best_version
    
    def archive_version(self, version: str) -> bool:
        """Archive a version"""
        
        if version not in self.versions:
            return False
        
        self.versions[version].status = VersionStatus.ARCHIVED
        self._save_version(self.versions[version])
        
        logger.info(f"Archived version {version}")
        return True
    
    def _generate_version_number(self, version_type: VersionType) -> str:
        """Generate semantic version number"""
        
        # Get existing versions of the same type
        existing_versions = [
            v for v in self.versions.values() 
            if v.version_type == version_type
        ]
        
        if not existing_versions:
            if version_type == VersionType.MAJOR:
                return "1.0.0"
            elif version_type == VersionType.MINOR:
                return "0.1.0"
            elif version_type == VersionType.PATCH:
                return "0.0.1"
            else:
                return "0.0.0"
        
        # Parse existing version numbers
        version_numbers = []
        for v in existing_versions:
            try:
                parts = v.version.split('.')
                if len(parts) == 3:
                    version_numbers.append([int(p) for p in parts])
            except ValueError:
                continue
        
        if not version_numbers:
            return "1.0.0"
        
        # Find the latest version
        latest = max(version_numbers)
        
        # Increment based on type
        if version_type == VersionType.MAJOR:
            return f"{latest[0] + 1}.0.0"
        elif version_type == VersionType.MINOR:
            return f"{latest[0]}.{latest[1] + 1}.0"
        elif version_type == VersionType.PATCH:
            return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
        else:
            return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
    
    def _save_model(self, version: str, model: nn.Module) -> str:
        """Save model to disk"""
        
        version_path = self.base_path / version
        version_path.mkdir(exist_ok=True)
        
        model_path = version_path / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        return str(model_path)
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Calculate model hash"""
        state_dict = model.state_dict()
        content = str(sorted(state_dict.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _save_version(self, version_info: VersionInfo):
        """Save version information to disk"""
        
        version_path = self.base_path / version_info.version
        version_path.mkdir(exist_ok=True)
        
        # Save version metadata
        metadata = {
                            'version': version_info.version,
                            'version_type': version_info.version_type.value,
            'status': version_info.status.value,
            'created_at': version_info.created_at.isoformat(),
                            'author': version_info.author,
                            'description': version_info.description,
            'model_path': version_info.model_path,
            'model_size': version_info.model_size,
            'model_architecture': version_info.model_architecture,
            'model_hash': version_info.model_hash,
            'performance_metrics': version_info.performance_metrics,
            'benchmark_results': version_info.benchmark_results,
            'training_config': version_info.training_config,
            'hyperparameters': version_info.hyperparameters,
            'dataset_info': version_info.dataset_info,
            'experiment_id': version_info.experiment_id,
            'run_id': version_info.run_id,
            'wandb_run_id': version_info.wandb_run_id,
            'parent_versions': version_info.parent_versions,
                            'child_versions': version_info.child_versions,
                            'tags': version_info.tags,
            'notes': version_info.notes,
            'changelog': version_info.changelog
        }
        
        with open(version_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_versions(self):
        """Load existing versions from disk"""
        
        for version_dir in self.base_path.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                version_info = VersionInfo(
                            version=metadata['version'],
                            version_type=VersionType(metadata['version_type']),
                            status=VersionStatus(metadata['status']),
                            created_at=datetime.fromisoformat(metadata['created_at']),
                            author=metadata['author'],
                            description=metadata['description'],
                            model_path=metadata.get('model_path'),
                            model_size=metadata.get('model_size'),
                            model_architecture=metadata.get('model_architecture'),
                            model_hash=metadata.get('model_hash'),
                            performance_metrics=metadata.get('performance_metrics', {}),
                            benchmark_results=metadata.get('benchmark_results', {}),
                            training_config=metadata.get('training_config', {}),
                            hyperparameters=metadata.get('hyperparameters', {}),
                            dataset_info=metadata.get('dataset_info', {}),
                            experiment_id=metadata.get('experiment_id'),
                            run_id=metadata.get('run_id'),
                            wandb_run_id=metadata.get('wandb_run_id'),
                            parent_versions=metadata.get('parent_versions', []),
                            child_versions=metadata.get('child_versions', []),
                            tags=metadata.get('tags', []),
                            notes=metadata.get('notes'),
                            changelog=metadata.get('changelog', [])
                        )
                        
                        self.versions[version_info.version] = version_info
                        
        except Exception as e:
                        logger.error(f"Failed to load version {version_dir.name}: {e}")
    
    def _log_version_creation(self, version_info: VersionInfo):
        """Log version creation to tracking systems"""
        
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_text(f"version/{version_info.version}/description", 
                                  version_info.description)
            self.tb_writer.add_text(f"version/{version_info.version}/author", 
                                  version_info.author)
        
        # Wandb
        if self.use_wandb:
            wandb.log({
                f"version/{version_info.version}/created": 1,
                f"version/{version_info.version}/type": version_info.version_type.value,
                f"version/{version_info.version}/author": version_info.author
            })
    
    def _log_checkpoint(self, version: str, checkpoint: ModelCheckpoint):
        """Log checkpoint to tracking systems"""
        
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar(f"checkpoint/{version}/loss", checkpoint.loss, checkpoint.epoch)
            for metric, value in checkpoint.metrics.items():
                self.tb_writer.add_scalar(f"checkpoint/{version}/{metric}", value, checkpoint.epoch)
        
        # Wandb
        if self.use_wandb:
            wandb.log({
                f"checkpoint/{version}/epoch": checkpoint.epoch,
                f"checkpoint/{version}/loss": checkpoint.loss,
                **{f"checkpoint/{version}/{k}": v for k, v in checkpoint.metrics.items()}
            })
    
    def _compare_configs(self, config1: Dict, config2: Dict) -> Dict[str, Any]:
        """Compare two configuration dictionaries"""
        
        differences = {}
        
        # Find keys in both configs
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                differences[key] = {
                    'old': val1,
                    'new': val2
                }
        
        return differences

# Factory functions
def create_version_manager(base_path: str = "versions", 
                         use_wandb: bool = True,
                         use_tensorboard: bool = True) -> VersionManager:
    """Create a new version manager"""
    return VersionManager(base_path=base_path, 
                         use_wandb=use_wandb, 
                         use_tensorboard=use_tensorboard)

def create_version(version_manager: VersionManager,
                  version_type: VersionType,
                  author: str,
                  description: str,
                  **kwargs) -> str:
    """Create a new version"""
    return version_manager.create_version(version_type, author, description, **kwargs)

def get_version_info(version_manager: VersionManager, version: str) -> Optional[VersionInfo]:
    """Get version information"""
    return version_manager.get_version_info(version)

def get_version_history(version_manager: VersionManager, limit: int = None) -> List[VersionInfo]:
    """Get version history"""
    return version_manager.get_version_history(limit)
