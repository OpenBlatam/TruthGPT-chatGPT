"""
Advanced Commit Tracker for TruthGPT Optimization Core
Deep Learning Enhanced Commit Tracking with Performance Analytics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import pickle
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommitStatus(Enum):
    """Commit status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OPTIMIZED = "optimized"

class CommitType(Enum):
    """Commit type enumeration"""
    OPTIMIZATION = "optimization"
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    PERFORMANCE = "performance"
    INFERENCE = "inference"
    TRAINING = "training"
    EVALUATION = "evaluation"

@dataclass
class OptimizationCommit:
    """Optimization commit with deep learning metrics"""
    commit_id: str
    commit_hash: str
    author: str
    timestamp: datetime
    message: str
    commit_type: CommitType
    status: CommitStatus
    
    # Performance metrics
    model_size: Optional[int] = None
    inference_time: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    
    # Optimization details
    optimization_techniques: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_architecture: Optional[str] = None
    dataset_size: Optional[int] = None
    training_epochs: Optional[int] = None
    
    # Version control
    parent_commits: List[str] = field(default_factory=list)
    child_commits: List[str] = field(default_factory=list)
    branch: str = "main"
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    notes: Optional[str] = None

class CommitDataset(Dataset):
    """PyTorch Dataset for commit data"""
    
    def __init__(self, commits: List[OptimizationCommit], feature_extractor=None):
        self.commits = commits
        self.feature_extractor = feature_extractor or self._default_feature_extractor
        
    def __len__(self):
        return len(self.commits)
    
    def __getitem__(self, idx):
        commit = self.commits[idx]
        features = self.feature_extractor(commit)
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'commit_id': commit.commit_id,
            'commit_type': commit.commit_type.value,
            'status': commit.status.value,
            'performance_metrics': self._extract_performance_metrics(commit)
        }
    
    def _default_feature_extractor(self, commit: OptimizationCommit) -> List[float]:
        """Extract numerical features from commit"""
        features = []
        
        # Basic features
        features.append(len(commit.message))
        features.append(len(commit.optimization_techniques))
        features.append(len(commit.hyperparameters))
        
        # Performance metrics (normalized)
        if commit.inference_time:
            features.append(min(commit.inference_time / 1000.0, 1.0))  # Normalize to 0-1
        else:
            features.append(0.0)
            
        if commit.memory_usage:
            features.append(min(commit.memory_usage / 10000.0, 1.0))  # Normalize to 0-1
        else:
            features.append(0.0)
            
        if commit.gpu_utilization:
            features.append(commit.gpu_utilization / 100.0)  # Normalize to 0-1
        else:
            features.append(0.0)
            
        if commit.accuracy:
            features.append(commit.accuracy)
        else:
            features.append(0.0)
            
        if commit.loss:
            features.append(min(commit.loss, 10.0) / 10.0)  # Normalize to 0-1
        else:
            features.append(0.0)
        
        # Model size (log scale)
        if commit.model_size:
            features.append(min(math.log(commit.model_size + 1) / 20.0, 1.0))
        else:
            features.append(0.0)
            
        # Training epochs
        if commit.training_epochs:
            features.append(min(commit.training_epochs / 100.0, 1.0))
        else:
            features.append(0.0)
            
        return features
    
    def _extract_performance_metrics(self, commit: OptimizationCommit) -> Dict[str, float]:
        """Extract performance metrics as dictionary"""
        return {
            'inference_time': commit.inference_time or 0.0,
            'memory_usage': commit.memory_usage or 0.0,
            'gpu_utilization': commit.gpu_utilization or 0.0,
            'accuracy': commit.accuracy or 0.0,
            'loss': commit.loss or 0.0,
            'model_size': commit.model_size or 0.0
        }

class CommitPerformancePredictor(nn.Module):
    """Neural network for predicting commit performance"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 6):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Performance prediction heads
        self.inference_time_head = nn.Linear(hidden_dim // 2, 1)
        self.memory_head = nn.Linear(hidden_dim // 2, 1)
        self.gpu_head = nn.Linear(hidden_dim // 2, 1)
        self.accuracy_head = nn.Linear(hidden_dim // 2, 1)
        self.loss_head = nn.Linear(hidden_dim // 2, 1)
        self.model_size_head = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        encoded = self.encoder(x)
        
        return {
            'inference_time': self.inference_time_head(encoded),
            'memory_usage': self.memory_head(encoded),
            'gpu_utilization': self.gpu_head(encoded),
            'accuracy': self.accuracy_head(encoded),
            'loss': self.loss_head(encoded),
            'model_size': self.model_size_head(encoded)
        }

class CommitTracker:
    """Advanced commit tracker with deep learning capabilities"""
    
    def __init__(self, 
                 device: str = None,
                 model_path: str = "commit_model.pth",
                 use_mixed_precision: bool = True):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.use_mixed_precision = use_mixed_precision
        
        # Initialize components
        self.commits: List[OptimizationCommit] = []
        self.performance_predictor = CommitPerformancePredictor().to(self.device)
        self.optimizer = optim.AdamW(self.performance_predictor.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Mixed precision scaler
        if self.use_mixed_precision and self.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Statistics
        self.stats = {
            'total_commits': 0,
            'optimization_commits': 0,
            'average_performance': {},
            'best_performance': {},
            'trends': {}
        }
        
        logger.info(f"CommitTracker initialized on device: {self.device}")
    
    def add_commit(self, commit: OptimizationCommit) -> str:
        """Add a new optimization commit"""
        try:
            # Validate commit
            self._validate_commit(commit)
            
            # Add to storage
            self.commits.append(commit)
            
            # Update statistics
            self._update_statistics(commit)
            
            # Save to disk
            self._save_commits()
            
            logger.info(f"Added commit {commit.commit_id} by {commit.author}")
            return commit.commit_id
            
        except Exception as e:
            logger.error(f"Failed to add commit: {e}")
            raise
    
    def get_commit(self, commit_id: str) -> Optional[OptimizationCommit]:
        """Get commit by ID"""
        for commit in self.commits:
            if commit.commit_id == commit_id:
                return commit
        return None
    
    def get_commits_by_author(self, author: str) -> List[OptimizationCommit]:
        """Get commits by author"""
        return [commit for commit in self.commits if commit.author == author]
    
    def get_commits_by_type(self, commit_type: CommitType) -> List[OptimizationCommit]:
        """Get commits by type"""
        return [commit for commit in self.commits if commit.commit_type == commit_type]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.commits:
            return {}
        
        # Calculate statistics
        inference_times = [c.inference_time for c in self.commits if c.inference_time]
        memory_usage = [c.memory_usage for c in self.commits if c.memory_usage]
        gpu_utilization = [c.gpu_utilization for c in self.commits if c.gpu_utilization]
        accuracies = [c.accuracy for c in self.commits if c.accuracy]
        losses = [c.loss for c in self.commits if c.loss]
        
        stats = {
            'total_commits': len(self.commits),
            'commits_with_metrics': len([c for c in self.commits if c.inference_time]),
            'average_inference_time': np.mean(inference_times) if inference_times else 0,
            'average_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'average_gpu_utilization': np.mean(gpu_utilization) if gpu_utilization else 0,
            'average_accuracy': np.mean(accuracies) if accuracies else 0,
            'average_loss': np.mean(losses) if losses else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'fastest_inference': min(inference_times) if inference_times else 0,
            'lowest_memory': min(memory_usage) if memory_usage else 0
        }
        
        return stats
    
    def train_performance_predictor(self, 
                                   epochs: int = 100,
                                   batch_size: int = 32,
                                   validation_split: float = 0.2) -> Dict[str, List[float]]:
        """Train the performance prediction model"""
        
        if len(self.commits) < 10:
            logger.warning("Not enough commits for training. Need at least 10 commits.")
            return {}
        
        # Create dataset
        dataset = CommitDataset(self.commits)
        
        # Split data
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.performance_predictor.train()
            train_loss = 0.0
            train_accuracy = 0.0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                metrics = batch['performance_metrics']
                
                # Convert metrics to tensor
                target_metrics = torch.stack([
                    torch.tensor([m['inference_time']], dtype=torch.float32) for m in metrics
                ]).to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        predictions = self.performance_predictor(features)
                        loss = nn.MSELoss()(predictions['inference_time'], target_metrics)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predictions = self.performance_predictor(features)
                    loss = nn.MSELoss()(predictions['inference_time'], target_metrics)
                    loss.backward()
                    self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.performance_predictor.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    metrics = batch['performance_metrics']
                    
                    target_metrics = torch.stack([
                        torch.tensor([m['inference_time']], dtype=torch.float32) for m in metrics
                    ]).to(self.device)
                    
                    predictions = self.performance_predictor(features)
                    loss = nn.MSELoss()(predictions['inference_time'], target_metrics)
                    val_loss += loss.item()
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model
        self._save_model()
        
        logger.info("Performance predictor training completed")
        return history
    
    def predict_performance(self, commit: OptimizationCommit) -> Dict[str, float]:
        """Predict performance for a commit"""
        self.performance_predictor.eval()
        
        # Extract features
        dataset = CommitDataset([commit])
        features = dataset[0]['features'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.performance_predictor(features)
        
        return {
            'inference_time': predictions['inference_time'].item(),
            'memory_usage': predictions['memory_usage'].item(),
            'gpu_utilization': predictions['gpu_utilization'].item(),
            'accuracy': predictions['accuracy'].item(),
            'loss': predictions['loss'].item(),
            'model_size': predictions['model_size'].item()
        }
    
    def get_optimization_recommendations(self, commit: OptimizationCommit) -> List[str]:
        """Get optimization recommendations based on commit analysis"""
        recommendations = []
        
        # Analyze performance metrics
        if commit.inference_time and commit.inference_time > 1000:  # > 1 second
            recommendations.append("Consider model quantization or pruning to reduce inference time")
        
        if commit.memory_usage and commit.memory_usage > 8000:  # > 8GB
            recommendations.append("Optimize memory usage with gradient checkpointing or model sharding")
        
        if commit.gpu_utilization and commit.gpu_utilization < 50:  # < 50%
            recommendations.append("Increase batch size or use mixed precision training for better GPU utilization")
        
        if commit.accuracy and commit.accuracy < 0.8:  # < 80%
            recommendations.append("Consider data augmentation or model architecture improvements")
        
        if commit.loss and commit.loss > 2.0:
            recommendations.append("Review loss function and learning rate schedule")
        
        # Architecture-specific recommendations
        if 'transformer' in commit.model_architecture.lower():
            recommendations.append("Consider using Flash Attention for better memory efficiency")
        
        if 'cnn' in commit.model_architecture.lower():
            recommendations.append("Try depthwise separable convolutions for efficiency")
        
        return recommendations
    
    def _validate_commit(self, commit: OptimizationCommit):
        """Validate commit data"""
        if not commit.commit_id:
            raise ValueError("Commit ID is required")
        
        if not commit.author:
            raise ValueError("Author is required")
        
        if not commit.message:
            raise ValueError("Message is required")
    
    def _update_statistics(self, commit: OptimizationCommit):
        """Update internal statistics"""
        self.stats['total_commits'] += 1
        
        if commit.commit_type == CommitType.OPTIMIZATION:
            self.stats['optimization_commits'] += 1
        
        # Update performance averages
        if commit.inference_time:
            if 'inference_time' not in self.stats['average_performance']:
                self.stats['average_performance']['inference_time'] = []
            self.stats['average_performance']['inference_time'].append(commit.inference_time)
    
    def _save_commits(self):
        """Save commits to disk"""
        try:
            with open(self.model_path.parent / "commits.json", 'w') as f:
                commits_data = []
                for commit in self.commits:
                    commit_dict = {
                        'commit_id': commit.commit_id,
                        'commit_hash': commit.commit_hash,
                        'author': commit.author,
                        'timestamp': commit.timestamp.isoformat(),
                        'message': commit.message,
                        'commit_type': commit.commit_type.value,
                        'status': commit.status.value,
                        'model_size': commit.model_size,
                        'inference_time': commit.inference_time,
                        'memory_usage': commit.memory_usage,
                        'gpu_utilization': commit.gpu_utilization,
                        'accuracy': commit.accuracy,
                        'loss': commit.loss,
                        'optimization_techniques': commit.optimization_techniques,
                        'hyperparameters': commit.hyperparameters,
                        'model_architecture': commit.model_architecture,
                        'dataset_size': commit.dataset_size,
                        'training_epochs': commit.training_epochs,
                        'parent_commits': commit.parent_commits,
                        'child_commits': commit.child_commits,
                        'branch': commit.branch,
                        'tags': commit.tags,
                        'experiment_id': commit.experiment_id,
                        'run_id': commit.run_id,
                        'notes': commit.notes
                    }
                    commits_data.append(commit_dict)
                
                json.dump(commits_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save commits: {e}")
    
    def _save_model(self):
        """Save the performance prediction model"""
        try:
            torch.save({
                'model_state_dict': self.performance_predictor.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'stats': self.stats
            }, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load the performance prediction model"""
        try:
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.performance_predictor.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.stats = checkpoint.get('stats', self.stats)
                logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

# Factory functions
def create_commit_tracker(device: str = None, model_path: str = "commit_model.pth") -> CommitTracker:
    """Create a new commit tracker instance"""
    return CommitTracker(device=device, model_path=model_path)

def track_optimization_commit(tracker: CommitTracker, 
                            commit_id: str,
                            author: str,
                            message: str,
                            commit_type: CommitType = CommitType.OPTIMIZATION,
                            **kwargs) -> str:
    """Track a new optimization commit"""
    commit = OptimizationCommit(
        commit_id=commit_id,
        commit_hash=kwargs.get('commit_hash', ''),
        author=author,
        timestamp=datetime.now(),
        message=message,
        commit_type=commit_type,
        status=CommitStatus.PENDING,
        **kwargs
    )
    
    return tracker.add_commit(commit)

def get_commit_history(tracker: CommitTracker, 
                      author: str = None,
                      commit_type: CommitType = None,
                      limit: int = None) -> List[OptimizationCommit]:
    """Get commit history with optional filtering"""
    commits = tracker.commits
    
    if author:
        commits = [c for c in commits if c.author == author]
    
    if commit_type:
        commits = [c for c in commits if c.commit_type == commit_type]
    
    if limit:
        commits = commits[:limit]
    
    return commits

def get_commit_statistics(tracker: CommitTracker) -> Dict[str, Any]:
    """Get comprehensive commit statistics"""
    return tracker.get_performance_statistics()
