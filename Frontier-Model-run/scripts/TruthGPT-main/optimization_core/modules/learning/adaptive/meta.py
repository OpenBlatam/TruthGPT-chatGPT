"""
Meta-Learning System
===================

Meta-learning for learning how to learn in adaptive systems.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import random
import time
from typing import Dict, Any, List
from collections import deque
from .config import AdaptiveLearningConfig

logger = logging.getLogger(__name__)

class MetaLearner:
    """Meta-learning system for learning how to learn"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.meta_model = self._create_meta_model()
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=config.meta_learning_rate)
        self.task_memory = deque(maxlen=1000)
        self.learning_curves = {}
        
        logger.info("✅ Meta-Learner initialized")
    
    def _create_meta_model(self) -> nn.Module:
        """Create meta-learning model"""
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def learn_from_task(self, task_features: np.ndarray, task_performance: float):
        """Learn from a completed task"""
        # Store task information
        task_info = {
            'features': task_features,
            'performance': task_performance,
            'timestamp': time.time()
        }
        self.task_memory.append(task_info)
        
        # Update meta-model
        if len(self.task_memory) >= self.config.meta_batch_size:
            self._update_meta_model()
    
    def _update_meta_model(self):
        """Update meta-learning model"""
        # Sample batch from task memory
        batch_size = min(self.config.meta_batch_size, len(self.task_memory))
        batch = random.sample(list(self.task_memory), batch_size)
        
        # Prepare training data
        features = np.array([task['features'] for task in batch])
        performances = np.array([task['performance'] for task in batch])
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        performances_tensor = torch.tensor(performances, dtype=torch.float32).unsqueeze(1)
        
        # Train meta-model
        self.meta_optimizer.zero_grad()
        
        # Forward pass
        predictions = self.meta_model(features_tensor)
        
        # Calculate loss
        loss = nn.MSELoss()(predictions, performances_tensor)
        
        # Backward pass
        loss.backward()
        self.meta_optimizer.step()
        
        logger.debug(f"✅ Meta-model updated (loss: {loss.item():.6f})")
    
    def predict_task_performance(self, task_features: np.ndarray) -> float:
        """Predict performance for a new task"""
        with torch.no_grad():
            features_tensor = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0)
            prediction = self.meta_model(features_tensor)
            return prediction.item()
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from meta-learning"""
        if not self.task_memory:
            return {'insights': 'No tasks learned yet'}
        
        # Analyze task patterns
        recent_tasks = list(self.task_memory)[-50:]  # Last 50 tasks
        
        performances = [task['performance'] for task in recent_tasks]
        avg_performance = np.mean(performances)
        performance_std = np.std(performances)
        
        # Analyze learning trends
        if len(performances) >= 10:
            recent_10 = performances[-10:]
            earlier_10 = performances[-20:-10] if len(performances) >= 20 else performances[:-10]
            
            improvement = np.mean(recent_10) - np.mean(earlier_10)
        else:
            improvement = 0.0
        
        return {
            'total_tasks': len(self.task_memory),
            'avg_performance': avg_performance,
            'performance_std': performance_std,
            'recent_improvement': improvement,
            'learning_trend': 'improving' if improvement > 0 else 'stable'
        }

