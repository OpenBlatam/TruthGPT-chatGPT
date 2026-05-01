"""
Multi-Task Learner
==================

Multi-task learning for continual learning systems.
"""
import torch
import torch.nn as nn
import logging
from collections import defaultdict
from typing import Dict, Any, Tuple
from .config import ContinualLearningConfig, CLStrategy

logger = logging.getLogger(__name__)

class MultiTaskLearner:
    """Multi-task learning implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.shared_encoder = self._create_shared_encoder()
        self.task_heads = {}
        self.task_weights = {}
        self.training_history = []
        logger.info("✅ Multi-Task Learner initialized")
    
    def _create_shared_encoder(self) -> nn.Module:
        """Create shared encoder"""
        encoder = nn.Sequential(
            nn.Linear(self.config.model_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU()
        )
        
        return encoder
    
    def create_task_head(self, task_id: int) -> nn.Module:
        """Create task-specific head"""
        head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 10)  # 10 classes
        )
        
        return head
    
    def add_task(self, task_id: int):
        """Add new task"""
        logger.info(f"➕ Adding task {task_id} to multi-task learner")
        
        # Create task head
        self.task_heads[task_id] = self.create_task_head(task_id)
        
        # Initialize task weight
        self.task_weights[task_id] = 1.0
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass for specific task"""
        if task_id not in self.task_heads:
            raise ValueError(f"Task {task_id} not found in multi-task learner")
        
        # Shared encoder
        shared_features = self.shared_encoder(x)
        
        # Task-specific head
        output = self.task_heads[task_id](shared_features)
        
        return output
    
    def train_multi_task(self, task_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]], 
                        num_epochs: int = 10) -> Dict[str, Any]:
        """Train multi-task learning"""
        logger.info("🏋️ Training multi-task learning")
        
        # Add tasks if not exist
        for task_id in task_data.keys():
            if task_id not in self.task_heads:
                self.add_task(task_id)
        
        # Create optimizer
        all_params = list(self.shared_encoder.parameters())
        for task_head in self.task_heads.values():
            all_params.extend(list(task_head.parameters()))
        
        optimizer = torch.optim.Adam(all_params, lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        task_losses = defaultdict(list)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Train on each task
            for task_id, (data, labels) in task_data.items():
                # Forward pass
                output = self.forward(data, task_id)
                loss = criterion(output, labels)
                
                # Weighted loss
                weighted_loss = self.task_weights[task_id] * loss
                epoch_loss += weighted_loss.item()
                
                task_losses[task_id].append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
            
            training_losses.append(epoch_loss)
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Total Loss = {epoch_loss:.4f}")
        
        training_result = {
            'strategy': CLStrategy.MULTI_TASK_LEARNING.value,
            'epochs': num_epochs,
            'training_losses': training_losses,
            'task_losses': dict(task_losses),
            'final_loss': training_losses[-1],
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result

