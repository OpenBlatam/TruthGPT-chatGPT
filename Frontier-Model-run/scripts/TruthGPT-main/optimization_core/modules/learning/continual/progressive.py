"""
Progressive Network
===================

Progressive neural networks for continual learning.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from .config import ContinualLearningConfig, CLStrategy

logger = logging.getLogger(__name__)

class ProgressiveNetwork:
    """Progressive network implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.task_networks = {}
        self.task_adapters = {}
        self.current_task = 0
        self.training_history = []
        logger.info("✅ Progressive Network initialized")
    
    def create_task_network(self, task_id: int) -> nn.Module:
        """Create network for specific task"""
        network = nn.Sequential(
            nn.Linear(self.config.model_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 10)  # 10 classes
        )
        
        return network
    
    def create_task_adapter(self, task_id: int) -> nn.Module:
        """Create adapter for specific task"""
        adapter = nn.Sequential(
            nn.Linear(self.config.model_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 10)  # 10 classes
        )
        
        return adapter
    
    def add_task(self, task_id: int):
        """Add new task to progressive network"""
        logger.info(f"➕ Adding task {task_id} to progressive network")
        
        # Create task network
        self.task_networks[task_id] = self.create_task_network(task_id)
        
        # Create task adapter
        self.task_adapters[task_id] = self.create_task_adapter(task_id)
        
        self.current_task = task_id
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass for specific task"""
        if task_id not in self.task_networks:
            raise ValueError(f"Task {task_id} not found in progressive network")
        
        # Use task-specific network
        output = self.task_networks[task_id](x)
        
        return output
    
    def train_task(self, task_id: int, data: torch.Tensor, 
                  labels: torch.Tensor, num_epochs: int = 10) -> Dict[str, Any]:
        """Train specific task"""
        logger.info(f"🏋️ Training task {task_id}")
        
        # Add task if not exists
        if task_id not in self.task_networks:
            self.add_task(task_id)
        
        # Get task network
        task_network = self.task_networks[task_id]
        optimizer = torch.optim.Adam(task_network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Forward pass
            output = task_network(data)
            loss = criterion(output, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            training_losses.append(epoch_loss)
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        training_result = {
            'strategy': CLStrategy.PROGRESSIVE_NETWORKS.value,
            'task_id': task_id,
            'epochs': num_epochs,
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result

