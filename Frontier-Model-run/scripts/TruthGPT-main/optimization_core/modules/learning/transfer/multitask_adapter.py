"""
Multi-Task Adapter
==================

Multi-task adapter implementation.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Tuple
from collections import defaultdict
from .config import TransferLearningConfig
from .enums import TransferStrategy

logger = logging.getLogger(__name__)

class MultiTaskAdapter:
    """Multi-task adapter implementation"""
    
    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.shared_encoder = None
        self.task_adapters = {}
        self.training_history = []
        logger.info("✅ Multi-Task Adapter initialized")
    
    def create_shared_encoder(self) -> nn.Module:
        """Create shared encoder"""
        shared_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.config.feature_dim)
        )
        return shared_encoder
    
    def create_task_adapter(self, task_id: int) -> nn.Module:
        """Create task-specific adapter"""
        adapter = nn.Sequential(
            nn.Linear(self.config.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.config.num_classes)
        )
        return adapter
    
    def adapt_multi_task(self, task_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]], 
                        num_epochs: int = 10) -> Dict[str, Any]:
        """Adapt to multiple tasks"""
        logger.info("🔄 Adapting to multiple tasks")
        
        # Create shared encoder
        self.shared_encoder = self.create_shared_encoder()
        
        # Create task adapters
        for task_id in task_data.keys():
            self.task_adapters[task_id] = self.create_task_adapter(task_id)
        
        # Create optimizers
        shared_optimizer = torch.optim.Adam(self.shared_encoder.parameters(), lr=self.config.learning_rate)
        adapter_optimizers = {}
        for task_id, adapter in self.task_adapters.items():
            adapter_optimizers[task_id] = torch.optim.Adam(adapter.parameters(), lr=self.config.learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        
        adaptation_losses = []
        task_accuracies = defaultdict(list)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Train on each task
            for task_id, (data, labels) in task_data.items():
                # Forward pass
                shared_features = self.shared_encoder(data)
                task_outputs = self.task_adapters[task_id](shared_features)
                
                # Calculate loss
                task_loss = criterion(task_outputs, labels)
                epoch_loss += task_loss.item()
                
                # Backward pass
                shared_optimizer.zero_grad()
                adapter_optimizers[task_id].zero_grad()
                
                task_loss.backward()
                
                shared_optimizer.step()
                adapter_optimizers[task_id].step()
                
                # Calculate accuracy
                _, predicted = torch.max(task_outputs.data, 1)
                accuracy = (predicted == labels).float().mean()
                task_accuracies[task_id].append(accuracy.item())
            
            adaptation_losses.append(epoch_loss)
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Total Loss = {epoch_loss:.4f}")
        
        adaptation_result = {
            'strategy': TransferStrategy.MULTI_TASK_ADAPTER.value,
            'epochs': num_epochs,
            'adaptation_losses': adaptation_losses,
            'task_accuracies': dict(task_accuracies),
            'final_loss': adaptation_losses[-1],
            'num_tasks': len(task_data),
            'status': 'success'
        }
        
        self.training_history.append(adaptation_result)
        return adaptation_result

