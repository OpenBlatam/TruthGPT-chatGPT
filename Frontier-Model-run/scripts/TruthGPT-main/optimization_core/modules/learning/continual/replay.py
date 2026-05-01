"""
Replay Buffer
=============

Experience replay buffer for continual learning.
"""
import torch
import torch.nn as nn
import logging
import random
from typing import Dict, Any, Tuple
from .config import ContinualLearningConfig, CLStrategy

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """Replay buffer implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.buffer = []
        self.buffer_labels = []
        self.current_size = 0
        self.training_history = []
        logger.info("✅ Replay Buffer initialized")
    
    def add_samples(self, data: torch.Tensor, labels: torch.Tensor):
        """Add samples to replay buffer"""
        logger.info(f"📝 Adding {data.shape[0]} samples to replay buffer")
        
        for i in range(data.shape[0]):
            if self.current_size < self.config.replay_buffer_size:
                # Add new sample
                self.buffer.append(data[i].clone())
                self.buffer_labels.append(labels[i].item())
                self.current_size += 1
            else:
                # Replace random sample
                idx = random.randint(0, self.config.replay_buffer_size - 1)
                self.buffer[idx] = data[i].clone()
                self.buffer_labels[idx] = labels[i].item()
    
    def get_samples(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get samples from replay buffer"""
        if self.current_size == 0:
            return torch.tensor([]), torch.tensor([])
        
        # Sample indices
        if num_samples >= self.current_size:
            indices = list(range(self.current_size))
        else:
            indices = random.sample(range(self.current_size), num_samples)
        
        # Get samples
        samples = torch.stack([self.buffer[i] for i in indices])
        labels = torch.tensor([self.buffer_labels[i] for i in indices])
        
        return samples, labels
    
    def train_with_replay(self, model: nn.Module, new_data: torch.Tensor, 
                         new_labels: torch.Tensor, num_epochs: int = 10) -> Dict[str, Any]:
        """Train model with replay"""
        logger.info("🔄 Training with replay")
        
        # Add new samples to buffer
        self.add_samples(new_data, new_labels)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        replay_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_replay_loss = 0.0
            
            # Train on new data
            output = model(new_data)
            task_loss = criterion(output, new_labels)
            epoch_loss += task_loss.item()
            
            # Train on replay data
            if self.current_size > 0:
                replay_data, replay_labels = self.get_samples(self.config.replay_batch_size)
                
                if replay_data.shape[0] > 0:
                    replay_output = model(replay_data)
                    replay_loss = criterion(replay_output, replay_labels)
                    epoch_replay_loss += replay_loss.item()
                    
                    # Combined loss
                    total_loss = task_loss + replay_loss
                else:
                    total_loss = task_loss
            else:
                total_loss = task_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            training_losses.append(epoch_loss)
            replay_losses.append(epoch_replay_loss)
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Task Loss = {task_loss.item():.4f}, Replay Loss = {epoch_replay_loss:.4f}")
        
        training_result = {
            'strategy': CLStrategy.REPLAY_BUFFER.value,
            'epochs': num_epochs,
            'training_losses': training_losses,
            'replay_losses': replay_losses,
            'final_task_loss': training_losses[-1],
            'final_replay_loss': replay_losses[-1],
            'buffer_size': self.current_size,
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result

