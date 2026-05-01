"""
Federated Client
================

Client-side implementation for federated learning.
"""
import time
import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from .config import FederatedLearningConfig, PrivacyLevel

logger = logging.getLogger(__name__)

class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: str, model: nn.Module, config: FederatedLearningConfig):
        self.client_id = client_id
        self.model = model
        self.config = config
        self.local_data = None
        self.local_updates = []
        self.participation_history = []
        logger.info(f"✅ Federated Client {client_id} initialized")
    
    def set_local_data(self, data: torch.Tensor, labels: torch.Tensor):
        """Set local training data"""
        self.local_data = (data, labels)
        logger.info(f"📊 Client {self.client_id} data set: {len(data)} samples")
    
    def local_training(self, global_model: nn.Module) -> Dict[str, Any]:
        """Perform local training"""
        logger.info(f"🏋️ Client {self.client_id} starting local training")
        
        # Copy global model
        self.model.load_state_dict(global_model.state_dict())
        
        # Local training setup
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        local_losses = []
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Simulate batch training
            if self.local_data is not None:
                data, labels = self.local_data
                batch_size = min(self.config.batch_size, len(data))
                
                for i in range(0, len(data), batch_size):
                    batch_data = data[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = criterion(output, batch_labels)
                    loss.backward()
                    
                    # Apply differential privacy if enabled
                    if self.config.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
                        self._apply_differential_privacy()
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            local_losses.append(epoch_loss / max(num_batches, 1))
        
        # Calculate local update
        local_update = self._calculate_local_update(global_model)
        
        training_result = {
            'client_id': self.client_id,
            'local_epochs': self.config.local_epochs,
            'final_loss': local_losses[-1],
            'local_update_norm': torch.norm(torch.cat([p.flatten() for p in local_update.values()])).item(),
            'participation_time': time.time(),
            'status': 'success'
        }
        
        self.local_updates.append(local_update)
        self.participation_history.append(training_result)
        
        return training_result
    
    def _apply_differential_privacy(self):
        """Apply differential privacy to gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                # Clip gradients
                grad_norm = torch.norm(param.grad)
                if grad_norm > self.config.l2_norm_clip:
                    param.grad = param.grad * self.config.l2_norm_clip / grad_norm
                
                # Add noise
                noise = torch.normal(0, self.config.noise_multiplier * self.config.l2_norm_clip, 
                                  size=param.grad.shape, device=param.grad.device)
                param.grad += noise
    
    def _calculate_local_update(self, global_model: nn.Module) -> Dict[str, torch.Tensor]:
        """Calculate local model update"""
        local_update = {}
        
        for name, param in self.model.named_parameters():
            global_param = dict(global_model.named_parameters())[name]
            local_update[name] = param.data - global_param.data
        
        return local_update

