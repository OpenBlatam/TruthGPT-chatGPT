"""
Elastic Weight Consolidation
===========================

Implementation of Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any
from .config import ContinualLearningConfig, CLStrategy

logger = logging.getLogger(__name__)

class EWC:
    """Elastic Weight Consolidation implementation"""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.fisher_information = {}
        self.optimal_params = {}
        self.training_history = []
        logger.info("✅ EWC initialized")
    
    def compute_fisher_information(self, model: nn.Module, data: torch.Tensor, 
                                 labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Fisher information matrix"""
        logger.info("🐟 Computing Fisher information matrix")
        
        model.train()
        fisher_info = {}
        
        # Compute gradients
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[param_name] = torch.zeros_like(param)
        
        # Compute Fisher information for each sample
        for i in range(data.shape[0]):
            model.zero_grad()
            
            # Forward pass
            output = model(data[i:i+1])
            loss = F.cross_entropy(output, labels[i:i+1])
            
            # Backward pass
            loss.backward()
            
            # Accumulate Fisher information
            for param_name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[param_name] += param.grad ** 2
        
        # Average Fisher information
        for param_name in fisher_info:
            fisher_info[param_name] /= data.shape[0]
        
        return fisher_info
    
    def update_fisher_information(self, model: nn.Module, data: torch.Tensor, 
                                labels: torch.Tensor):
        """Update Fisher information matrix"""
        logger.info("🔄 Updating Fisher information matrix")
        
        # Compute new Fisher information
        new_fisher_info = self.compute_fisher_information(model, data, labels)
        
        # Update existing Fisher information
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name in self.fisher_information:
                    # Combine with existing Fisher information
                    self.fisher_information[param_name] = \
                        self.config.ewc_importance * self.fisher_information[param_name] + \
                        new_fisher_info[param_name]
                else:
                    self.fisher_information[param_name] = new_fisher_info[param_name]
        
        # Store optimal parameters
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[param_name] = param.data.clone()
    
    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC loss"""
        ewc_loss = 0.0
        
        for param_name, param in model.named_parameters():
            if param.requires_grad and param_name in self.fisher_information:
                # EWC loss: lambda * Fisher * (theta - theta*)^2
                fisher_info = self.fisher_information[param_name]
                optimal_param = self.optimal_params[param_name]
                
                ewc_loss += (fisher_info * (param - optimal_param) ** 2).sum()
        
        return self.config.ewc_lambda * ewc_loss
    
    def train_with_ewc(self, model: nn.Module, data: torch.Tensor, 
                      labels: torch.Tensor, num_epochs: int = 10) -> Dict[str, Any]:
        """Train model with EWC"""
        logger.info("🏋️ Training with EWC")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        ewc_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ewc_loss = 0.0
            
            # Forward pass
            output = model(data)
            task_loss = criterion(output, labels)
            
            # EWC loss
            ewc_loss = self.compute_ewc_loss(model)
            
            # Total loss
            total_loss = task_loss + ewc_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += task_loss.item()
            epoch_ewc_loss += ewc_loss.item()
            
            training_losses.append(epoch_loss)
            ewc_losses.append(epoch_ewc_loss)
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Task Loss = {task_loss.item():.4f}, EWC Loss = {ewc_loss.item():.4f}")
        
        training_result = {
            'strategy': CLStrategy.EWC.value,
            'epochs': num_epochs,
            'training_losses': training_losses,
            'ewc_losses': ewc_losses,
            'final_task_loss': training_losses[-1],
            'final_ewc_loss': ewc_losses[-1],
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result

