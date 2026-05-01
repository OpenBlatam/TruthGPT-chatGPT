"""
Differentiable NAS
==================

Differentiable Neural Architecture Search implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any
from .config import NASConfig

logger = logging.getLogger(__name__)

class DifferentiableNAS:
    """Differentiable Neural Architecture Search"""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.supernet = None
        self.architecture_weights = None
        
        logger.info("✅ Differentiable NAS initialized")
    
    def build_supernet(self, input_size: int = 128, output_size: int = 10):
        """Build supernet containing all possible operations"""
        # This is a simplified implementation
        # In practice, this would be much more sophisticated
        
        self.supernet = nn.ModuleDict({
            'linear_32': nn.Linear(input_size, 32),
            'linear_64': nn.Linear(input_size, 64),
            'linear_128': nn.Linear(input_size, 128),
            'linear_256': nn.Linear(input_size, 256),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'dropout': nn.Dropout(0.1)
        })
        
        # Initialize architecture weights
        num_ops = len(self.supernet)
        self.architecture_weights = nn.Parameter(torch.randn(num_ops))
        
        logger.info(f"✅ Supernet built with {num_ops} operations")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through supernet"""
        if self.supernet is None:
            raise ValueError("Supernet not built. Call build_supernet() first.")
        
        # Apply softmax to architecture weights
        weights = torch.softmax(self.architecture_weights, dim=0)
        
        # Weighted combination of operations
        outputs = []
        for i, (name, module) in enumerate(self.supernet.items()):
            if 'linear' in name:
                output = module(x)
                outputs.append(weights[i] * output)
        
        # Combine outputs
        if outputs:
            combined = torch.stack(outputs, dim=0).sum(dim=0)
        else:
            combined = x
        
        # Apply activations
        combined = self.supernet['relu'](combined)
        
        return combined
    
    def search(self, train_loader, val_loader, epochs: int = 100) -> Dict[str, Any]:
        """Perform differentiable architecture search"""
        logger.info("🚀 Starting Differentiable NAS...")
        
        if self.supernet is None:
            self.build_supernet()
        
        optimizer = optim.Adam(self.supernet.parameters(), lr=0.001)
        arch_optimizer = optim.Adam([self.architecture_weights], lr=0.01)
        
        best_accuracy = 0.0
        search_history = []
        
        for epoch in range(epochs):
            # Train architecture weights
            self.supernet.train()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                arch_optimizer.zero_grad()
                
                output = self.forward(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                loss.backward()
                optimizer.step()
                arch_optimizer.step()
            
            # Validate
            self.supernet.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.forward(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = correct / total
            best_accuracy = max(best_accuracy, accuracy)
            
            search_history.append({
                'epoch': epoch,
                'accuracy': accuracy,
                'best_accuracy': best_accuracy
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Accuracy = {accuracy:.4f}")
        
        logger.info(f"✅ Differentiable NAS completed. Best accuracy: {best_accuracy:.4f}")
        
        return {
            'best_accuracy': best_accuracy,
            'search_history': search_history,
            'final_architecture_weights': self.architecture_weights.detach().cpu().numpy()
        }

