"""
Federated Learning Module
Advanced federated learning with privacy and security
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import random
import time

logger = logging.getLogger(__name__)

class FederatedStrategy(Enum):
    """Federated learning strategies"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAGRAD = "fedadagrad"
    FEDYOGI = "fedyogi"
    FEDADAM = "fedadam"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"

@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    strategy: FederatedStrategy = FederatedStrategy.FEDAVG
    num_clients: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    use_differential_privacy: bool = False
    noise_multiplier: float = 1.0
    l2_norm_clip: float = 1.0
    use_secure_aggregation: bool = False
    use_compression: bool = False
    compression_ratio: float = 0.1

class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: int, model: nn.Module, config: FederatedConfig):
        self.client_id = client_id
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.data_size = 0
        self.training_history = []
    
    def train_local(self, data_loader, epochs: int) -> Dict[str, Any]:
        """Train model locally"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch in data_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch['input_ids'])
                loss = nn.CrossEntropyLoss()(outputs, batch['labels'])
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.config.use_differential_privacy:
                    self._apply_dp_noise()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Store training history
        avg_loss = total_loss / num_batches
        self.training_history.append(avg_loss)
        
        return {
            "client_id": self.client_id,
            "model_state": self.model.state_dict(),
            "data_size": self.data_size,
            "loss": avg_loss
        }
    
    def _apply_dp_noise(self):
        """Apply differential privacy noise"""
        for param in self.model.parameters():
            if param.grad is not None:
                # Clip gradients
                param.grad = torch.clamp(param.grad, -self.config.l2_norm_clip, self.config.l2_norm_clip)
                
                # Add noise
                noise = torch.normal(0, self.config.noise_multiplier * self.config.l2_norm_clip, 
                                   size=param.grad.shape, device=param.grad.device)
                param.grad += noise

class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, global_model: nn.Module, config: FederatedConfig):
        self.global_model = global_model
        self.config = config
        self.clients = []
        self.global_history = []
        self.round = 0
    
    def add_client(self, client: FederatedClient):
        """Add client to federation"""
        self.clients.append(client)
    
    def federated_averaging(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform federated averaging"""
        if not client_updates:
            return {}
        
        # Calculate total data size
        total_data_size = sum(update['data_size'] for update in client_updates)
        
        # Initialize aggregated model
        aggregated_state = {}
        for key in client_updates[0]['model_state'].keys():
            aggregated_state[key] = torch.zeros_like(client_updates[0]['model_state'][key])
        
        # Weighted averaging
        for update in client_updates:
            weight = update['data_size'] / total_data_size
            for key, param in update['model_state'].items():
                aggregated_state[key] += weight * param
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        # Calculate average loss
        avg_loss = sum(update['loss'] for update in client_updates) / len(client_updates)
        
        return {
            "round": self.round,
            "avg_loss": avg_loss,
            "num_clients": len(client_updates),
            "total_data_size": total_data_size
        }
    
    def run_federation(self, client_data_loaders: List[Any]) -> Dict[str, Any]:
        """Run federated learning"""
        logger.info(f"Starting federated learning with {len(self.clients)} clients")
        
        for round_num in range(self.config.num_rounds):
            self.round = round_num
            
            # Select clients for this round
            selected_clients = self._select_clients()
            
            # Train clients locally
            client_updates = []
            for client in selected_clients:
                data_loader = client_data_loaders[client.client_id]
                update = client.train_local(data_loader, self.config.local_epochs)
                client_updates.append(update)
            
            # Aggregate updates
            aggregation_result = self.federated_averaging(client_updates)
            self.global_history.append(aggregation_result)
            
            # Log progress
            if round_num % 10 == 0:
                logger.info(f"Round {round_num}: Avg Loss = {aggregation_result['avg_loss']:.4f}")
        
        return {
            "total_rounds": self.config.num_rounds,
            "final_loss": self.global_history[-1]['avg_loss'],
            "history": self.global_history
        }
    
    def _select_clients(self) -> List[FederatedClient]:
        """Select clients for current round"""
        # Simple random selection
        num_selected = min(len(self.clients), self.config.num_clients)
        return random.sample(self.clients, num_selected)

class SecureAggregation:
    """Secure aggregation for federated learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.secret_shares = {}
    
    def generate_secret_shares(self, num_clients: int) -> Dict[int, List[float]]:
        """Generate secret shares for secure aggregation"""
        # Simplified secret sharing
        shares = {}
        for client_id in range(num_clients):
            shares[client_id] = [random.random() for _ in range(10)]
        return shares
    
    def aggregate_securely(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Securely aggregate client updates"""
        # Simplified secure aggregation
        aggregated_state = {}
        for key in client_updates[0]['model_state'].keys():
            aggregated_state[key] = torch.zeros_like(client_updates[0]['model_state'][key])
        
        for update in client_updates:
            for key, param in update['model_state'].items():
                aggregated_state[key] += param
        
        # Average the aggregated parameters
        for key in aggregated_state:
            aggregated_state[key] /= len(client_updates)
        
        return aggregated_state

class DifferentialPrivacy:
    """Differential privacy for federated learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.privacy_budget = 0.0
    
    def add_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add differential privacy noise to gradients"""
        noise_scale = self.config.noise_multiplier * self.config.l2_norm_clip
        noise = torch.normal(0, noise_scale, size=gradients.shape, device=gradients.device)
        return gradients + noise
    
    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Clip gradients for differential privacy"""
        return torch.clamp(gradients, -self.config.l2_norm_clip, self.config.l2_norm_clip)
    
    def calculate_privacy_budget(self, num_rounds: int, num_clients: int) -> float:
        """Calculate privacy budget consumption"""
        # Simplified privacy budget calculation
        self.privacy_budget = num_rounds * num_clients * self.config.noise_multiplier
        return self.privacy_budget

# Factory functions
def create_federated_client(client_id: int, model: nn.Module, config: FederatedConfig) -> FederatedClient:
    """Create federated client"""
    return FederatedClient(client_id, model, config)

def create_federated_server(global_model: nn.Module, config: FederatedConfig) -> FederatedServer:
    """Create federated server"""
    return FederatedServer(global_model, config)

def create_federated_config(**kwargs) -> FederatedConfig:
    """Create federated configuration"""
    return FederatedConfig(**kwargs)

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = create_federated_config(
        strategy=FederatedStrategy.FEDAVG,
        num_clients=5,
        num_rounds=10,
        use_differential_privacy=True
    )
    
    # Create global model
    global_model = nn.Linear(10, 1)
    
    # Create federated server
    server = create_federated_server(global_model, config)
    
    # Add clients
    for i in range(5):
        client_model = nn.Linear(10, 1)
        client = create_federated_client(i, client_model, config)
        server.add_client(client)
    
    print("Federated learning system created successfully!")



