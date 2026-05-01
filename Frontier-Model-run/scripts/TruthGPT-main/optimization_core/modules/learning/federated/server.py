"""
Federated Server
================

Server-side implementation for federated learning.
"""
import time
import torch
import torch.nn as nn
import random
import logging
from typing import Dict, Any, List
from .config import FederatedLearningConfig, ClientSelectionStrategy
from .client import FederatedClient

logger = logging.getLogger(__name__)

class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, global_model: nn.Module, config: FederatedLearningConfig):
        self.global_model = global_model
        self.config = config
        self.clients = {}
        self.global_updates = []
        self.aggregation_history = []
        logger.info("✅ Federated Server initialized")
    
    def add_client(self, client: FederatedClient):
        """Add client to federated learning"""
        self.clients[client.client_id] = client
        logger.info(f"➕ Added client {client.client_id}")
    
    def select_clients(self, round_num: int) -> List[FederatedClient]:
        """Select clients for current round"""
        logger.info(f"🎯 Selecting clients for round {round_num}")
        
        if self.config.client_selection_strategy == ClientSelectionStrategy.RANDOM:
            selected_clients = random.sample(list(self.clients.values()), 
                                          min(self.config.clients_per_round, len(self.clients)))
        elif self.config.client_selection_strategy == ClientSelectionStrategy.ROUND_ROBIN:
            client_list = list(self.clients.values())
            start_idx = round_num % len(client_list)
            selected_clients = client_list[start_idx:start_idx + self.config.clients_per_round]
            if len(selected_clients) < self.config.clients_per_round:
                selected_clients.extend(client_list[:self.config.clients_per_round - len(selected_clients)])
        else:
            # Default to random selection
            selected_clients = random.sample(list(self.clients.values()), 
                                          min(self.config.clients_per_round, len(self.clients)))
        
        logger.info(f"📋 Selected {len(selected_clients)} clients")
        return selected_clients
    
    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]], 
                         client_weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """Aggregate client updates"""
        logger.info("🔄 Aggregating client updates")
        
        if client_weights is None:
            client_weights = [1.0] * len(client_updates)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated update
        aggregated_update = {}
        for name in client_updates[0].keys():
            aggregated_update[name] = torch.zeros_like(client_updates[0][name])
        
        # Weighted aggregation
        for update, weight in zip(client_updates, client_weights):
            for name, param_update in update.items():
                aggregated_update[name] += weight * param_update
        
        aggregation_result = {
            'method': self.config.aggregation_method.value,
            'num_clients': len(client_updates),
            'aggregation_time': time.time(),
            'status': 'success'
        }
        
        self.aggregation_history.append(aggregation_result)
        return aggregated_update
    
    def update_global_model(self, aggregated_update: Dict[str, torch.Tensor]):
        """Update global model with aggregated update"""
        logger.info("🌐 Updating global model")
        
        # Apply aggregated update to global model
        for name, param in self.global_model.named_parameters():
            if name in aggregated_update:
                param.data += aggregated_update[name]
        
        # Store global update
        self.global_updates.append(aggregated_update)

class AsyncFederatedServer(FederatedServer):
    """Asynchronous federated learning server"""
    
    def __init__(self, global_model: nn.Module, config: FederatedLearningConfig):
        super().__init__(global_model, config)
        self.async_updates = {}
        logger.info("✅ Async Federated Server initialized")
    
    def receive_async_update(self, client_id: str, update: Dict[str, torch.Tensor]):
        """Receive asynchronous update from client"""
        logger.info(f"📨 Received async update from client {client_id}")
        
        self.async_updates[client_id] = {
            'update': update,
            'timestamp': time.time(),
            'round': len(self.global_updates)
        }
        
        # Process update if enough clients have sent updates
        if len(self.async_updates) >= self.config.clients_per_round:
            self._process_async_updates()
    
    def _process_async_updates(self):
        """Process asynchronous updates"""
        logger.info("⚡ Processing async updates")
        
        # Extract updates and weights
        updates = []
        weights = []
        
        for client_id, update_info in self.async_updates.items():
            updates.append(update_info['update'])
            weights.append(1.0)  # Equal weights for simplicity
        
        # Aggregate updates
        aggregated_update = self.aggregate_updates(updates, weights)
        
        # Update global model
        self.update_global_model(aggregated_update)
        
        # Clear processed updates
        self.async_updates.clear()

