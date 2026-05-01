"""
Federated Learning System
=========================

Main orchestrator for federated learning workflows.
"""
import time
import logging
import torch
import torch.nn as nn
from typing import Dict, Any
from .config import FederatedLearningConfig
from .client import FederatedClient
from .server import FederatedServer, AsyncFederatedServer
from .privacy import PrivacyPreservation

logger = logging.getLogger(__name__)

class FederatedLearningSystem:
    """Main federated learning system"""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        
        # Components
        if config.enable_asynchronous_updates:
            self.server = AsyncFederatedServer(nn.Sequential(), config)
        else:
            self.server = FederatedServer(nn.Sequential(), config)
            
        self.privacy_preservation = PrivacyPreservation(config)
        
        # Federated learning state
        self.federated_history = []
        self.current_round = 0
        
        logger.info("✅ Federated Learning System initialized")
    
    def add_client(self, client_id: str, model: nn.Module, data: torch.Tensor, labels: torch.Tensor):
        """Add client to federated learning"""
        client = FederatedClient(client_id, model, self.config)
        client.set_local_data(data, labels)
        self.server.add_client(client)
        logger.info(f"➕ Added client {client_id} to federated learning")
    
    def run_federated_learning(self) -> Dict[str, Any]:
        """Run federated learning"""
        logger.info("🚀 Starting federated learning")
        
        federated_results = {
            'start_time': time.time(),
            'config': self.config,
            'rounds': []
        }
        
        # Federated learning rounds
        for round_num in range(self.config.num_rounds):
            logger.info(f"🔄 Starting round {round_num + 1}/{self.config.num_rounds}")
            
            round_result = self._run_federated_round(round_num)
            federated_results['rounds'].append(round_result)
            
            self.current_round = round_num + 1
        
        # Final evaluation
        federated_results['end_time'] = time.time()
        federated_results['total_duration'] = federated_results['end_time'] - federated_results['start_time']
        
        # Store results
        self.federated_history.append(federated_results)
        
        logger.info("✅ Federated learning completed")
        return federated_results
    
    def _run_federated_round(self, round_num: int) -> Dict[str, Any]:
        """Run single federated learning round"""
        round_start_time = time.time()
        
        # Select clients
        selected_clients = self.server.select_clients(round_num)
        
        # Local training
        client_updates = []
        client_weights = []
        training_results = []
        
        for client in selected_clients:
            training_result = client.local_training(self.server.global_model)
            training_results.append(training_result)
            
            # Get local update
            if client.local_updates:
                client_updates.append(client.local_updates[-1])
                client_weights.append(1.0)  # Equal weights
        
        # Apply privacy preservation
        if client_updates:
            private_updates = self.privacy_preservation.apply_privacy_preservation(client_updates)
            
            # Aggregate updates
            aggregated_update = self.server.aggregate_updates(private_updates, client_weights)
            
            # Update global model
            self.server.update_global_model(aggregated_update)
        
        round_end_time = time.time()
        
        round_result = {
            'round_number': round_num + 1,
            'selected_clients': [c.client_id for c in selected_clients],
            'training_results': training_results,
            'round_duration': round_end_time - round_start_time,
            'status': 'success'
        }
        
        return round_result
    
    def generate_federated_report(self, results: Dict[str, Any]) -> str:
        """Generate federated learning report"""
        report = []
        report.append("=" * 50)
        report.append("FEDERATED LEARNING REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nFEDERATED LEARNING CONFIGURATION:")
        report.append("-" * 35)
        report.append(f"Aggregation Method: {self.config.aggregation_method.value}")
        report.append(f"Client Selection Strategy: {self.config.client_selection_strategy.value}")
        
        # Results
        report.append("\nFEDERATED LEARNING RESULTS:")
        report.append("-" * 30)
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        
        return "\n".join(report)

