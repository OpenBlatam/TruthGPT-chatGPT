"""
Unit tests for federated optimization
Tests distributed optimization, federated learning, and collaborative optimization
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestFederatedOptimization(unittest.TestCase):
    """Test suite for federated optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_federated_learning_framework(self):
        """Test federated learning framework"""
        class FederatedLearningFramework:
            def __init__(self, n_clients=5, aggregation_method='fedavg'):
                self.n_clients = n_clients
                self.aggregation_method = aggregation_method
                self.clients = []
                self.global_model = None
                self.aggregation_history = []
                self.communication_rounds = 0
                
            def initialize_clients(self, model_template):
                """Initialize federated clients"""
                self.clients = []
                for i in range(self.n_clients):
                    client = {
                        'id': i,
                        'model': self._copy_model(model_template),
                        'data_size': np.random.randint(100, 1000),
                        'local_updates': [],
                        'participation_rate': np.random.uniform(0.5, 1.0)
                    }
                    self.clients.append(client)
                    
            def _copy_model(self, model):
                """Create model copy for client"""
                # Simulate model copying
                return model
                
            def federated_round(self, global_model, participation_threshold=0.5):
                """Execute one federated learning round"""
                # Select participating clients
                participating_clients = self._select_participating_clients(participation_threshold)
                
                # Local training on participating clients
                local_updates = []
                for client in participating_clients:
                    local_update = self._local_training(client, global_model)
                    local_updates.append(local_update)
                    
                # Aggregate local updates
                aggregated_update = self._aggregate_updates(local_updates)
                
                # Update global model
                updated_global_model = self._update_global_model(global_model, aggregated_update)
                
                # Record aggregation
                self.aggregation_history.append({
                    'round': self.communication_rounds,
                    'participating_clients': len(participating_clients),
                    'aggregation_method': self.aggregation_method,
                    'update_norm': np.linalg.norm(aggregated_update) if isinstance(aggregated_update, np.ndarray) else 0
                })
                
                self.communication_rounds += 1
                return updated_global_model
                
            def _select_participating_clients(self, threshold):
                """Select clients for participation"""
                participating = []
                for client in self.clients:
                    if client['participation_rate'] >= threshold:
                        participating.append(client)
                return participating
                
            def _local_training(self, client, global_model):
                """Simulate local training on client"""
                # Simulate local training
                local_update = {
                    'client_id': client['id'],
                    'data_size': client['data_size'],
                    'local_loss': np.random.uniform(0, 1),
                    'gradients': np.random.uniform(-1, 1, 10),
                    'training_time': np.random.uniform(1, 10)
                }
                
                client['local_updates'].append(local_update)
                return local_update
                
            def _aggregate_updates(self, local_updates):
                """Aggregate local updates"""
                if self.aggregation_method == 'fedavg':
                    return self._fedavg_aggregation(local_updates)
                elif self.aggregation_method == 'fedprox':
                    return self._fedprox_aggregation(local_updates)
                else:
                    return self._simple_aggregation(local_updates)
                    
            def _fedavg_aggregation(self, local_updates):
                """Federated averaging aggregation"""
                if not local_updates:
                    return np.zeros(10)
                    
                # Weight by data size
                total_data_size = sum(update['data_size'] for update in local_updates)
                weighted_gradients = []
                
                for update in local_updates:
                    weight = update['data_size'] / total_data_size
                    weighted_gradients.append(weight * update['gradients'])
                    
                return np.sum(weighted_gradients, axis=0)
                
            def _fedprox_aggregation(self, local_updates):
                """FedProx aggregation with proximal term"""
                if not local_updates:
                    return np.zeros(10)
                    
                # Similar to FedAvg but with proximal regularization
                fedavg_update = self._fedavg_aggregation(local_updates)
                
                # Add proximal term (simplified)
                proximal_term = 0.01 * fedavg_update
                return fedavg_update + proximal_term
                
            def _simple_aggregation(self, local_updates):
                """Simple averaging aggregation"""
                if not local_updates:
                    return np.zeros(10)
                    
                return np.mean([update['gradients'] for update in local_updates], axis=0)
                
            def _update_global_model(self, global_model, aggregated_update):
                """Update global model with aggregated update"""
                # Simulate global model update
                return global_model
                
            def get_federated_stats(self):
                """Get federated learning statistics"""
                return {
                    'total_clients': self.n_clients,
                    'communication_rounds': self.communication_rounds,
                    'aggregation_method': self.aggregation_method,
                    'total_aggregations': len(self.aggregation_history),
                    'avg_participating_clients': np.mean([agg['participating_clients'] for agg in self.aggregation_history]) if self.aggregation_history else 0
                }
        
        # Test federated learning framework
        framework = FederatedLearningFramework(n_clients=5, aggregation_method='fedavg')
        model_template = nn.Linear(256, 512)
        
        # Initialize clients
        framework.initialize_clients(model_template)
        self.assertEqual(len(framework.clients), 5)
        
        # Test federated round
        global_model = nn.Linear(256, 512)
        updated_model = framework.federated_round(global_model)
        
        # Verify results
        self.assertIsNotNone(updated_model)
        self.assertEqual(framework.communication_rounds, 1)
        self.assertEqual(len(framework.aggregation_history), 1)
        
        # Check federated stats
        stats = framework.get_federated_stats()
        self.assertEqual(stats['total_clients'], 5)
        self.assertEqual(stats['communication_rounds'], 1)
        self.assertEqual(stats['aggregation_method'], 'fedavg')
        self.assertEqual(stats['total_aggregations'], 1)
        
    def test_distributed_optimization(self):
        """Test distributed optimization"""
        class DistributedOptimizer:
            def __init__(self, n_workers=4, synchronization_method='allreduce'):
                self.n_workers = n_workers
                self.synchronization_method = synchronization_method
                self.workers = []
                self.optimization_history = []
                self.communication_overhead = 0
                
            def initialize_workers(self, model_template):
                """Initialize distributed workers"""
                self.workers = []
                for i in range(self.n_workers):
                    worker = {
                        'id': i,
                        'model': self._copy_model(model_template),
                        'local_gradients': None,
                        'communication_delay': np.random.uniform(0.1, 1.0)
                    }
                    self.workers.append(worker)
                    
            def _copy_model(self, model):
                """Create model copy for worker"""
                return model
                
            def distributed_optimization_step(self, global_model, data, target):
                """Execute distributed optimization step"""
                # Distribute data to workers
                worker_data = self._distribute_data(data, target)
                
                # Compute gradients on each worker
                worker_gradients = []
                for i, worker in enumerate(self.workers):
                    gradients = self._compute_worker_gradients(worker, worker_data[i])
                    worker_gradients.append(gradients)
                    
                # Synchronize gradients
                synchronized_gradients = self._synchronize_gradients(worker_gradients)
                
                # Update global model
                updated_model = self._update_global_model(global_model, synchronized_gradients)
                
                # Record optimization step
                self.optimization_history.append({
                    'step': len(self.optimization_history),
                    'workers_used': self.n_workers,
                    'synchronization_method': self.synchronization_method,
                    'communication_overhead': self.communication_overhead
                })
                
                return updated_model
                
            def _distribute_data(self, data, target):
                """Distribute data to workers"""
                batch_size = data.shape[0] // self.n_workers
                worker_data = []
                
                for i in range(self.n_workers):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size if i < self.n_workers - 1 else data.shape[0]
                    
                    worker_data.append({
                        'data': data[start_idx:end_idx],
                        'target': target[start_idx:end_idx]
                    })
                    
                return worker_data
                
            def _compute_worker_gradients(self, worker, worker_data):
                """Compute gradients on worker"""
                # Simulate gradient computation
                gradients = np.random.uniform(-1, 1, 10)
                
                # Simulate communication delay
                time.sleep(worker['communication_delay'])
                self.communication_overhead += worker['communication_delay']
                
                return gradients
                
            def _synchronize_gradients(self, worker_gradients):
                """Synchronize gradients across workers"""
                if self.synchronization_method == 'allreduce':
                    return self._allreduce_synchronization(worker_gradients)
                elif self.synchronization_method == 'parameter_server':
                    return self._parameter_server_synchronization(worker_gradients)
                else:
                    return self._simple_synchronization(worker_gradients)
                    
            def _allreduce_synchronization(self, worker_gradients):
                """AllReduce synchronization"""
                # Simulate AllReduce operation
                return np.mean(worker_gradients, axis=0)
                
            def _parameter_server_synchronization(self, worker_gradients):
                """Parameter server synchronization"""
                # Simulate parameter server aggregation
                return np.mean(worker_gradients, axis=0)
                
            def _simple_synchronization(self, worker_gradients):
                """Simple gradient synchronization"""
                return np.mean(worker_gradients, axis=0)
                
            def _update_global_model(self, global_model, synchronized_gradients):
                """Update global model with synchronized gradients"""
                # Simulate global model update
                return global_model
                
            def get_distributed_stats(self):
                """Get distributed optimization statistics"""
                return {
                    'total_workers': self.n_workers,
                    'synchronization_method': self.synchronization_method,
                    'total_steps': len(self.optimization_history),
                    'communication_overhead': self.communication_overhead,
                    'avg_communication_per_step': self.communication_overhead / len(self.optimization_history) if self.optimization_history else 0
                }
        
        # Test distributed optimization
        optimizer = DistributedOptimizer(n_workers=4, synchronization_method='allreduce')
        model_template = nn.Linear(256, 512)
        
        # Initialize workers
        optimizer.initialize_workers(model_template)
        self.assertEqual(len(optimizer.workers), 4)
        
        # Test distributed optimization step
        global_model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=8, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        updated_model = optimizer.distributed_optimization_step(global_model, data, target)
        
        # Verify results
        self.assertIsNotNone(updated_model)
        self.assertEqual(len(optimizer.optimization_history), 1)
        self.assertGreater(optimizer.communication_overhead, 0)
        
        # Check distributed stats
        stats = optimizer.get_distributed_stats()
        self.assertEqual(stats['total_workers'], 4)
        self.assertEqual(stats['synchronization_method'], 'allreduce')
        self.assertEqual(stats['total_steps'], 1)
        self.assertGreater(stats['communication_overhead'], 0)
        
    def test_collaborative_optimization(self):
        """Test collaborative optimization"""
        class CollaborativeOptimizer:
            def __init__(self, n_collaborators=3, collaboration_method='consensus'):
                self.n_collaborators = n_collaborators
                self.collaboration_method = collaboration_method
                self.collaborators = []
                self.consensus_history = []
                self.collaboration_rounds = 0
                
            def initialize_collaborators(self, optimization_problem):
                """Initialize collaborative optimizers"""
                self.collaborators = []
                for i in range(self.n_collaborators):
                    collaborator = {
                        'id': i,
                        'local_solution': np.random.uniform(-5, 5, 5),
                        'confidence': np.random.uniform(0.5, 1.0),
                        'expertise': np.random.uniform(0.3, 1.0),
                        'communication_radius': np.random.uniform(0.1, 1.0)
                    }
                    self.collaborators.append(collaborator)
                    
            def collaborative_optimization_round(self, optimization_problem):
                """Execute one collaborative optimization round"""
                # Local optimization on each collaborator
                local_solutions = []
                for collaborator in self.collaborators:
                    local_solution = self._local_optimization(collaborator, optimization_problem)
                    local_solutions.append(local_solution)
                    
                # Collaborative consensus
                consensus_solution = self._reach_consensus(local_solutions)
                
                # Update collaborator solutions
                for i, collaborator in enumerate(self.collaborators):
                    collaborator['local_solution'] = consensus_solution
                    
                # Record consensus
                self.consensus_history.append({
                    'round': self.collaboration_rounds,
                    'consensus_solution': consensus_solution,
                    'consensus_quality': self._evaluate_consensus_quality(local_solutions, consensus_solution)
                })
                
                self.collaboration_rounds += 1
                return consensus_solution
                
            def _local_optimization(self, collaborator, optimization_problem):
                """Local optimization on collaborator"""
                # Simulate local optimization
                local_solution = collaborator['local_solution'] + np.random.normal(0, 0.1, 5)
                return local_solution
                
            def _reach_consensus(self, local_solutions):
                """Reach consensus among collaborators"""
                if self.collaboration_method == 'consensus':
                    return self._consensus_method(local_solutions)
                elif self.collaboration_method == 'weighted_average':
                    return self._weighted_average_method(local_solutions)
                else:
                    return self._simple_average_method(local_solutions)
                    
            def _consensus_method(self, local_solutions):
                """Consensus-based collaboration"""
                # Weight by confidence and expertise
                weights = []
                for i, collaborator in enumerate(self.collaborators):
                    weight = collaborator['confidence'] * collaborator['expertise']
                    weights.append(weight)
                    
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                consensus = np.zeros_like(local_solutions[0])
                for i, solution in enumerate(local_solutions):
                    consensus += weights[i] * solution
                    
                return consensus
                
            def _weighted_average_method(self, local_solutions):
                """Weighted average collaboration"""
                # Weight by expertise
                weights = [collaborator['expertise'] for collaborator in self.collaborators]
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                consensus = np.zeros_like(local_solutions[0])
                for i, solution in enumerate(local_solutions):
                    consensus += weights[i] * solution
                    
                return consensus
                
            def _simple_average_method(self, local_solutions):
                """Simple averaging collaboration"""
                return np.mean(local_solutions, axis=0)
                
            def _evaluate_consensus_quality(self, local_solutions, consensus_solution):
                """Evaluate consensus quality"""
                # Calculate variance from consensus
                variances = [np.var(solution - consensus_solution) for solution in local_solutions]
                return 1.0 / (1.0 + np.mean(variances))
                
            def get_collaboration_stats(self):
                """Get collaboration statistics"""
                return {
                    'total_collaborators': self.n_collaborators,
                    'collaboration_rounds': self.collaboration_rounds,
                    'collaboration_method': self.collaboration_method,
                    'total_consensus': len(self.consensus_history),
                    'avg_consensus_quality': np.mean([c['consensus_quality'] for c in self.consensus_history]) if self.consensus_history else 0
                }
        
        # Test collaborative optimization
        optimizer = CollaborativeOptimizer(n_collaborators=3, collaboration_method='consensus')
        optimization_problem = {'objective': lambda x: np.sum(x**2)}
        
        # Initialize collaborators
        optimizer.initialize_collaborators(optimization_problem)
        self.assertEqual(len(optimizer.collaborators), 3)
        
        # Test collaborative optimization round
        consensus_solution = optimizer.collaborative_optimization_round(optimization_problem)
        
        # Verify results
        self.assertIsNotNone(consensus_solution)
        self.assertEqual(len(consensus_solution), 5)
        self.assertEqual(optimizer.collaboration_rounds, 1)
        self.assertEqual(len(optimizer.consensus_history), 1)
        
        # Check collaboration stats
        stats = optimizer.get_collaboration_stats()
        self.assertEqual(stats['total_collaborators'], 3)
        self.assertEqual(stats['collaboration_rounds'], 1)
        self.assertEqual(stats['collaboration_method'], 'consensus')
        self.assertEqual(stats['total_consensus'], 1)
        self.assertGreater(stats['avg_consensus_quality'], 0)

class TestDistributedOptimization(unittest.TestCase):
    """Test suite for distributed optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_parameter_server_optimization(self):
        """Test parameter server optimization"""
        class ParameterServerOptimizer:
            def __init__(self, n_workers=4, server_learning_rate=0.01):
                self.n_workers = n_workers
                self.server_learning_rate = server_learning_rate
                self.parameter_server = None
                self.workers = []
                self.optimization_history = []
                self.server_updates = 0
                
            def initialize_parameter_server(self, model_template):
                """Initialize parameter server"""
                self.parameter_server = {
                    'global_parameters': np.random.uniform(-1, 1, 10),
                    'parameter_history': [],
                    'update_count': 0
                }
                
            def initialize_workers(self, model_template):
                """Initialize workers"""
                self.workers = []
                for i in range(self.n_workers):
                    worker = {
                        'id': i,
                        'local_parameters': np.random.uniform(-1, 1, 10),
                        'gradients': None,
                        'update_count': 0
                    }
                    self.workers.append(worker)
                    
            def parameter_server_step(self, worker_id, gradients):
                """Execute parameter server step"""
                # Update global parameters
                self.parameter_server['global_parameters'] -= self.server_learning_rate * gradients
                self.parameter_server['update_count'] += 1
                self.server_updates += 1
                
                # Record parameter update
                self.parameter_server['parameter_history'].append({
                    'update': self.parameter_server['update_count'],
                    'parameters': self.parameter_server['global_parameters'].copy(),
                    'gradients': gradients.copy()
                })
                
                # Update worker parameters
                self.workers[worker_id]['local_parameters'] = self.parameter_server['global_parameters'].copy()
                self.workers[worker_id]['update_count'] += 1
                
                return self.parameter_server['global_parameters']
                
            def worker_step(self, worker_id, local_data, local_target):
                """Execute worker step"""
                worker = self.workers[worker_id]
                
                # Compute local gradients
                local_gradients = self._compute_local_gradients(worker, local_data, local_target)
                worker['gradients'] = local_gradients
                
                # Send gradients to parameter server
                updated_parameters = self.parameter_server_step(worker_id, local_gradients)
                
                return updated_parameters
                
            def _compute_local_gradients(self, worker, data, target):
                """Compute local gradients"""
                # Simulate gradient computation
                gradients = np.random.uniform(-1, 1, 10)
                return gradients
                
            def get_parameter_server_stats(self):
                """Get parameter server statistics"""
                return {
                    'total_workers': self.n_workers,
                    'server_updates': self.server_updates,
                    'parameter_history_length': len(self.parameter_server['parameter_history']),
                    'avg_worker_updates': np.mean([worker['update_count'] for worker in self.workers])
                }
        
        # Test parameter server optimization
        optimizer = ParameterServerOptimizer(n_workers=4, server_learning_rate=0.01)
        model_template = nn.Linear(256, 512)
        
        # Initialize parameter server and workers
        optimizer.initialize_parameter_server(model_template)
        optimizer.initialize_workers(model_template)
        
        self.assertIsNotNone(optimizer.parameter_server)
        self.assertEqual(len(optimizer.workers), 4)
        
        # Test worker step
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        updated_parameters = optimizer.worker_step(0, data, target)
        
        # Verify results
        self.assertIsNotNone(updated_parameters)
        self.assertEqual(len(updated_parameters), 10)
        self.assertEqual(optimizer.server_updates, 1)
        self.assertEqual(optimizer.workers[0]['update_count'], 1)
        
        # Check parameter server stats
        stats = optimizer.get_parameter_server_stats()
        self.assertEqual(stats['total_workers'], 4)
        self.assertEqual(stats['server_updates'], 1)
        self.assertEqual(stats['parameter_history_length'], 1)
        self.assertGreater(stats['avg_worker_updates'], 0)

if __name__ == '__main__':
    unittest.main()





