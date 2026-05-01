"""
Active Learning System
======================

Main integrator for the active learning process.
"""
import time
import logging
import numpy as np
import torch.nn as nn
from typing import Dict, Any, List, Optional, Callable

from .config import ActiveLearningConfig
from .enums import ActiveLearningStrategy
from .samplers import (
    UncertaintySampler, 
    DiversitySampler, 
    QueryByCommittee, 
    ExpectedModelChange, 
    BatchActiveLearning
)

logger = logging.getLogger(__name__)

class ActiveLearningSystem:
    """Main active learning system engine."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        
        # Strategies
        self.uncertainty_sampler = UncertaintySampler(config)
        self.diversity_sampler = DiversitySampler(config)
        self.query_by_committee = QueryByCommittee(config)
        self.expected_model_change = ExpectedModelChange(config)
        self.batch_active_learning = BatchActiveLearning(config)
        
        # State
        self.history = []
        self.labeled_data = []
        self.labeled_labels = []
        self.unlabeled_data = []
        
        logger.info("✅ Active Learning System initialized")
    
    def run_active_learning(self, model: nn.Module, initial_data: np.ndarray, 
                           initial_labels: np.ndarray, unlabeled_data: np.ndarray,
                           query_function: Callable = None) -> Dict[str, Any]:
        """Execute the iterative active learning loop."""
        logger.info(f"🚀 Running active learning with strategy: {self.config.active_learning_strategy.value}")
        
        start_time = time.time()
        self.labeled_data = initial_data.copy()
        self.labeled_labels = initial_labels.copy()
        self.unlabeled_data = unlabeled_data.copy()
        
        active_results = {
            'start_time': start_time,
            'config': self.config,
            'iterations': []
        }
        
        for iteration in range(self.config.max_iterations):
            if len(self.unlabeled_data) < self.config.n_query_samples:
                logger.info("⏹️ Pool exhausted")
                break
                
            strategy = self.config.active_learning_strategy
            
            # 1. Selection Stage
            if strategy == ActiveLearningStrategy.UNCERTAINTY_SAMPLING:
                queried = self.uncertainty_sampler.sample_uncertain(model, self.unlabeled_data)
            elif strategy == ActiveLearningStrategy.DIVERSITY_SAMPLING:
                queried = self.diversity_sampler.sample_diverse(self.unlabeled_data, self.labeled_data)
            elif strategy == ActiveLearningStrategy.QUERY_BY_COMMITTEE:
                if not self.query_by_committee.committee_models:
                    self.query_by_committee.create_committee(model)
                queried = self.query_by_committee.query_by_committee(self.unlabeled_data)
            elif strategy == ActiveLearningStrategy.EXPECTED_MODEL_CHANGE:
                queried = self.expected_model_change.query_expected_model_change(
                    model, self.unlabeled_data, self.labeled_data, self.labeled_labels
                )
            elif strategy == ActiveLearningStrategy.BATCH_ACTIVE_LEARNING:
                queried = self.batch_active_learning.query_batch(model, self.unlabeled_data, self.labeled_data)
            else:
                queried = self.uncertainty_sampler.sample_uncertain(model, self.unlabeled_data)
            
            # 2. Oracle Stage (Labeling)
            if query_function:
                new_labels = query_function(queried)
            else:
                new_labels = np.random.randint(0, 10, len(queried))
            
            # 3. Update Stage
            self.labeled_data = np.vstack([self.labeled_data, queried])
            self.labeled_labels = np.append(self.labeled_labels, new_labels)
            
            # Remove from pool (inefficient lookup, assuming manageable size)
            indices_to_remove = []
            for q in queried:
                for idx, pool_item in enumerate(self.unlabeled_data):
                    if np.allclose(q, pool_item):
                        indices_to_remove.append(idx)
                        break
            self.unlabeled_data = np.delete(self.unlabeled_data, indices_to_remove, axis=0)
            
            active_results['iterations'].append({
                'id': iteration,
                'queried_size': len(queried),
                'total_labeled': len(self.labeled_data)
            })
            
        active_results['end_time'] = time.time()
        active_results['duration'] = active_results['end_time'] - start_time
        self.history.append(active_results)
        return active_results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Create a human-readable summary of the AL runs."""
        lines = ["Active Learning Summary Report", "=========================="]
        lines.append(f"Strategy: {self.config.active_learning_strategy.value}")
        lines.append(f"Iterations: {len(results.get('iterations', []))}")
        lines.append(f"Final pool size: {len(self.labeled_data)}")
        lines.append(f"Processing time: {results.get('duration', 0):.2f}s")
        return "\n".join(lines)

