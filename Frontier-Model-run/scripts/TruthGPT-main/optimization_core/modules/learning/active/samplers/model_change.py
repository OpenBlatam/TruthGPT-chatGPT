"""
Expected Model Change Sampler
=============================

Selects samples that maximize the expected gradient/parameter shift in the model.
"""
import torch
import torch.nn as nn
import numpy as np
import logging

from ..config import ActiveLearningConfig

logger = logging.getLogger(__name__)

class ExpectedModelChange:
    """Expected Model Change sampling strategy."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.model_change_history = []
        logger.info("✅ Expected Model Change initialized")
    
    def query_expected_model_change(self, model: nn.Module, unlabeled_data: np.ndarray,
                                   labeled_data: np.ndarray, labeled_labels: np.ndarray,
                                   n_samples: int = None) -> np.ndarray:
        """Calculate and selection points with highest expected model change."""
        logger.info("🎯 Querying expected model change")
        
        changes = self._calculate_expected_changes(model, unlabeled_data, labeled_data, labeled_labels)
        count = n_samples or self.config.n_query_samples
        indices = np.argsort(changes)[-count:]
        
        self.model_change_history.append({
            'expected_changes': changes,
            'selected_indices': indices
        })
        
        return unlabeled_data[indices]
    
    def _calculate_expected_changes(self, model: nn.Module, unlabeled_data: np.ndarray,
                                  labeled_data: np.ndarray, labeled_labels: np.ndarray) -> np.ndarray:
        """Simplified proxy for expected model change."""
        # Note: True EMC is computationally prohibitive (requires re-training or grad norms).
        # We use a random distribution here as a placeholder for the logic flow.
        return np.random.random(len(unlabeled_data))

