"""
Uncertainty Sampler
===================

Implements various uncertainty-based sampling strategies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from ..config import ActiveLearningConfig
from ..enums import UncertaintyMeasure

logger = logging.getLogger(__name__)

class UncertaintySampler:
    """Uncertainty-based sampling"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.uncertainty_history = []
        logger.info("✅ Uncertainty Sampler initialized")
    
    def sample_uncertain(self, model: nn.Module, unlabeled_data: np.ndarray, 
                        n_samples: int = None) -> np.ndarray:
        """Sample most uncertain points"""
        logger.info(f"🎯 Sampling uncertain points using measure: {self.config.uncertainty_measure.value}")
        
        if n_samples is None:
            n_samples = self.config.n_query_samples
        
        # Calculate uncertainties
        uncertainties = self._calculate_uncertainties(model, unlabeled_data)
        
        # Select most uncertain points
        # argsort gives asc, so last n are max
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        uncertain_samples = unlabeled_data[uncertain_indices]
        
        # Store uncertainty history
        self.uncertainty_history.append({
            'uncertainties': uncertainties,
            'selected_indices': uncertain_indices,
            'uncertainty_measure': self.config.uncertainty_measure.value
        })
        
        return uncertain_samples
    
    def _calculate_uncertainties(self, model: nn.Module, data: np.ndarray) -> np.ndarray:
        """Calculate uncertainties for data points"""
        model.eval()
        
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            outputs = model(data_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            measure = self.config.uncertainty_measure
            
            if measure == UncertaintyMeasure.ENTROPY:
                u = self._calculate_entropy(probabilities)
            elif measure == UncertaintyMeasure.MARGIN:
                u = self._calculate_margin(probabilities)
            elif measure == UncertaintyMeasure.LEAST_CONFIDENT:
                u = self._calculate_least_confident(probabilities)
            elif measure == UncertaintyMeasure.VARIANCE:
                u = self._calculate_variance(probabilities)
            elif measure == UncertaintyMeasure.BALD:
                u = self._calculate_bald(probabilities)
            else:
                u = self._calculate_entropy(probabilities)
            
            return u.numpy()
    
    def _calculate_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate entropy uncertainty: -sum(p * log(p))"""
        return -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
    
    def _calculate_margin(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate margin uncertainty: 1 - (p1 - p2)"""
        sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return 1.0 - margin
    
    def _calculate_least_confident(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate least confident uncertainty: 1 - max(p)"""
        max_probs, _ = torch.max(probabilities, dim=1)
        return 1.0 - max_probs
    
    def _calculate_variance(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate variance uncertainty"""
        return torch.var(probabilities, dim=1)
    
    def _calculate_bald(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate BALD uncertainty (simplified)"""
        return self._calculate_entropy(probabilities)

