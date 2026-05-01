"""
Batch Active Learning Sampler
=============================

Weighted combination of uncertainty and diversity for batch selection.
"""
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

from ..config import ActiveLearningConfig
from .uncertainty import UncertaintySampler

logger = logging.getLogger(__name__)

class BatchActiveLearning:
    """Batch-oriented active learning query strategy."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.batch_history = []
        self.uncertainty_sampler = UncertaintySampler(config)
        logger.info("✅ Batch Active Learning initialized")
    
    def query_batch(self, model, unlabeled_data: np.ndarray,
                   labeled_data: np.ndarray = None, n_samples: int = None) -> np.ndarray:
        """Query a batch of points balancing informativeness and diversity."""
        logger.info(f"🎯 Querying batch of {n_samples or self.config.batch_size} samples")
        
        count = n_samples or self.config.batch_size
        
        # Informativeness (Uncertainty)
        u_scores = self.uncertainty_sampler._calculate_uncertainties(model, unlabeled_data)
        
        # Diversity (Distance to labeled set)
        d_scores = self._calculate_diversity_scores(unlabeled_data, labeled_data)
        
        # Combined objective: weighted sum
        w = self.config.batch_diversity_weight
        scores = (w * d_scores) + ((1.0 - w) * u_scores)
        
        indices = np.argsort(scores)[-count:]
        
        self.batch_history.append({
            'uncertainties': u_scores,
            'diversity': d_scores,
            'combined': scores,
            'selected_indices': indices
        })
        
        return unlabeled_data[indices]
    
    def _calculate_diversity_scores(self, unlabeled_data: np.ndarray, 
                                   labeled_data: np.ndarray = None) -> np.ndarray:
        """Diversity measured as distance to nearest labeled point."""
        if labeled_data is None or len(labeled_data) == 0:
            return np.ones(len(unlabeled_data))
            
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(labeled_data)
        dists, _ = nn.kneighbors(unlabeled_data)
        dists = dists.flatten()
        # Normalize to [0, 1] range
        if dists.max() > 0:
            dists = dists / dists.max()
        return dists

