"""
Query By Committee Sampler
==========================

Implements sampling by training a committee of models and querying disagreements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Optional

from ..config import ActiveLearningConfig

logger = logging.getLogger(__name__)

class QueryByCommittee:
    """Query by committee disagreement implementation."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.committee_models: List[nn.Module] = []
        self.committee_history = []
        logger.info("✅ Query by Committee initialized")
    
    def create_committee(self, base_model: nn.Module, n_members: int = None) -> List[nn.Module]:
        """Instantiate committee members with weight perturbations."""
        n = n_members or self.config.n_committee_members
        logger.info(f"👥 Creating committee with {n} members")
        
        committee = []
        for i in range(n):
            # Assumes model class has no-arg constructor or can be cloned
            model_copy = type(base_model)()
            model_copy.load_state_dict(base_model.state_dict())
            # Perturb weights
            for param in model_copy.parameters():
                param.data += torch.randn_like(param) * 0.01
            committee.append(model_copy)
            
        self.committee_models = committee
        return committee
    
    def query_by_committee(self, unlabeled_data: np.ndarray, n_samples: int = None) -> np.ndarray:
        """Select samples where committee members disagree most."""
        if not self.committee_models:
            logger.warning("No committee exists. Falling back to random.")
            indices = np.random.choice(len(unlabeled_data), n_samples or 10, replace=False)
            return unlabeled_data[indices]
            
        disagreements = self._calculate_disagreements(unlabeled_data)
        count = n_samples or self.config.n_query_samples
        indices = np.argsort(disagreements)[-count:]
        
        self.committee_history.append({
            'disagreements': disagreements,
            'selected_indices': indices,
            'n_committee_members': len(self.committee_models)
        })
        
        return unlabeled_data[indices]
    
    def _calculate_disagreements(self, data: np.ndarray) -> np.ndarray:
        """Disagreement quantised by prediction variance."""
        preds = []
        data_tensor = torch.FloatTensor(data)
        
        for m in self.committee_models:
            m.eval()
            with torch.no_grad():
                probs = F.softmax(m(data_tensor), dim=1)
                preds.append(probs)
                
        preds_tensor = torch.stack(preds) # [members, samples, classes]
        # Sum of variance across class dimensions for each sample
        disagreement = torch.var(preds_tensor, dim=0).sum(dim=1)
        return disagreement.numpy()

