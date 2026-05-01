"""
Active Learning Configuration
=============================

Configuration settings for the active learning system.
"""
from dataclasses import dataclass, field
from .enums import ActiveLearningStrategy, UncertaintyMeasure, QueryStrategy

@dataclass
class ActiveLearningConfig:
    """Configuration for active learning system"""
    # Basic settings
    active_learning_strategy: ActiveLearningStrategy = ActiveLearningStrategy.UNCERTAINTY_SAMPLING
    uncertainty_measure: UncertaintyMeasure = UncertaintyMeasure.ENTROPY
    query_strategy: QueryStrategy = QueryStrategy.UNCERTAINTY_BASED
    
    # Sampling settings
    n_initial_samples: int = 100
    n_query_samples: int = 10
    n_total_samples: int = 1000
    max_iterations: int = 50
    
    # Uncertainty sampling settings
    uncertainty_threshold: float = 0.5
    entropy_threshold: float = 0.8
    margin_threshold: float = 0.1
    
    # Diversity sampling settings
    diversity_method: str = "kmeans"
    n_clusters: int = 10
    diversity_weight: float = 0.5
    
    # Query by committee settings
    n_committee_members: int = 5
    disagreement_threshold: float = 0.3
    
    # Batch active learning settings
    batch_size: int = 20
    batch_diversity_weight: float = 0.3
    
    # Advanced features
    enable_adaptive_sampling: bool = True
    enable_cost_sensitive_sampling: bool = False
    enable_online_learning: bool = True
    enable_model_uncertainty: bool = True
    
    def __post_init__(self):
        """Validate active learning configuration"""
        if self.n_initial_samples <= 0:
            raise ValueError("Number of initial samples must be positive")
        if self.n_query_samples <= 0:
            raise ValueError("Number of query samples must be positive")
        if self.n_total_samples <= 0:
            raise ValueError("Number of total samples must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Maximum iterations must be positive")
        if not (0 <= self.uncertainty_threshold <= 1):
            raise ValueError("Uncertainty threshold must be between 0 and 1")
        if not (0 <= self.entropy_threshold <= 1):
            raise ValueError("Entropy threshold must be between 0 and 1")
        if not (0 <= self.margin_threshold <= 1):
            raise ValueError("Margin threshold must be between 0 and 1")
        if self.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
        if not (0 <= self.diversity_weight <= 1):
            raise ValueError("Diversity weight must be between 0 and 1")
        if self.n_committee_members <= 0:
            raise ValueError("Number of committee members must be positive")
        if not (0 <= self.disagreement_threshold <= 1):
            raise ValueError("Disagreement threshold must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not (0 <= self.batch_diversity_weight <= 1):
            raise ValueError("Batch diversity weight must be between 0 and 1")

