"""
Ensemble Learning Configuration
===============================

Configuration dataclasses for ensemble learning systems.
"""
from dataclasses import dataclass, field
from typing import List
from .enums import EnsembleStrategy, VotingStrategy, BoostingMethod

@dataclass
class EnsembleConfig:
    """Configuration for ensemble learning system"""
    # Basic settings
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.VOTING_ENSEMBLE
    voting_strategy: VotingStrategy = VotingStrategy.SOFT_VOTING
    boosting_method: BoostingMethod = BoostingMethod.GRADIENT_BOOSTING
    
    # Model settings
    num_models: int = 5
    model_types: List[str] = field(default_factory=lambda: ["neural_network", "random_forest", "svm"])
    model_diversity: float = 0.8
    
    # Voting settings
    enable_weighted_voting: bool = True
    weight_learning_rate: float = 0.01
    
    # Stacking settings
    meta_learner_type: str = "logistic_regression"
    cross_validation_folds: int = 5
    
    # Bagging settings
    bootstrap_ratio: float = 0.8
    feature_sampling_ratio: float = 0.8
    
    # Boosting settings
    boosting_iterations: int = 100
    learning_rate: float = 0.1
    
    # Advanced features
    enable_dynamic_weighting: bool = True
    enable_model_selection: bool = True
    enable_uncertainty_estimation: bool = True
    
    def __post_init__(self):
        """Validate ensemble configuration"""
        if self.num_models <= 0:
            raise ValueError("Number of models must be positive")
        if not (0 <= self.model_diversity <= 1):
            raise ValueError("Model diversity must be between 0 and 1")
        if not (0 <= self.weight_learning_rate <= 1):
            raise ValueError("Weight learning rate must be between 0 and 1")
        if self.cross_validation_folds <= 0:
            raise ValueError("Cross-validation folds must be positive")
        if not (0 <= self.bootstrap_ratio <= 1):
            raise ValueError("Bootstrap ratio must be between 0 and 1")
        if not (0 <= self.feature_sampling_ratio <= 1):
            raise ValueError("Feature sampling ratio must be between 0 and 1")
        if self.boosting_iterations <= 0:
            raise ValueError("Boosting iterations must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

