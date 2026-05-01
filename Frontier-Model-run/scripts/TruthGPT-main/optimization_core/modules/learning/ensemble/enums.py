"""
Ensemble Learning Enums
=======================

Enums for ensemble strategies, voting methods, and boosting types.
"""
from enum import Enum

class EnsembleStrategy(Enum):
    """Ensemble learning strategies"""
    VOTING_ENSEMBLE = "voting_ensemble"
    STACKING_ENSEMBLE = "stacking_ensemble"
    BAGGING_ENSEMBLE = "bagging_ensemble"
    BOOSTING_ENSEMBLE = "boosting_ensemble"
    DYNAMIC_ENSEMBLE = "dynamic_ensemble"
    NEURAL_ENSEMBLE = "neural_ensemble"

class VotingStrategy(Enum):
    """Voting strategies"""
    HARD_VOTING = "hard_voting"
    SOFT_VOTING = "soft_voting"
    WEIGHTED_VOTING = "weighted_voting"
    CONFIDENCE_VOTING = "confidence_voting"

class BoostingMethod(Enum):
    """Boosting methods"""
    ADABOOST = "adaboost"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"

