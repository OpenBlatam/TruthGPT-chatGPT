"""
Ensemble Learning Package
=========================

Advanced ensemble learning systems with Voting, Stacking, Bagging, and Boosting.
"""
from .enums import EnsembleStrategy, VotingStrategy, BoostingMethod
from .config import EnsembleConfig
from .base import BaseModel
from .voting import VotingEnsemble
from .stacking import StackingEnsemble
from .bagging import BaggingEnsemble
from .boosting import BoostingEnsemble
from .dynamic import DynamicEnsemble
from .system import EnsembleTrainer

# Compatibility aliases
EnsembleManager = EnsembleTrainer

# Factory functions
def create_ensemble_config(**kwargs) -> EnsembleConfig:
    return EnsembleConfig(**kwargs)

def create_base_model(model_id: int, model_type: str, config: EnsembleConfig) -> BaseModel:
    return BaseModel(model_id, model_type, config)

def create_voting_ensemble(config: EnsembleConfig) -> VotingEnsemble:
    return VotingEnsemble(config)

def create_stacking_ensemble(config: EnsembleConfig) -> StackingEnsemble:
    return StackingEnsemble(config)

def create_bagging_ensemble(config: EnsembleConfig) -> BaggingEnsemble:
    return BaggingEnsemble(config)

def create_boosting_ensemble(config: EnsembleConfig) -> BoostingEnsemble:
    return BoostingEnsemble(config)

def create_dynamic_ensemble(config: EnsembleConfig) -> DynamicEnsemble:
    return DynamicEnsemble(config)

def create_ensemble_trainer(config: EnsembleConfig) -> EnsembleTrainer:
    return EnsembleTrainer(config)

__all__ = [
    'EnsembleStrategy',
    'VotingStrategy',
    'BoostingMethod',
    'EnsembleConfig',
    'BaseModel',
    'VotingEnsemble',
    'StackingEnsemble',
    'BaggingEnsemble',
    'BoostingEnsemble',
    'DynamicEnsemble',
    'EnsembleTrainer',
    'create_ensemble_config',
    'create_base_model',
    'create_voting_ensemble',
    'create_stacking_ensemble',
    'create_bagging_ensemble',
    'create_boosting_ensemble',
    'create_dynamic_ensemble',
    'create_ensemble_trainer'
]

