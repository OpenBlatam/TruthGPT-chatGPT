"""
Learning Algorithms Module
===========================

Collection of advanced learning algorithms for optimization core.
Includes evolutionary computing, causal inference, and more.
"""

from __future__ import annotations
from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS = {
    # Evolutionary Computing
    'EvolutionaryOptimizer': '.evolutionary',
    'EvolutionaryConfig': '.evolutionary',
    'EvolutionaryAlgorithm': '.evolutionary',
    'create_evolutionary_config': '.evolutionary',
    'create_individual': '.evolutionary',
    'create_population': '.evolutionary',
    'create_evolutionary_optimizer': '.evolutionary',
    'evolutionary_computing': '.evolutionary',
    
    # Causal Inference
    'CausalInferenceEngine': '.causal',
    'CausalConfig': '.causal',
    'CausalInferenceSystem': '.causal',
    'CausalMethod': '.causal',
    'CausalEffectType': '.causal',
    'causal_inference': '.causal',
    
    # Active Learning
    'ActiveLearningStrategy': '.active',
    'ActiveLearningConfig': '.active',
    'ActiveLearningSampler': '.active',
    'ActiveLearningSystem': '.active',
    'active_learning': '.active',
    
    # Adaptive Learning
    'AdaptiveLearningSystem': '.adaptive',
    'AdaptiveLearningConfig': '.adaptive',
    'adaptive_learning': '.adaptive',
    
    # Adversarial Learning
    'AdversarialLearningSystem': '.adversarial',
    'AdversarialConfig': '.adversarial',
    'AdversarialAttackType': '.adversarial',
    'GANType': '.adversarial',
    'DefenseStrategy': '.adversarial',
    'create_adversarial_learning_system': '.adversarial',
    'create_adversarial_config': '.adversarial',
    'create_adversarial_attacker': '.adversarial',
    'create_gan_trainer': '.adversarial',
    'create_adversarial_defense': '.adversarial',
    'create_robustness_analyzer': '.adversarial',
    'adversarial_learning': '.adversarial',
    
    # Bayesian Optimization (Legacy mapping to new hpo package or bayesian subpackage)
    # The implementation plan says bayesian_optimization.py -> bayesian/
    'BayesianOptimizer': '.bayesian',
    'BayesianConfig': '.bayesian',
    'bayesian_optimization': '.bayesian',
    
    # Continual Learning
    'ContinualLearner': '.continual',
    'ContinualConfig': '.continual',
    'continual_learning': '.continual',
    
    # Ensemble Learning
    'EnsembleManager': '.ensemble',
    'EnsembleConfig': '.ensemble',
    'ensemble_learning': '.ensemble',
    
    # Federated Learning
    'FederatedServer': '.federated',
    'FederatedClient': '.federated',
    'FederatedConfig': '.federated',
    'federated_learning': '.federated',
    
    # Hyperparameter Optimization
    'HyperparameterOptimizer': '.hpo',
    'HPOConfig': '.hpo',
    'hyperparameter_optimization': '.hpo',
    
    # Meta Learning
    'MetaLearner': '.meta',
    'MetaConfig': '.meta',
    'meta_learning': '.meta',
    
    # Multitask Learning
    'MultitaskModel': '.multitask',
    'MultitaskConfig': '.multitask',
    'multitask_learning': '.multitask',
    
    # NAS
    'NeuralArchitectureSearch': '.nas',
    'NASConfig': '.nas',
    'nas': '.nas',
    
    # Reinforcement Learning
    'RLAgent': '.reinforcement',
    'RLConfig': '.reinforcement',
    'RLBuffer': '.reinforcement',
    'RLSystem': '.reinforcement',
    'reinforcement_learning': '.reinforcement',
    
    # Self-Supervised Learning
    'SelfSupervisedTrainer': '.self_supervised',
    'SelfSupervisedConfig': '.self_supervised',
    'self_supervised_learning': '.self_supervised',
    
    # Transfer Learning
    'TransferLearningManager': '.transfer',
    'TransferLearningConfig': '.transfer',
    'transfer_learning': '.transfer',
}

def __getattr__(name: str):
    """Lazy import system for learning module."""
    return resolve_lazy_import(name, __package__ or 'learning', _LAZY_IMPORTS)

__all__ = list(_LAZY_IMPORTS.keys())

