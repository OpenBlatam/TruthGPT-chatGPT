"""
Adversarial Learning Package
===========================

Tools for attack generation, GAN-based data augmentation, and robustness defense.
"""
from .enums import AdversarialAttackType, GANType, DefenseStrategy
from .config import AdversarialConfig
from .attacks import AdversarialAttacker
from .gan import GANTrainer
from .defense import AdversarialDefense
from .analysis import RobustnessAnalyzer
from .system import AdversarialLearningSystem

# Factory functions
def create_adversarial_config(**kwargs) -> AdversarialConfig:
    return AdversarialConfig(**kwargs)

def create_adversarial_learning_system(config: AdversarialConfig) -> AdversarialLearningSystem:
    return AdversarialLearningSystem(config)

__all__ = [
    'AdversarialAttackType',
    'GANType',
    'DefenseStrategy',
    'AdversarialConfig',
    'AdversarialAttacker',
    'GANTrainer',
    'AdversarialDefense',
    'RobustnessAnalyzer',
    'AdversarialLearningSystem',
    'create_adversarial_config',
    'create_adversarial_learning_system'
]

