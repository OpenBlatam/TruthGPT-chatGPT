"""
Adversarial Learning Configuration
==================================

Configuration dataclasses for adversarial attacks and defenses.
"""
from dataclasses import dataclass, field
from .enums import AdversarialAttackType, GANType, DefenseStrategy

@dataclass
class AdversarialConfig:
    """Configuration for adversarial learning system"""
    # Basic settings
    attack_type: AdversarialAttackType = AdversarialAttackType.FGSM
    gan_type: GANType = GANType.VANILLA_GAN
    defense_strategy: DefenseStrategy = DefenseStrategy.ADVERSARIAL_TRAINING
    
    # Attack settings
    attack_epsilon: float = 0.1
    attack_alpha: float = 0.01
    attack_iterations: int = 10
    attack_norm: str = "inf"
    attack_targeted: bool = False
    
    # GAN settings
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    gan_beta1: float = 0.5
    gan_beta2: float = 0.999
    gan_latent_dim: int = 100
    
    # Defense settings
    defense_epsilon: float = 0.1
    defense_alpha: float = 0.01
    defense_iterations: int = 10
    defense_norm: str = "inf"
    
    # Training settings
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 0.001
    
    # Advanced features
    enable_robustness_analysis: bool = True
    enable_attack_generation: bool = True
    enable_defense_training: bool = True
    enable_adversarial_training: bool = True
    
    def __post_init__(self):
        """Validate adversarial configuration"""
        if self.attack_epsilon <= 0:
            raise ValueError("Attack epsilon must be positive")
        if self.attack_alpha <= 0:
            raise ValueError("Attack alpha must be positive")
        if self.attack_iterations <= 0:
            raise ValueError("Attack iterations must be positive")
        if self.generator_lr <= 0:
            raise ValueError("Generator learning rate must be positive")
        if self.discriminator_lr <= 0:
            raise ValueError("Discriminator learning rate must be positive")
        if self.gan_latent_dim <= 0:
            raise ValueError("GAN latent dimension must be positive")
        if self.defense_epsilon <= 0:
            raise ValueError("Defense epsilon must be positive")
        if self.defense_alpha <= 0:
            raise ValueError("Defense alpha must be positive")
        if self.defense_iterations <= 0:
            raise ValueError("Defense iterations must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

