"""
GAN Discriminator
=================

Discriminator/Critic networks for adversarial data validation.
"""
import torch
import torch.nn as nn
import logging
from ..enums import GANType
from ..config import AdversarialConfig

logger = logging.getLogger(__name__)

class GANDiscriminatorCreator:
    """Factory for GAN Discriminator networks."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.discriminator = None
        logger.info("✅ GAN Discriminator initialized")
    
    def create_discriminator(self, input_dim: int) -> nn.Module:
        """Create a discriminator based on config type."""
        d_type = self.config.gan_type
        logger.info(f"🏗️ Creating {d_type.value} discriminator")
        
        if d_type == GANType.VANILLA_GAN:
            net = self._vanilla(input_dim)
        elif d_type == GANType.DCGAN:
            net = self._dcgan(input_dim)
        elif d_type == GANType.WGAN:
            net = self._wgan(input_dim)
        else:
            net = self._vanilla(input_dim)
            
        self.discriminator = net
        return net
    
    def _vanilla(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _dcgan(self, in_dim: int) -> nn.Module:
        return self._vanilla(in_dim)
    
    def _wgan(self, in_dim: int) -> nn.Module:
        # WGAN Critic doesn't use Sigmoid
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

