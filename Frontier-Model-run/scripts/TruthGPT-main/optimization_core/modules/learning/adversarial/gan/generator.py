"""
GAN Generator
=============

Generator networks for synthetic adversarial data generation.
"""
import torch
import torch.nn as nn
import logging
from ..enums import GANType
from ..config import AdversarialConfig

logger = logging.getLogger(__name__)

class GANGeneratorCreator:
    """Factory for GAN Generator networks."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.generator = None
        logger.info("✅ GAN Generator initialized")
    
    def create_generator(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create a generator based on config type."""
        g_type = self.config.gan_type
        logger.info(f"🏗️ Creating {g_type.value} generator")
        
        if g_type == GANType.VANILLA_GAN:
            net = self._vanilla(input_dim, output_dim)
        elif g_type == GANType.DCGAN:
            net = self._dcgan(input_dim, output_dim)
        elif g_type == GANType.WGAN:
            net = self._wgan(input_dim, output_dim)
        else:
            net = self._vanilla(input_dim, output_dim)
            
        self.generator = net
        return net
    
    def _vanilla(self, latent_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.Tanh()
        )
    
    def _dcgan(self, latent_dim: int, out_dim: int) -> nn.Module:
        # Simplified DCGAN using Linear (assuming flattened input for generic engine)
        return self._vanilla(latent_dim, out_dim)
    
    def _wgan(self, latent_dim: int, out_dim: int) -> nn.Module:
        return self._vanilla(latent_dim, out_dim)
    
    def generate_samples(self, n: int) -> torch.Tensor:
        """Generate N synthetic samples from latent space."""
        if self.generator is None:
            raise RuntimeError("Generator not created")
            
        noise = torch.randn(n, self.config.gan_latent_dim)
        with torch.no_grad():
            return self.generator(noise)

