"""
GAN Trainer
===========

Orchestrates the training of GAN generators and discriminators.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from typing import Dict, Any

from ..config import AdversarialConfig
from .generator import GANGeneratorCreator
from .discriminator import GANDiscriminatorCreator

logger = logging.getLogger(__name__)

class GANTrainer:
    """Manages GAN training iterations and losses."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.gen_creator = GANGeneratorCreator(config)
        self.disc_creator = GANDiscriminatorCreator(config)
        self.history = []
        logger.info("✅ GAN Trainer initialized")
        
    def train_gan(self, real_data: torch.Tensor) -> Dict[str, Any]:
        """Train GAN on real data distribution."""
        logger.info(f"🚀 Training {self.config.gan_type.value} GAN")
        
        in_dim = real_data.shape[1]
        generator = self.gen_creator.create_generator(self.config.gan_latent_dim, in_dim)
        discriminator = self.disc_creator.create_discriminator(in_dim)
        
        g_opt = torch.optim.Adam(generator.parameters(), lr=self.config.generator_lr, 
                                betas=(self.config.gan_beta1, self.config.gan_beta2))
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=self.config.discriminator_lr, 
                                betas=(self.config.gan_beta1, self.config.gan_beta2))
        
        epochs_data = []
        for epoch in range(self.config.num_epochs):
            # 1. Train Discriminator
            d_loss = self._step_discriminator(discriminator, generator, real_data, d_opt)
            
            # 2. Train Generator
            g_loss = self._step_generator(discriminator, generator, g_opt)
            
            epochs_data.append({'epoch': epoch, 'd_loss': d_loss, 'g_loss': g_loss})
            if epoch % 20 == 0:
                logger.debug(f"GAN Epoch {epoch}: D={d_loss:.4f}, G={g_loss:.4f}")
                
        return {'epochs': epochs_data, 'status': 'success'}

    def _step_discriminator(self, d, g, real, opt) -> float:
        d.train()
        g.eval()
        bs = real.shape[0]
        
        # Real
        r_out = d(real)
        r_loss = F.binary_cross_entropy(r_out, torch.ones(bs, 1))
        
        # Fake
        noise = torch.randn(bs, self.config.gan_latent_dim)
        f_data = g(noise)
        f_out = d(f_data.detach())
        f_loss = F.binary_cross_entropy(f_out, torch.zeros(bs, 1))
        
        loss = r_loss + f_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def _step_generator(self, d, g, opt) -> float:
        g.train()
        d.eval()
        bs = self.config.batch_size
        
        noise = torch.randn(bs, self.config.gan_latent_dim)
        f_data = g(noise)
        out = d(f_data)
        
        loss = F.binary_cross_entropy(out, torch.ones(bs, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

