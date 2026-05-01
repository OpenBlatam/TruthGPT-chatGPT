"""
Generative & Representation Learning
====================================

Generative models and representation learning implementation.
"""
import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict, Any
from .config import SSLConfig

logger = logging.getLogger(__name__)

class RepresentationLearner:
    """Representation learning implementation"""
    
    def __init__(self, config: SSLConfig):
        self.config = config
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.training_history = []
        logger.info("✅ Representation Learner initialized")
    
    def _create_encoder(self) -> nn.Module:
        """Create encoder network"""
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.config.encoder_dim)
        )
        return encoder
    
    def _create_decoder(self) -> nn.Module:
        """Create decoder network"""
        decoder = nn.Sequential(
            nn.Linear(self.config.encoder_dim, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.Tanh()
        )
        return decoder
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Encode
        features = self.encoder(x)
        
        # Decode
        reconstruction = self.decoder(features)
        
        return features, reconstruction
    
    def train_representation(self, data: torch.Tensor) -> Dict[str, Any]:
        """Train representation learning"""
        logger.info("🧠 Training representation learning")
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        
        # Training loop
        self.encoder.train()
        self.decoder.train()
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()
            
            features, reconstruction = self.forward(data)
            loss = criterion(reconstruction, data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
        
        training_result = {
            'total_loss': total_loss,
            'epochs': self.config.num_epochs,
            'status': 'success'
        }
        
        return training_result

