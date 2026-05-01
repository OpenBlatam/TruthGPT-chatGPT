"""
Contrastive Learner
===================

Contrastive learning implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple
from .config import SSLConfig
from .enums import ContrastiveLossType

logger = logging.getLogger(__name__)

class ContrastiveLearner:
    """Contrastive learning implementation"""
    
    def __init__(self, config: SSLConfig):
        self.config = config
        self.encoder = self._create_encoder()
        self.projector = self._create_projector()
        self.memory_bank = None
        self.training_history = []
        logger.info("✅ Contrastive Learner initialized")
    
    def _create_encoder(self) -> nn.Module:
        """Create encoder network"""
        # Simplified encoder creation - in real scenario this would be more flexible
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
    
    def _create_projector(self) -> nn.Module:
        """Create projector network"""
        projector = nn.Sequential(
            nn.Linear(self.config.encoder_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.projection_dim)
        )
        return projector
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Encode
        features = self.encoder(x)
        # Project
        projections = self.projector(features)
        return features, projections
    
    def compute_contrastive_loss(self, projections: torch.Tensor, 
                               labels: torch.Tensor = None) -> torch.Tensor:
        """Compute contrastive loss"""
        # batch_size = projections.shape[0] # Unused variable
        
        if self.config.contrastive_loss == ContrastiveLossType.INFO_NCE:
            return self._compute_info_nce_loss(projections)
        elif self.config.contrastive_loss == ContrastiveLossType.NT_XENT:
            return self._compute_nt_xent_loss(projections)
        elif self.config.contrastive_loss == ContrastiveLossType.TRIPLET_LOSS:
            return self._compute_triplet_loss(projections, labels)
        else:
            return self._compute_contrastive_loss(projections)
    
    def _compute_info_nce_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss"""
        batch_size = projections.shape[0]
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T)
        
        # Create positive pairs (diagonal)
        positive_pairs = torch.diag(similarity_matrix)
        
        # Compute InfoNCE loss - simplified implementation
        # Ideally we mask self-similarity and use true positives vs negatives
        # Here we follow the logic from original file
        numerator = torch.exp(positive_pairs / self.config.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.config.temperature), dim=1)
        
        loss = -torch.log(numerator / denominator).mean()
        
        return loss
    
    def _compute_nt_xent_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss"""
        # Same as InfoNCE in this implementation context
        return self._compute_info_nce_loss(projections)
    
    def _compute_triplet_loss(self, projections: torch.Tensor, 
                            labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss"""
        if labels is None:
            return torch.tensor(0.0)
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute pairwise distances
        # distances = torch.cdist(projections, projections) # Unused variable
        distances = torch.cdist(projections, projections)
        
        # Create triplets
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~positive_mask
        
        # Compute triplet loss
        anchor_positive_dist = distances[positive_mask]
        anchor_negative_dist = distances[negative_mask]
        
        # Handle case where masks might result in different sizes or empty tensors
        # Simplified: using mean of available distances
        if anchor_positive_dist.numel() > 0 and anchor_negative_dist.numel() > 0:
            pos_dist = anchor_positive_dist.mean()
            neg_dist = anchor_negative_dist.mean()
            margin = 1.0
            loss = F.relu(pos_dist - neg_dist + margin)
        else:
            loss = torch.tensor(0.0)
        
        return loss
    
    def _compute_contrastive_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """Compute standard contrastive loss"""
        batch_size = projections.shape[0]
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T)
        
        # Create positive pairs (diagonal)
        positive_pairs = torch.diag(similarity_matrix)
        
        # Create negative pairs (off-diagonal)
        negative_pairs = similarity_matrix[~torch.eye(batch_size, dtype=bool)]
        
        # Compute contrastive loss
        positive_loss = -positive_pairs.mean()
        negative_loss = negative_pairs.mean()
        
        loss = positive_loss + negative_loss
        
        return loss

