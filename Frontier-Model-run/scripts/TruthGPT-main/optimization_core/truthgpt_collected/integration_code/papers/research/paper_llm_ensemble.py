#!/usr/bin/env python3
"""
Majority Rules: LLM Ensemble is a Winning Approach
===================================================

Muestra que usar un ensamble ("ensemble") de LLMs mejora significativamente
el rendimiento frente a usar un solo modelo.

Técnica principal: Ensemble de múltiples LLMs con votación mayoritaria y combinación ponderada.

Basado en: arXiv paper (November 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMEnsembleConfig:
    """Configuración para LLM Ensemble."""
    hidden_dim: int = 512
    num_models: int = 3
    ensemble_method: str = "weighted"  # weighted, majority, average
    use_confidence_weighting: bool = True
    use_diversity_regularization: bool = True
    diversity_weight: float = 0.1


class LLMEnsemble(nn.Module):
    """
    Ensemble de múltiples modelos LLM.
    
    Técnica: Combina predicciones de múltiples modelos usando
    votación mayoritaria o combinación ponderada.
    """
    
    def __init__(self, config: LLMEnsembleConfig):
        super().__init__()
        assert config.num_models > 1, f"num_models must be > 1, got {config.num_models}"
        assert config.ensemble_method in ["weighted", "majority", "average"], \
            f"ensemble_method must be one of ['weighted', 'majority', 'average'], got {config.ensemble_method}"
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_models = config.num_models
        
        # Ensemble weights (learnable)
        if config.ensemble_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(config.num_models) / config.num_models)
        else:
            self.register_buffer('ensemble_weights', torch.ones(config.num_models) / config.num_models)
        
        # Confidence estimator (for confidence weighting)
        if config.use_confidence_weighting:
            self.confidence_estimator = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            nn.init.xavier_uniform_(self.confidence_estimator[0].weight)
            nn.init.xavier_uniform_(self.confidence_estimator[2].weight)
        else:
            self.confidence_estimator = None
        
        # Metrics
        self.register_buffer('ensemble_diversity', torch.tensor(0.0))
        self.register_buffer('avg_confidence', torch.tensor(0.5))
        self.register_buffer('model_agreement', torch.tensor(1.0))
        self.register_buffer('ensemble_variance', torch.tensor(0.0))
        self.register_buffer('weight_entropy', torch.tensor(0.0))
        self.register_buffer('consensus_strength', torch.tensor(1.0))
        
        logger.info(f"Initialized LLMEnsemble: {config.num_models} models, method={config.ensemble_method}")
    
    def _compute_diversity(self, predictions: List[torch.Tensor]) -> float:
        """
        Compute diversity between model predictions.
        
        Args:
            predictions: List of [batch, seq, hidden_dim] tensors
            
        Returns:
            diversity: Diversity score (higher = more diverse)
        """
        if len(predictions) < 2:
            return 0.0
        
        # Compute pairwise differences
        diversity_sum = 0.0
        count = 0
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                diff = (predictions[i] - predictions[j]).norm(dim=-1).mean().item()
                diversity_sum += diff
                count += 1
        
        diversity = diversity_sum / count if count > 0 else 0.0
        return diversity
    
    def _compute_agreement(self, predictions: List[torch.Tensor]) -> float:
        """
        Compute agreement between model predictions.
        
        Args:
            predictions: List of [batch, seq, hidden_dim] tensors
            
        Returns:
            agreement: Agreement score (higher = more agreement)
        """
        if len(predictions) < 2:
            return 1.0
        
        # Compute cosine similarity between predictions
        similarities = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # Flatten for similarity
                pred_i = predictions[i].view(-1, self.hidden_dim)
                pred_j = predictions[j].view(-1, self.hidden_dim)
                
                # Cosine similarity
                sim = F.cosine_similarity(pred_i, pred_j, dim=-1).mean().item()
                similarities.append(sim)
        
        agreement = sum(similarities) / len(similarities) if similarities else 1.0
        return agreement
    
    def forward(self, model_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Combine predictions from multiple models.
        
        Args:
            model_outputs: List of [batch, seq, hidden_dim] tensors from different models
            
        Returns:
            ensemble_output: [batch, seq, hidden_dim]
            metadata: Dict with ensemble info
        """
        assert len(model_outputs) == self.num_models, \
            f"Expected {self.num_models} model outputs, got {len(model_outputs)}"
        
        batch_size, seq_len, hidden_dim = model_outputs[0].shape
        
        # Compute diversity and agreement
        diversity = self._compute_diversity(model_outputs)
        agreement = self._compute_agreement(model_outputs)
        
        self.ensemble_diversity = 0.9 * self.ensemble_diversity + 0.1 * diversity
        self.model_agreement = 0.9 * self.model_agreement + 0.1 * agreement
        
        # Ensemble variance (variance of predictions)
        stacked = torch.stack(model_outputs, dim=0)  # [num_models, batch, seq, hidden_dim]
        variance = stacked.var(dim=0).mean().item()
        self.ensemble_variance = 0.9 * self.ensemble_variance + 0.1 * variance
        
        # Weight entropy (diversity of ensemble weights)
        if self.config.ensemble_method == "weighted":
            weight_entropy = -(self.ensemble_weights * torch.log(self.ensemble_weights + 1e-8)).sum().item()
            self.weight_entropy = 0.9 * self.weight_entropy + 0.1 * weight_entropy
        
        # Consensus strength (how strong is the agreement)
        consensus = agreement * (1.0 - variance / (variance + 1.0))
        self.consensus_strength = 0.9 * self.consensus_strength + 0.1 * consensus
        
        # Compute confidence weights if enabled
        if self.confidence_estimator is not None:
            confidences = []
            for output in model_outputs:
                # Use last token for confidence
                last_token = output[:, -1, :]  # [batch, hidden_dim]
                confidence = self.confidence_estimator(last_token).squeeze(-1)  # [batch]
                confidences.append(confidence)
            
            # Normalize confidences
            confidence_tensor = torch.stack(confidences, dim=0)  # [num_models, batch]
            confidence_weights = F.softmax(confidence_tensor, dim=0)  # [num_models, batch]
            
            avg_confidence = confidence_tensor.mean().item()
            self.avg_confidence = 0.9 * self.avg_confidence + 0.1 * avg_confidence
        else:
            confidence_weights = None
        
        # Combine predictions
        if self.config.ensemble_method == "weighted":
            # Weighted combination
            if confidence_weights is not None:
                # Use confidence-weighted combination
                weights = confidence_weights.unsqueeze(-1).unsqueeze(-1)  # [num_models, batch, 1, 1]
            else:
                # Use learned ensemble weights
                weights = F.softmax(self.ensemble_weights, dim=0)
                weights = weights.view(self.num_models, 1, 1, 1).expand(-1, batch_size, seq_len, hidden_dim)
            
            stacked = torch.stack(model_outputs, dim=0)  # [num_models, batch, seq, hidden_dim]
            if confidence_weights is not None:
                ensemble_output = (stacked * weights).sum(dim=0)  # [batch, seq, hidden_dim]
            else:
                ensemble_output = (stacked * weights).sum(dim=0)
        
        elif self.config.ensemble_method == "majority":
            # Majority voting (for discrete outputs, here we use mean as proxy)
            stacked = torch.stack(model_outputs, dim=0)  # [num_models, batch, seq, hidden_dim]
            ensemble_output = stacked.mean(dim=0)  # [batch, seq, hidden_dim]
        
        else:  # average
            # Simple average
            stacked = torch.stack(model_outputs, dim=0)  # [num_models, batch, seq, hidden_dim]
            ensemble_output = stacked.mean(dim=0)  # [batch, seq, hidden_dim]
        
        # Diversity regularization (encourage diversity)
        if self.config.use_diversity_regularization:
            diversity_loss = -self.config.diversity_weight * diversity  # Negative to encourage diversity
        else:
            diversity_loss = 0.0
        
        metadata = {
            'diversity': diversity,
            'agreement': agreement,
            'avg_confidence': self.avg_confidence.item() if self.confidence_estimator else None,
            'ensemble_method': self.config.ensemble_method,
            'diversity_loss': diversity_loss
        }
        
        return ensemble_output, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ensemble metrics."""
        return {
            'ensemble_diversity': self.ensemble_diversity.item(),
            'model_agreement': self.model_agreement.item(),
            'avg_confidence': self.avg_confidence.item(),
            'ensemble_variance': self.ensemble_variance.item(),
            'weight_entropy': self.weight_entropy.item(),
            'consensus_strength': self.consensus_strength.item(),
            'ensemble_weights': self.ensemble_weights.detach().cpu().numpy().tolist(),
            'num_models': self.num_models,
            'ensemble_method': self.config.ensemble_method,
            'ensemble_quality': (self.model_agreement.item() + self.consensus_strength.item()) / 2.0
        }


class LLMEnsembleModule(nn.Module):
    """
    Módulo LLM Ensemble completo.
    """
    
    def __init__(self, config: LLMEnsembleConfig):
        super().__init__()
        self.config = config
        self.ensemble = LLMEnsemble(config)
        
        logger.info(f"Initialized LLMEnsembleModule with config: {config}")
    
    def forward(self, model_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass."""
        return self.ensemble(model_outputs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return self.ensemble.get_metrics()


class TruthGPT_LLMEnsemble_Integration(nn.Module):
    """Integración de LLM Ensemble con TruthGPT."""
    
    def __init__(self, base_models: List[nn.Module], ensemble_config: LLMEnsembleConfig):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.ensemble_module = LLMEnsembleModule(ensemble_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass integrado con ensemble."""
        # Get outputs from all models
        model_outputs = [model(*args, **kwargs) for model in self.base_models]
        
        # Ensemble combination
        ensemble_output, metadata = self.ensemble_module(model_outputs)
        return ensemble_output


if __name__ == "__main__":
    config = LLMEnsembleConfig(
        hidden_dim=512,
        num_models=3,
        ensemble_method="weighted",
        use_confidence_weighting=True
    )
    
    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        def forward(self, x):
            return self.proj(x)
    
    models = [DummyModel(config.hidden_dim) for _ in range(config.num_models)]
    ensemble = LLMEnsembleModule(config)
    
    x = torch.randn(2, 32, config.hidden_dim)
    model_outputs = [model(x) for model in models]
    output, metadata = ensemble(model_outputs)
    metrics = ensemble.get_metrics()
    
    print(f"✅ LLM Ensemble test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Diversity: {metadata['diversity']:.4f}")
    print(f"   Agreement: {metadata['agreement']:.4f}")
    print(f"   Ensemble method: {metrics['ensemble_method']}")


