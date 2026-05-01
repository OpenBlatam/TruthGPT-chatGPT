#!/usr/bin/env python3
"""
PlanU: Large Language Model Reasoning through Planning under Uncertainty
========================================================================

Aborda cómo los LLMs pueden planear en entornos con incertidumbre, considerando
tanto la incertidumbre del modelo como del entorno.

Técnica principal: Planning under uncertainty with model and environment uncertainty.

Basado en: arXiv paper (November 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlanUConfig:
    """Configuración para PlanU Planning under Uncertainty."""
    hidden_dim: int = 512
    planning_horizon: int = 5
    uncertainty_dim: int = 128
    use_model_uncertainty: bool = True
    use_environment_uncertainty: bool = True
    uncertainty_aggregation: str = "weighted"  # weighted, max, mean
    use_monte_carlo: bool = True
    num_samples: int = 10


class UncertaintyEstimator(nn.Module):
    """
    Estimador de incertidumbre para modelo y entorno.
    """
    
    def __init__(self, config: PlanUConfig, uncertainty_type: str = "model"):
        super().__init__()
        assert uncertainty_type in ["model", "environment"], \
            f"uncertainty_type must be 'model' or 'environment', got {uncertainty_type}"
        
        self.config = config
        self.uncertainty_type = uncertainty_type
        self.hidden_dim = config.hidden_dim
        self.uncertainty_dim = config.uncertainty_dim
        
        # Uncertainty projection
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.uncertainty_dim * 2),
            nn.GELU(),
            nn.Linear(config.uncertainty_dim * 2, config.uncertainty_dim)
        )
        
        # Uncertainty head (predicts mean and variance)
        self.uncertainty_mean = nn.Linear(config.uncertainty_dim, config.uncertainty_dim)
        self.uncertainty_var = nn.Linear(config.uncertainty_dim, config.uncertainty_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.uncertainty_proj[0].weight)
        nn.init.xavier_uniform_(self.uncertainty_proj[2].weight)
        nn.init.xavier_uniform_(self.uncertainty_mean.weight)
        nn.init.xavier_uniform_(self.uncertainty_var.weight)
        
        logger.info(f"Initialized UncertaintyEstimator ({uncertainty_type})")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty (mean and variance).
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            uncertainty_mean: [batch, seq, uncertainty_dim]
            uncertainty_var: [batch, seq, uncertainty_dim]
        """
        # Project to uncertainty space
        uncertainty_features = self.uncertainty_proj(hidden_states)  # [batch, seq, uncertainty_dim]
        
        # Predict mean and variance
        uncertainty_mean = self.uncertainty_mean(uncertainty_features)
        uncertainty_var = F.softplus(self.uncertainty_var(uncertainty_features)) + 1e-6  # Ensure positive
        
        return uncertainty_mean, uncertainty_var


class PlanningUnderUncertainty(nn.Module):
    """
    Planning under Uncertainty module.
    
    Técnica: Planifica considerando incertidumbre del modelo y del entorno.
    
    Mejoras avanzadas:
    - Adaptive planning horizon
    - Uncertainty-aware planning
    - Plan quality scoring
    - Temporal planning consistency
    """
    
    def __init__(self, config: PlanUConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.planning_horizon = config.planning_horizon
        
        # Uncertainty estimators
        if config.use_model_uncertainty:
            self.model_uncertainty = UncertaintyEstimator(config, "model")
        else:
            self.model_uncertainty = None
        
        if config.use_environment_uncertainty:
            self.env_uncertainty = UncertaintyEstimator(config, "environment")
        else:
            self.env_uncertainty = None
        
        # Planning network
        self.planning_network = nn.Sequential(
            nn.Linear(config.hidden_dim + config.uncertainty_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Plan quality scorer
        self.plan_quality_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptive horizon controller
        self.horizon_controller = nn.Linear(config.uncertainty_dim, 1)
        
        # Initialize
        for module in self.planning_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Metrics
        self.register_buffer('avg_model_uncertainty', torch.tensor(0.0))
        self.register_buffer('avg_env_uncertainty', torch.tensor(0.0))
        self.register_buffer('planning_confidence', torch.tensor(1.0))
        self.register_buffer('uncertainty_reduction', torch.tensor(0.0))
        self.register_buffer('planning_stability', torch.tensor(1.0))
        self.register_buffer('monte_carlo_variance', torch.tensor(0.0))
        self.register_buffer('plan_quality', torch.tensor(0.5))
        self.register_buffer('adaptive_horizon', torch.tensor(float(config.planning_horizon)))
        
        logger.info(f"Initialized PlanningUnderUncertainty: horizon={config.planning_horizon}")
    
    def _aggregate_uncertainty(self, model_mean: torch.Tensor, model_var: torch.Tensor,
                              env_mean: torch.Tensor, env_var: torch.Tensor) -> torch.Tensor:
        """
        Aggregate model and environment uncertainty.
        
        Args:
            model_mean, model_var: [batch, seq, uncertainty_dim]
            env_mean, env_var: [batch, seq, uncertainty_dim]
            
        Returns:
            aggregated_uncertainty: [batch, seq, uncertainty_dim]
        """
        if self.config.uncertainty_aggregation == "weighted":
            # Weighted combination (model uncertainty weighted more)
            model_weight = 0.6
            env_weight = 0.4
            aggregated = model_weight * (model_mean + model_var) + env_weight * (env_mean + env_var)
        elif self.config.uncertainty_aggregation == "max":
            # Take maximum uncertainty
            model_unc = model_mean + model_var
            env_unc = env_mean + env_var
            aggregated = torch.max(model_unc, env_unc)
        else:  # mean
            # Simple mean
            aggregated = (model_mean + model_var + env_mean + env_var) / 4.0
        
        return aggregated
    
    def _monte_carlo_planning(self, hidden_states: torch.Tensor, 
                             uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Monte Carlo planning: sample multiple plans and aggregate.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            uncertainty: [batch, seq, uncertainty_dim]
            
        Returns:
            planned_states: [batch, seq, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_samples = self.config.num_samples
        
        # Sample from uncertainty distribution
        uncertainty_expanded = uncertainty.unsqueeze(2).expand(-1, -1, num_samples, -1)  # [batch, seq, samples, unc_dim]
        
        # Sample noise
        noise = torch.randn_like(uncertainty_expanded)
        sampled_uncertainty = uncertainty_expanded + noise * 0.1  # Add noise
        
        # Combine with hidden states for each sample
        hidden_expanded = hidden_states.unsqueeze(2).expand(-1, -1, num_samples, -1)  # [batch, seq, samples, hidden_dim]
        combined = torch.cat([hidden_expanded, sampled_uncertainty], dim=-1)  # [batch, seq, samples, hidden_dim + unc_dim]
        
        # Plan for each sample
        combined_flat = combined.view(batch_size * seq_len * num_samples, -1)
        planned_flat = self.planning_network(combined_flat)
        planned = planned_flat.view(batch_size, seq_len, num_samples, self.hidden_dim)
        
        # Aggregate plans (mean)
        planned_states = planned.mean(dim=2)  # [batch, seq, hidden_dim]
        
        return planned_states
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: planning under uncertainty.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            planned_states: [batch, seq, hidden_dim]
            metadata: Dict with uncertainty info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Estimate uncertainties
        if self.model_uncertainty is not None:
            model_mean, model_var = self.model_uncertainty(hidden_states)
            model_uncertainty = model_mean + model_var
        else:
            model_mean = model_var = torch.zeros(batch_size, seq_len, self.config.uncertainty_dim, device=hidden_states.device)
            model_uncertainty = model_mean
        
        if self.env_uncertainty is not None:
            env_mean, env_var = self.env_uncertainty(hidden_states)
            env_uncertainty = env_mean + env_var
        else:
            env_mean = env_var = torch.zeros(batch_size, seq_len, self.config.uncertainty_dim, device=hidden_states.device)
            env_uncertainty = env_mean
        
        # Aggregate uncertainty
        if self.model_uncertainty is not None and self.env_uncertainty is not None:
            aggregated_uncertainty = self._aggregate_uncertainty(model_mean, model_var, env_mean, env_var)
        elif self.model_uncertainty is not None:
            aggregated_uncertainty = model_uncertainty
        elif self.env_uncertainty is not None:
            aggregated_uncertainty = env_uncertainty
        else:
            aggregated_uncertainty = torch.zeros(batch_size, seq_len, self.config.uncertainty_dim, device=hidden_states.device)
        
        # Planning
        if self.config.use_monte_carlo:
            planned_states = self._monte_carlo_planning(hidden_states, aggregated_uncertainty)
        else:
            # Deterministic planning
            combined = torch.cat([hidden_states, aggregated_uncertainty], dim=-1)
            planned_states = self.planning_network(combined)
        
        # Compute plan quality
        plan_quality = self.plan_quality_scorer(planned_states).squeeze(-1).mean().item()  # [batch, seq] -> scalar
        self.plan_quality = 0.9 * self.plan_quality + 0.1 * plan_quality
        
        # Adaptive horizon based on uncertainty
        if aggregated_uncertainty.size(-1) > 0:
            uncertainty_summary = aggregated_uncertainty.mean(dim=[0, 1])  # [uncertainty_dim]
            horizon_adjustment = torch.sigmoid(self.horizon_controller(uncertainty_summary)).item()
            adaptive_horizon = self.config.planning_horizon * (1.0 + horizon_adjustment)
            self.adaptive_horizon = 0.9 * self.adaptive_horizon + 0.1 * adaptive_horizon
        
        # Combine with original states (weighted by plan quality)
        plan_weight = plan_quality
        output = hidden_states + plan_weight * planned_states
        
        # Update metrics
        self.avg_model_uncertainty = 0.9 * self.avg_model_uncertainty + 0.1 * model_uncertainty.mean().item()
        self.avg_env_uncertainty = 0.9 * self.avg_env_uncertainty + 0.1 * env_uncertainty.mean().item()
        
        # Planning confidence (inverse of uncertainty)
        confidence = 1.0 / (1.0 + aggregated_uncertainty.mean().item())
        self.planning_confidence = 0.9 * self.planning_confidence + 0.1 * confidence
        
        # Uncertainty reduction (how much we reduced uncertainty through planning)
        if hasattr(self, '_previous_uncertainty'):
            reduction = self._previous_uncertainty - aggregated_uncertainty.mean().item()
            self.uncertainty_reduction = 0.9 * self.uncertainty_reduction + 0.1 * max(0, reduction)
        self._previous_uncertainty = aggregated_uncertainty.mean().item()
        
        # Planning stability (variance of planned states)
        if self.config.use_monte_carlo:
            # Variance across Monte Carlo samples
            variance = planned_states.var(dim=-1).mean().item()
            self.monte_carlo_variance = 0.9 * self.monte_carlo_variance + 0.1 * variance
            stability = 1.0 / (1.0 + variance)
            self.planning_stability = 0.9 * self.planning_stability + 0.1 * stability
        
        metadata = {
            'model_uncertainty': model_uncertainty.mean().item(),
            'env_uncertainty': env_uncertainty.mean().item(),
            'aggregated_uncertainty': aggregated_uncertainty.mean().item(),
            'planning_confidence': confidence,
            'plan_quality': plan_quality,
            'adaptive_horizon': self.adaptive_horizon.item()
        }
        
        return output, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get planning metrics."""
        return {
            'avg_model_uncertainty': self.avg_model_uncertainty.item(),
            'avg_env_uncertainty': self.avg_env_uncertainty.item(),
            'planning_confidence': self.planning_confidence.item(),
            'uncertainty_reduction': self.uncertainty_reduction.item(),
            'planning_stability': self.planning_stability.item(),
            'monte_carlo_variance': self.monte_carlo_variance.item(),
            'plan_quality': self.plan_quality.item(),
            'adaptive_horizon': self.adaptive_horizon.item(),
            'planning_horizon': self.planning_horizon,
            'uncertainty_ratio': self.avg_model_uncertainty.item() / (self.avg_env_uncertainty.item() + 1e-8),
            'planning_efficiency': self.plan_quality.item() / (self.avg_model_uncertainty.item() + self.avg_env_uncertainty.item() + 1e-8)
        }


class PlanUModule(nn.Module):
    """
    Módulo PlanU completo para planning under uncertainty.
    """
    
    def __init__(self, config: PlanUConfig):
        super().__init__()
        self.config = config
        self.planning = PlanningUnderUncertainty(config)
        
        logger.info(f"Initialized PlanUModule with config: {config}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass."""
        return self.planning(hidden_states)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return self.planning.get_metrics()


class TruthGPT_PlanU_Integration(nn.Module):
    """Integración de PlanU con TruthGPT."""
    
    def __init__(self, base_model, planu_config: PlanUConfig):
        super().__init__()
        self.base_model = base_model
        self.planu_module = PlanUModule(planu_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass integrado con PlanU."""
        output = self.base_model(*args, **kwargs)
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            enhanced_output, metadata = self.planu_module(output)
            return enhanced_output
        return output


if __name__ == "__main__":
    config = PlanUConfig(
        hidden_dim=512,
        planning_horizon=5,
        use_model_uncertainty=True,
        use_environment_uncertainty=True
    )
    module = PlanUModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ PlanU test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Model uncertainty: {metadata['model_uncertainty']:.4f}")
    print(f"   Env uncertainty: {metadata['env_uncertainty']:.4f}")
    print(f"   Planning confidence: {metrics['planning_confidence']:.4f}")


