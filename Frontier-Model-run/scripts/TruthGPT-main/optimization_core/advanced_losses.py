"""
Advanced loss functions for enhanced training.
Integrated from loss.py and GRPO.py optimization files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .experience_buffer import Experience

def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Monte-Carlo approximation of KL divergence, k3 estimator."""
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio.exp() - log_ratio - 1

def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = None,
) -> torch.Tensor:
    """Compute masked mean of tensor."""
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1)

def get_token_log_probs(model, input_ids, attention_mask):
    """Compute log-probabilities of tokens under the given model."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp

def compute_probability_ratio(curr_logp, old_logp):
    """Compute the probability ratio between current and old policies."""
    return torch.exp(curr_logp - old_logp)

def compute_clipped_ratio(ratio, epsilon):
    """Clip the ratio to [1-epsilon, 1+epsilon]"""
    return torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

def compute_kl_penalty(curr_logp, ref_logp):
    """Compute per-token KL divergence penalty term."""
    diff = ref_logp - curr_logp
    return torch.exp(diff) - diff - 1

def compute_surrogate_advantage(ratio, clipped_ratio, advantages):
    """Compute the surrogate advantage loss per token using PPO-style clipping."""
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    return torch.min(loss1, loss2)

def compute_per_token_loss(ratio, clipped_ratio, advantages, kl_penalty, beta):
    """Combine surrogate advantage and KL penalty per token."""
    adv_loss = compute_surrogate_advantage(ratio, clipped_ratio, advantages)
    return -(adv_loss - beta * kl_penalty)

class GRPOLoss(nn.Module):
    """Enhanced GRPO actor loss with advanced features."""

    def __init__(self, clip_eps: float = 0.2, kl_weight: float = 0.1, 
                 entropy_weight: float = 0.01, value_weight: float = 0.5):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
        entropy: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute GRPO loss with enhanced features."""
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages
        returns = experience.returns

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)

        total_loss = masked_mean(policy_loss + self.kl_weight * kl, action_mask, dim=-1).mean()

        metrics = {
            "policy_loss": masked_mean(policy_loss, action_mask, dim=-1).mean(),
            "kl_divergence": kl.mean(),
            "ratio_mean": ratio.mean(),
            "advantages_mean": advantages.mean(),
        }

        if entropy is not None:
            entropy_loss = -self.entropy_weight * masked_mean(entropy, action_mask, dim=-1).mean()
            total_loss += entropy_loss
            metrics["entropy_loss"] = entropy_loss

        if values is not None and returns is not None:
            value_loss = F.mse_loss(values, returns)
            total_loss += self.value_weight * value_loss
            metrics["value_loss"] = value_loss

        return total_loss, metrics

class EnhancedGRPOLoss(nn.Module):
    """Enhanced GRPO loss with multiple models and advanced clipping."""

    def __init__(self, clip_eps: float = 0.2, kl_weight: float = 0.1, 
                 beta: float = 1.0, use_adaptive_clipping: bool = True):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.beta = beta
        self.use_adaptive_clipping = use_adaptive_clipping
        self.adaptive_clip_factor = 1.0

    def compute_grpo_loss(
        self,
        current_model,
        old_model,
        ref_model,
        input_ids,
        attention_mask,
        advantages,
    ) -> torch.Tensor:
        """Compute enhanced GRPO loss for a batch of sequences."""
        curr_logp = get_token_log_probs(current_model, input_ids, attention_mask)
        
        with torch.no_grad():
            old_logp = get_token_log_probs(old_model, input_ids, attention_mask)
            ref_logp = get_token_log_probs(ref_model, input_ids, attention_mask)

        ratio = compute_probability_ratio(curr_logp, old_logp)
        
        clip_eps = self.clip_eps
        if self.use_adaptive_clipping:
            clip_eps *= self.adaptive_clip_factor
        
        clipped = compute_clipped_ratio(ratio, clip_eps)
        kl_penalty = compute_kl_penalty(curr_logp, ref_logp)

        per_token_loss = compute_per_token_loss(ratio, clipped, advantages, kl_penalty, self.beta)

        mask = attention_mask.float()
        lengths = mask.sum(dim=1).clamp(min=1)
        loss_per_seq = (per_token_loss * mask).sum(dim=1) / lengths
        
        if self.use_adaptive_clipping:
            clip_fraction = ((ratio - clipped).abs() > 1e-6).float().mean()
            if clip_fraction > 0.3:
                self.adaptive_clip_factor *= 0.9
            elif clip_fraction < 0.1:
                self.adaptive_clip_factor *= 1.1
            self.adaptive_clip_factor = torch.clamp(
                torch.tensor(self.adaptive_clip_factor), 0.1, 2.0
            ).item()
        
        return loss_per_seq.mean()

    def forward(self, *args, **kwargs):
        """Forward pass for compatibility."""
        return self.compute_grpo_loss(*args, **kwargs)

class AdversarialLoss(nn.Module):
    """Adversarial loss for enhanced training stability."""
    
    def __init__(self, discriminator_weight: float = 0.1, gradient_penalty_weight: float = 10.0):
        super().__init__()
        self.discriminator_weight = discriminator_weight
        self.gradient_penalty_weight = gradient_penalty_weight
    
    def compute_gradient_penalty(self, discriminator, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        disc_interpolated = discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def forward(self, generator_output, discriminator, real_data=None):
        """Compute adversarial loss."""
        fake_scores = discriminator(generator_output)
        generator_loss = -fake_scores.mean()
        
        total_loss = generator_loss
        
        if real_data is not None:
            gradient_penalty = self.compute_gradient_penalty(
                discriminator, real_data, generator_output.detach()
            )
            total_loss += self.gradient_penalty_weight * gradient_penalty
        
        return total_loss * self.discriminator_weight

class CurriculumLoss(nn.Module):
    """Curriculum learning loss with adaptive difficulty."""
    
    def __init__(self, initial_difficulty: float = 0.1, max_difficulty: float = 1.0,
                 difficulty_increment: float = 0.01):
        super().__init__()
        self.difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_increment = difficulty_increment
        self.performance_history = []
    
    def update_difficulty(self, performance_metric: float):
        """Update curriculum difficulty based on performance."""
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) >= 10:
            recent_performance = sum(self.performance_history[-10:]) / 10
            
            if recent_performance > 0.8:
                self.difficulty = min(
                    self.max_difficulty, 
                    self.difficulty + self.difficulty_increment
                )
            elif recent_performance < 0.5:
                self.difficulty = max(
                    0.1, 
                    self.difficulty - self.difficulty_increment
                )
    
    def forward(self, base_loss: torch.Tensor, difficulty_mask: torch.Tensor):
        """Apply curriculum weighting to loss."""
        curriculum_weight = self.difficulty * difficulty_mask + (1 - self.difficulty) * (1 - difficulty_mask)
        return (base_loss * curriculum_weight).mean()

def create_loss_function(loss_type: str = "grpo", **kwargs):
    """Factory function to create loss functions."""
    if loss_type == "grpo":
        return GRPOLoss(**kwargs)
    elif loss_type == "enhanced_grpo":
        return EnhancedGRPOLoss(**kwargs)
    elif loss_type == "adversarial":
        return AdversarialLoss(**kwargs)
    elif loss_type == "curriculum":
        return CurriculumLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
