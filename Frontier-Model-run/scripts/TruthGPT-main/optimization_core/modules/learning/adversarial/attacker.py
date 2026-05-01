"""
Adversarial Attacker
====================

Algorithms for generating adversarial attacks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Any

from .config import AdversarialConfig
from .enums import AdversarialAttackType

logger = logging.getLogger(__name__)

class AdversarialAttacker:
    """Adversarial attack generator"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.attack_history = []
        logger.info("✅ Adversarial Attacker initialized")
    
    def generate_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial attack"""
        logger.info(f"🎯 Generating {self.config.attack_type.value} attack")
        
        if self.config.attack_type == AdversarialAttackType.FGSM:
            adversarial_x = self._fgsm_attack(model, x, y)
        elif self.config.attack_type == AdversarialAttackType.PGD:
            adversarial_x = self._pgd_attack(model, x, y)
        elif self.config.attack_type == AdversarialAttackType.C_W:
            adversarial_x = self._cw_attack(model, x, y)
        elif self.config.attack_type == AdversarialAttackType.DEEPFOOL:
            adversarial_x = self._deepfool_attack(model, x, y)
        elif self.config.attack_type == AdversarialAttackType.BIM:
            adversarial_x = self._bim_attack(model, x, y)
        elif self.config.attack_type == AdversarialAttackType.MIM:
            adversarial_x = self._mim_attack(model, x, y)
        else:
            adversarial_x = self._fgsm_attack(model, x, y)
        
        # Store attack history
        self.attack_history.append({
            'attack_type': self.config.attack_type.value,
            'original_x': x.cpu() if x.is_cuda else x,
            'adversarial_x': adversarial_x.cpu() if adversarial_x.is_cuda else adversarial_x,
            'epsilon': self.config.attack_epsilon
        })
        
        return adversarial_x
    
    def _fgsm_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fast Gradient Sign Method attack"""
        x_adv = x.clone().detach().requires_grad_(True)
        
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.data
        x_adv = x_adv + self.config.attack_epsilon * grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()
    
    def _pgd_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Projected Gradient Descent attack"""
        x_adv = x.clone().detach()
        
        for i in range(self.config.attack_iterations):
            x_adv.requires_grad_(True)
            output = model(x_adv)
            loss = F.cross_entropy(output, y)
            
            model.zero_grad()
            loss.backward()
            
            grad = x_adv.grad.data
            x_adv = x_adv + self.config.attack_alpha * grad.sign()
            
            # Projection
            delta = torch.clamp(x_adv - x, min=-self.config.attack_epsilon, max=self.config.attack_epsilon)
            x_adv = torch.clamp(x + delta, 0, 1).detach()
            
        return x_adv
    
    def _cw_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Carlini & Wagner attack (simplified)"""
        x_adv = x.clone().detach()
        # Placeholder for real CW logic which is complex
        logger.warning("Simplified CW attack used as placeholder")
        return self._pgd_attack(model, x, y)
    
    def _deepfool_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """DeepFool attack (simplified)"""
        logger.warning("Simplified DeepFool attack used as placeholder")
        return self._pgd_attack(model, x, y)
    
    def _bim_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Basic Iterative Method attack"""
        return self._pgd_attack(model, x, y)
    
    def _mim_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Momentum Iterative Method attack"""
        x_adv = x.clone().detach()
        momentum = torch.zeros_like(x)
        
        for i in range(self.config.attack_iterations):
            x_adv.requires_grad_(True)
            output = model(x_adv)
            loss = F.cross_entropy(output, y)
            
            model.zero_grad()
            loss.backward()
            
            grad = x_adv.grad.data
            # Normalized gradient
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            momentum = 0.9 * momentum + grad
            
            x_adv = x_adv + self.config.attack_alpha * momentum.sign()
            delta = torch.clamp(x_adv - x, min=-self.config.attack_epsilon, max=self.config.attack_epsilon)
            x_adv = torch.clamp(x + delta, 0, 1).detach()
            
        return x_adv

