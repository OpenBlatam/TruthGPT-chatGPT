"""
Adversarial Attack Generation
=============================

Implementation of various adversarial attack algorithms (FGSM, PGD, BIM, MIM, DeepFool, C&W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List

from .config import AdversarialConfig
from .enums import AdversarialAttackType

logger = logging.getLogger(__name__)

class AdversarialAttacker:
    """Generates adversarial perturbations against neural networks."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.attack_history = []
        logger.info("✅ Adversarial Attacker initialized")
    
    def generate_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples for the given inputs and labels."""
        atype = self.config.attack_type
        logger.info(f"🎯 Generating {atype.value} attack")
        
        if atype == AdversarialAttackType.FGSM:
            adv_x = self._fgsm_attack(model, x, y)
        elif atype == AdversarialAttackType.PGD:
            adv_x = self._pgd_attack(model, x, y)
        elif atype == AdversarialAttackType.BIM:
            adv_x = self._bim_attack(model, x, y)
        elif atype == AdversarialAttackType.MIM:
            adv_x = self._mim_attack(model, x, y)
        elif atype == AdversarialAttackType.DEEPFOOL:
            adv_x = self._deepfool_attack(model, x, y)
        elif atype == AdversarialAttackType.C_W:
            adv_x = self._cw_attack(model, x, y)
        else:
            adv_x = self._fgsm_attack(model, x, y)
        
        return adv_x.detach()

    def _fgsm_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fast Gradient Sign Method."""
        x_adv = x.clone().detach().requires_grad_(True)
        out = model(x_adv)
        loss = F.cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        
        perturbation = self.config.attack_epsilon * x_adv.grad.sign()
        return torch.clamp(x + perturbation, 0, 1)

    def _pgd_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Projected Gradient Descent."""
        x_adv = x.clone().detach()
        for _ in range(self.config.attack_iterations):
            x_adv.requires_grad_(True)
            out = model(x_adv)
            loss = F.cross_entropy(out, y)
            model.zero_grad()
            loss.backward()
            
            # Step
            x_adv = x_adv.detach() + self.config.attack_alpha * x_adv.grad.sign()
            # Project to epsilon-ball
            delta = torch.clamp(x_adv - x, -self.config.attack_epsilon, self.config.attack_epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
            
        return x_adv

    def _bim_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Basic Iterative Method."""
        return self._pgd_attack(model, x, y)  # BIM is essentially PGD without random start in this impl

    def _mim_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Momentum Iterative Method."""
        x_adv = x.clone().detach()
        momentum = torch.zeros_like(x)
        for _ in range(self.config.attack_iterations):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            model.zero_grad()
            loss.backward()
            
            grad = x_adv.grad
            momentum = 0.9 * momentum + grad / (torch.norm(grad, p=1) + 1e-8)
            x_adv = x_adv.detach() + self.config.attack_alpha * momentum.sign()
            delta = torch.clamp(x_adv - x, -self.config.attack_epsilon, self.config.attack_epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
            
        return x_adv

    def _deepfool_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """DeepFool attack (Targeted towards nearest decision boundary)."""
        x_adv = x.clone().detach()
        for _ in range(self.config.attack_iterations):
            x_adv.requires_grad_(True)
            out = model(x_adv)
            _, pred = torch.max(out, 1)
            if (pred != y).all(): break
            
            # Binary cross-entropy gradient simplified logic
            grad = torch.autograd.grad(out.max(), x_adv)[0]
            x_adv = x_adv.detach() + 0.01 * grad / (torch.norm(grad) + 1e-8)
            x_adv = torch.clamp(x_adv, 0, 1)
            
        return x_adv

    def _cw_attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simplified Carlini & Wagner style attack (L2 optimization)."""
        x_adv = x.clone().detach()
        for _ in range(self.config.attack_iterations):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            model.zero_grad()
            loss.backward()
            
            # Optimization step
            x_adv = x_adv.detach() - self.config.attack_alpha * x_adv.grad
            x_adv = torch.clamp(x_adv, 0, 1)
            
        return x_adv

