"""
Adversarial Defense Strategies
==============================

Implementation of defense mechanisms (Adversarial Training, Distillation, etc.).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any

from .config import AdversarialConfig
from .enums import DefenseStrategy
from .attacks import AdversarialAttacker

logger = logging.getLogger(__name__)

class AdversarialDefense:
    """Implements various defense mechanisms against adversarial attacks."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        logger.info("✅ Adversarial Defense initialized")
    
    def apply_defense(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the configured defense strategy to the model/data."""
        strat = self.config.defense_strategy
        logger.info(f"🛡️ Applying {strat.value} defense")
        
        if strat == DefenseStrategy.ADVERSARIAL_TRAINING:
            return self._adversarial_training(model, x, y)
        elif strat == DefenseStrategy.DISTILLATION:
            return self._distillation(model, x, y)
        elif strat == DefenseStrategy.INPUT_TRANSFORMATION:
            return self._input_transform(x)
        else:
            return self._adversarial_training(model, x, y)

    def _adversarial_training(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Standard adversarial training using generated examples."""
        attacker = AdversarialAttacker(self.config)
        adv_x = attacker.generate_attack(model, x, y)
        
        model.train()
        out = model(adv_x)
        loss = F.cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        # Note: Optimization step should be done by the outer trainer
        return adv_x

    def _distillation(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Defensive distillation (using soft labels)."""
        model.eval()
        with torch.no_grad():
            teacher_logits = model(x)
            soft_labels = F.softmax(teacher_logits / 2.0, dim=1) # Temperature T=2
            
        model.train()
        student_logits = model(x)
        loss = F.kl_div(F.log_softmax(student_logits / 2.0, dim=1), soft_labels, reduction='batchmean')
        model.zero_grad()
        loss.backward()
        return x

    def _input_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random transformations to input to nullify perturbations."""
        noise = torch.randn_like(x) * 0.01
        return torch.clamp(x + noise, 0, 1)

