"""
Robustness Analyzer
===================

Metrics and tools to evaluate model robustness against adversarial perturbations.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any

from .config import AdversarialConfig
from .attacks import AdversarialAttacker

logger = logging.getLogger(__name__)

class RobustnessAnalyzer:
    """Analyzes model performance degradation under attack."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        logger.info("✅ Robustness Analyzer initialized")
    
    def analyze_robustness(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Calculate clean vs adversarial accuracy."""
        logger.info("🔍 Analyzing model robustness")
        
        # Clean Accuracy
        clean_acc = self._eval_acc(model, x, y)
        
        # Adversarial Accuracy
        attacker = AdversarialAttacker(self.config)
        adv_x = attacker.generate_attack(model, x, y)
        adv_acc = self._eval_acc(model, adv_x, y)
        
        metrics = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'robustness_gap': clean_acc - adv_acc,
            'robustness_ratio': adv_acc / clean_acc if clean_acc > 0 else 0
        }
        return metrics

    def _eval_acc(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            out = model(x)
            prec = torch.argmax(out, 1)
            return (prec == y).float().mean().item()

