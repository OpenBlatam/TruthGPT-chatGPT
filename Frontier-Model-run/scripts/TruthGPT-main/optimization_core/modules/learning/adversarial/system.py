"""
Adversarial Learning System
===========================

High-level orchestrator for adversarial attacks, defenses, and GAN training.
"""
import time
import logging
import torch
import torch.nn as nn
from typing import Dict, Any

from .config import AdversarialConfig
from .enums import GANType
from .attacks import AdversarialAttacker
from .gan import GANTrainer
from .defense import AdversarialDefense
from .analysis import RobustnessAnalyzer

logger = logging.getLogger(__name__)

class AdversarialLearningSystem:
    """Integrated system for robust neural network development."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.attacker = AdversarialAttacker(config)
        self.gan_trainer = GANTrainer(config)
        self.defense_engine = AdversarialDefense(config)
        self.analyzer = RobustnessAnalyzer(config)
        self.history = []
        logger.info("✅ Adversarial Learning System initialized")
    
    def run_adversarial_learning(self, model: nn.Module, train_data: torch.Tensor, 
                                train_labels: torch.Tensor, test_data: torch.Tensor, 
                                test_labels: torch.Tensor) -> Dict[str, Any]:
        """Execute full adversarial robustness cycle."""
        logger.info(f"🚀 Running adversarial learning cycle for {self.config.attack_type.value}")
        results = {'start_time': time.time(), 'stages': {}}
        
        # 1. GAN Training (if applicable)
        if self.config.gan_type != GANType.VANILLA_GAN:
            results['stages']['gan_training'] = self.gan_trainer.train_gan(train_data)
            
        # 2. Defense Training
        if self.config.enable_defense_training:
            results['stages']['defense_training'] = self._train_defense(model, train_data, train_labels)
            
        # 3. Robustness Analysis
        if self.config.enable_robustness_analysis:
            results['stages']['robustness_analysis'] = self.analyzer.analyze_robustness(
                model, test_data, test_labels
            )
            
        results['end_time'] = time.time()
        self.history.append(results)
        return results

    def _train_defense(self, model, data, labels) -> Dict[str, Any]:
        """Perform a single round of defensive training/augmentation."""
        # Simple implementation: apply defense to data subset
        subset_x = data[:min(128, len(data))]
        subset_y = labels[:min(128, len(labels))]
        defended_x = self.defense_engine.apply_defense(model, subset_x, subset_y)
        return {'status': 'success', 'n_samples': len(defended_x)}

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Create a human-readable summary of the adversarial session."""
        lines = ["ADVERSARIAL LEARNING SESSION REPORT", "="*40]
        lines.append(f"Attack: {self.config.attack_type.value} (eps={self.config.attack_epsilon})")
        lines.append(f"Defense: {self.config.defense_strategy.value}")
        
        if 'robustness_analysis' in results['stages']:
            m = results['stages']['robustness_analysis']
            lines.append(f"Clean Accuracy: {m['clean_accuracy']:.4f}")
            lines.append(f"Robust Accuracy: {m['adversarial_accuracy']:.4f}")
            lines.append(f"Robustness Gap: {m['robustness_gap']:.4f}")
            
        return "\n".join(lines)

