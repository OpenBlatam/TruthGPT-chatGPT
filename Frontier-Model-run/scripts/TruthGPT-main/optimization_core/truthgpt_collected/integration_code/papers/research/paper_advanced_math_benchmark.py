#!/usr/bin/env python3
"""
Benchmarking LLMs on Advanced Mathematical Reasoning
====================================================

Berkeley EECS, 2025. www2.eecs.berkeley.edu

No es un "método", sino un benchmark nuevo: 77 preguntas de nivel PhD para evaluar
razonamiento de prueba matemática. Ayuda a medir hasta qué punto los LLMs pueden
generar demostraciones formales y razonamientos muy avanzados.

Técnica principal: Benchmark evaluation framework for advanced mathematical reasoning.
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
class AdvancedMathBenchmarkConfig:
    """Configuración para Advanced Math Benchmark."""
    hidden_dim: int = 512
    num_questions: int = 77
    difficulty_levels: List[str] = None  # ["undergraduate", "graduate", "phd"]
    use_formal_proof_evaluation: bool = True
    use_step_by_step_scoring: bool = True
    evaluation_metrics: List[str] = None  # ["correctness", "completeness", "rigor"]

    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ["undergraduate", "graduate", "phd"]
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["correctness", "completeness", "rigor"]


class ProofEvaluator(nn.Module):
    """
    Evalúa demostraciones formales.
    """
    
    def __init__(self, config: AdvancedMathBenchmarkConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Proof evaluator network
        self.evaluator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # problem + proof
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, len(config.evaluation_metrics))
        )
        
        # Initialize
        for module in self.evaluator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized ProofEvaluator: metrics={config.evaluation_metrics}")
    
    def forward(self, problem: torch.Tensor, proof: torch.Tensor) -> torch.Tensor:
        """
        Evaluate proof.
        
        Args:
            problem: [batch, seq, hidden_dim]
            proof: [batch, seq, hidden_dim]
            
        Returns:
            scores: [batch, num_metrics]
        """
        # Combine problem and proof
        combined = torch.cat([problem, proof], dim=-1)  # [batch, seq, hidden_dim * 2]
        
        # Use last token
        last_token = combined[:, -1, :]  # [batch, hidden_dim * 2]
        
        # Evaluate
        scores = self.evaluator(last_token)  # [batch, num_metrics]
        scores = torch.sigmoid(scores)  # Normalize to [0, 1]
        
        return scores


class StepByStepScorer(nn.Module):
    """
    Evalúa razonamiento paso a paso.
    """
    
    def __init__(self, config: AdvancedMathBenchmarkConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Step scorer
        self.step_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.step_scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized StepByStepScorer")
    
    def forward(self, reasoning_steps: torch.Tensor) -> torch.Tensor:
        """
        Score reasoning steps.
        
        Args:
            reasoning_steps: [batch, num_steps, hidden_dim]
            
        Returns:
            step_scores: [batch, num_steps]
        """
        # Score each step
        step_scores = self.step_scorer(reasoning_steps).squeeze(-1)  # [batch, num_steps]
        
        return step_scores


class AdvancedMathBenchmarkModule(nn.Module):
    """
    Módulo de evaluación de benchmark matemático avanzado.
    """
    
    def __init__(self, config: AdvancedMathBenchmarkConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Components
        if config.use_formal_proof_evaluation:
            self.proof_evaluator = ProofEvaluator(config)
        else:
            self.proof_evaluator = None
        
        if config.use_step_by_step_scoring:
            self.step_scorer = StepByStepScorer(config)
        else:
            self.step_scorer = None
        
        # Metrics tracking
        self.register_buffer('avg_correctness', torch.tensor(0.0))
        self.register_buffer('avg_completeness', torch.tensor(0.0))
        self.register_buffer('avg_rigor', torch.tensor(0.0))
        self.register_buffer('overall_score', torch.tensor(0.0))
        self.register_buffer('phd_level_performance', torch.tensor(0.0))
        
        logger.info(f"Initialized AdvancedMathBenchmarkModule: {config.num_questions} questions")
    
    def evaluate_proof(self, problem: torch.Tensor, proof: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate a mathematical proof.
        
        Args:
            problem: [batch, seq, hidden_dim]
            proof: [batch, seq, hidden_dim]
            
        Returns:
            Dict with evaluation scores
        """
        if self.proof_evaluator is None:
            return {metric: 0.5 for metric in self.config.evaluation_metrics}
        
        # Evaluate
        scores = self.proof_evaluator(problem, proof)  # [batch, num_metrics]
        avg_scores = scores.mean(dim=0)  # [num_metrics]
        
        # Map to metrics
        evaluation = {}
        for i, metric in enumerate(self.config.evaluation_metrics):
            score = avg_scores[i].item()
            evaluation[metric] = score
            
            # Update metrics
            if metric == "correctness":
                self.avg_correctness = 0.9 * self.avg_correctness + 0.1 * score
            elif metric == "completeness":
                self.avg_completeness = 0.9 * self.avg_completeness + 0.1 * score
            elif metric == "rigor":
                self.avg_rigor = 0.9 * self.avg_rigor + 0.1 * score
        
        # Overall score
        overall = avg_scores.mean().item()
        self.overall_score = 0.9 * self.overall_score + 0.1 * overall
        
        return evaluation
    
    def evaluate_reasoning_steps(self, reasoning_steps: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate step-by-step reasoning.
        
        Args:
            reasoning_steps: [batch, num_steps, hidden_dim]
            
        Returns:
            Dict with step evaluation
        """
        if self.step_scorer is None:
            return {'step_scores': [0.5] * reasoning_steps.size(1)}
        
        # Score steps
        step_scores = self.step_scorer(reasoning_steps)  # [batch, num_steps]
        avg_step_scores = step_scores.mean(dim=0)  # [num_steps]
        
        return {
            'step_scores': avg_step_scores.cpu().numpy().tolist(),
            'avg_step_score': avg_step_scores.mean().item(),
            'num_steps': reasoning_steps.size(1)
        }
    
    def forward(self, problem: torch.Tensor, solution: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: evaluate mathematical reasoning.
        
        Args:
            problem: [batch, seq, hidden_dim]
            solution: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_solution: [batch, seq, hidden_dim]
            metadata: Dict with evaluation info
        """
        # Evaluate proof
        proof_evaluation = self.evaluate_proof(problem, solution)
        
        # Evaluate reasoning steps (if available)
        step_evaluation = None
        if solution.size(1) > 1:
            # Treat each position as a step
            step_evaluation = self.evaluate_reasoning_steps(solution)
        
        # Enhance solution based on evaluation
        # Higher scores = more confidence in solution
        correctness = proof_evaluation.get('correctness', 0.5)
        enhancement_weight = correctness
        
        enhanced = problem + enhancement_weight * (solution - problem)
        
        metadata = {
            'proof_evaluation': proof_evaluation,
            'step_evaluation': step_evaluation,
            'overall_score': self.overall_score.item()
        }
        
        return enhanced, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get benchmark metrics."""
        return {
            'avg_correctness': self.avg_correctness.item(),
            'avg_completeness': self.avg_completeness.item(),
            'avg_rigor': self.avg_rigor.item(),
            'overall_score': self.overall_score.item(),
            'phd_level_performance': self.phd_level_performance.item(),
            'num_questions': self.config.num_questions,
            'difficulty_levels': self.config.difficulty_levels
        }


if __name__ == "__main__":
    config = AdvancedMathBenchmarkConfig(
        hidden_dim=512,
        num_questions=77,
        use_formal_proof_evaluation=True
    )
    module = AdvancedMathBenchmarkModule(config)
    problem = torch.randn(2, 32, config.hidden_dim)
    solution = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(problem, solution)
    metrics = module.get_metrics()
    print(f"✅ Advanced Math Benchmark test:")
    print(f"   Problem {problem.shape} -> Solution {solution.shape}")
    print(f"   Proof evaluation: {metadata['proof_evaluation']}")
    print(f"   Overall score: {metadata['overall_score']:.4f}")



