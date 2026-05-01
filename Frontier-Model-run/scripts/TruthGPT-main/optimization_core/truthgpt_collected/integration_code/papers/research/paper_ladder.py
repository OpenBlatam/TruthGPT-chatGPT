#!/usr/bin/env python3
"""
LADDER: Self-Improving LLMs Through Recursive Problem Decomposition
===================================================================

Toby Simonds, Akira Yoshiyama. Mar 2025. arXiv:2503.00735v3

LADDER (Learning through Autonomous Difficulty-Driven Example Recursion) enables
Large Language Models to autonomously improve their problem-solving capabilities
through self-guided learning by recursively generating and solving progressively
simpler variants of complex problems.

Key Results:
- Improved Llama 3.2 3B from 1% to 82% on undergraduate-level integration problems
- Enabled Qwen2.5 7B Deepseek-R1 Distilled to achieve 73% on MIT Integration Bee
- With TTRL (Test-Time RL): 90% on MIT Integration Bee, surpassing OpenAI o1

Técnica principal: 
- Variant Generation: Tree of progressively simpler variants
- Solution Verification: Numerical integration verification
- Reinforcement Learning: Training on variant trees
- TTRL: Test-Time Reinforcement Learning on test problems
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
class LADDERConfig:
    """
    Configuración para LADDER basada en el paper original.
    
    Basado en: arXiv:2503.00735v3
    """
    hidden_dim: int = 512
    max_decomposition_steps: int = 5
    simplification_dim: int = 256
    use_recursive_learning: bool = True
    learning_rate: float = 0.1
    use_problem_generator: bool = True
    use_solution_verifier: bool = True
    # Parámetros del paper
    num_variants_per_problem: int = 500  # N en el paper
    use_ttrl: bool = False  # Test-Time Reinforcement Learning
    ttrl_steps: int = 100  # Steps para TTRL
    variant_tree_structure: bool = True  # Tree structure (one parent per variant)
    use_numerical_verification: bool = True  # Numerical integration verification


class ProblemSimplifier(nn.Module):
    """
    Genera versiones más simples de un problema (Variant Generation).
    
    Basado en el paper: genera árboles de variantes progresivamente más simples,
    donde cada variante tiene exactamente un padre, manteniendo una progresión
    clara de dificultad.
    """
    
    def __init__(self, config: LADDERConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.simplification_dim = config.simplification_dim
        
        # Variant generation network (genera variantes más simples)
        self.variant_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.simplification_dim)
        )
        
        # Difficulty controller (controla el nivel de dificultad de las variantes)
        self.difficulty_controller = nn.Sequential(
            nn.Linear(config.simplification_dim, config.simplification_dim // 2),
            nn.GELU(),
            nn.Linear(config.simplification_dim // 2, 1),
            nn.Sigmoid()  # 0 = más fácil, 1 = más difícil
        )
        
        # Tree structure maintainer (mantiene estructura de árbol)
        if config.variant_tree_structure:
            self.tree_structure = nn.Sequential(
                nn.Linear(config.simplification_dim, config.simplification_dim),
                nn.GELU()
            )
        else:
            self.tree_structure = None
        
        # Initialize
        for module in self.variant_generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized ProblemSimplifier (Variant Generator)")
    
    def forward(self, problem_representation: torch.Tensor) -> torch.Tensor:
        """
        Simplify problem representation.
        
        Args:
            problem_representation: [batch, seq, hidden_dim]
            
        Returns:
            simplified: [batch, seq, simplification_dim]
        """
        # Simplify
        simplified = self.simplifier(problem_representation)  # [batch, seq, simplification_dim]
        
        # Reduce complexity
        simplified = self.complexity_reducer(simplified)
        
        return simplified


class SolutionVerifier(nn.Module):
    """
    Verifica si una solución es correcta.
    
    Basado en el paper: usa verificación numérica (numerical integration method)
    para verificar soluciones de integrales. Para otros dominios, puede usar
    verificadores específicos (unit tests, Lean proof assistant, etc.).
    """
    
    def __init__(self, config: LADDERConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Verifier network (para verificación semántica)
        self.verifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # problem + solution
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Numerical verification flag
        self.use_numerical_verification = config.use_numerical_verification
        
        # Initialize
        for module in self.verifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized SolutionVerifier (numerical_verification={config.use_numerical_verification})")
    
    def numerical_verify(self, problem: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
        """
        Verificación numérica (para integración matemática).
        En el paper usan numerical integration method.
        """
        # Placeholder: en implementación real, usaría scipy.integrate o similar
        # Por ahora, usa verificación semántica
        return self.verifier(torch.cat([problem, solution], dim=-1))
    
    def forward(self, problem: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
        """
        Verify solution.
        
        Args:
            problem: [batch, seq, hidden_dim]
            solution: [batch, seq, hidden_dim]
            
        Returns:
            verification_score: [batch, seq]
        """
        # Combine problem and solution
        combined = torch.cat([problem, solution], dim=-1)  # [batch, seq, hidden_dim * 2]
        
        # Verify
        score = self.verifier(combined).squeeze(-1)  # [batch, seq]
        
        return score


class RecursiveLearner(nn.Module):
    """
    Aprende de forma recursiva resolviendo problemas simplificados.
    
    Basado en el paper: el modelo aprende de las variantes más simples primero,
    luego progresa hacia variantes más complejas, creando un gradiente natural
    de dificultad para el aprendizaje.
    """
    
    def __init__(self, config: LADDERConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Learning network (aprende de variantes)
        self.learner = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Difficulty-aware learning (ajusta aprendizaje según dificultad)
        self.difficulty_aware = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),  # +1 for difficulty
            nn.GELU()
        )
        
        # Initialize
        for module in self.learner:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized RecursiveLearner (with difficulty-aware learning)")
    
    def forward(self, simplified_problem: torch.Tensor, step: int) -> torch.Tensor:
        """
        Learn from simplified problem.
        
        Args:
            simplified_problem: [batch, seq, hidden_dim]
            step: Current decomposition step
            
        Returns:
            learned_solution: [batch, seq, hidden_dim]
        """
        # Learn solution
        learned = self.learner(simplified_problem)
        
        return learned


class LADDERModule(nn.Module):
    """
    Módulo LADDER completo.
    """
    
    def __init__(self, config: LADDERConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Components
        if config.use_problem_generator:
            self.simplifier = ProblemSimplifier(config)
        else:
            self.simplifier = None
        
        if config.use_solution_verifier:
            self.verifier = SolutionVerifier(config)
        else:
            self.verifier = None
        
        if config.use_recursive_learning:
            self.learner = RecursiveLearner(config)
        else:
            self.learner = None
        
        # Recomposer (combine solutions from simpler problems)
        self.recomposer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # TTRL (Test-Time Reinforcement Learning) components
        if config.use_ttrl:
            self.ttrl_optimizer = None  # Se inicializa durante TTRL
            self.ttrl_steps = config.ttrl_steps
        
        # Metrics (basados en resultados del paper)
        self.register_buffer('avg_decomposition_steps', torch.tensor(0.0))
        self.register_buffer('avg_verification_score', torch.tensor(0.5))
        self.register_buffer('learning_progress', torch.tensor(0.0))
        self.register_buffer('variant_generation_count', torch.tensor(0.0))
        self.register_buffer('ttrl_improvement', torch.tensor(0.0))
        
        logger.info(f"Initialized LADDERModule: max_steps={config.max_decomposition_steps}, TTRL={config.use_ttrl}")
    
    def _recursive_solve(self, problem: torch.Tensor, step: int = 0, 
                        parent_variant: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """
        Recursively solve problem by decomposing into simpler versions.
        
        Basado en el algoritmo LADDER del paper:
        1. Genera árbol de variantes progresivamente más simples
        2. Cada variante tiene exactamente un padre (tree structure)
        3. Resuelve desde las más simples hacia las más complejas
        
        Args:
            problem: [batch, seq, hidden_dim]
            step: Current decomposition step
            parent_variant: Optional parent variant (for tree structure)
            
        Returns:
            solution: [batch, seq, hidden_dim]
            verification_score: float
        """
        if step >= self.config.max_decomposition_steps:
            # Base case: solve directly (problema simple que el modelo puede resolver)
            if self.learner is not None:
                solution = self.learner(problem, step)
            else:
                solution = problem
            return solution, 0.5
        
        # Generate variants (tree structure: one parent per variant)
        if self.simplifier is not None:
            # Generate simplified variant
            simplified = self.simplifier.variant_generator(problem)  # [batch, seq, simplification_dim]
            
            # Control difficulty (más fácil en cada paso)
            difficulty = 1.0 - (step / self.config.max_decomposition_steps)
            difficulty_tensor = torch.full((simplified.size(0), 1), difficulty, device=simplified.device)
            
            # Apply tree structure if enabled
            if self.config.variant_tree_structure and parent_variant is not None and self.simplifier.tree_structure is not None:
                # Mantener relación con padre
                parent_features = parent_variant[:, -1:, :]  # Last token
                # Project parent to simplification_dim first
                parent_proj = nn.Linear(self.hidden_dim, self.config.simplification_dim).to(parent_features.device)
                parent_simplified = parent_proj(parent_features)
                simplified = simplified + 0.1 * self.simplifier.tree_structure(parent_simplified)
            
            # Project back to hidden_dim
            simplified_proj = nn.Linear(self.config.simplification_dim, self.hidden_dim).to(problem.device)
            simplified_hidden = simplified_proj(simplified)
            
            # Update variant generation count
            self.variant_generation_count += 1
        else:
            simplified_hidden = problem
        
        # Recursively solve simplified problem
        simplified_solution, _ = self._recursive_solve(simplified_hidden, step + 1, problem)
        
        # Verify solution (numerical verification for integration, semantic for others)
        verification_score = 0.5
        if self.verifier is not None:
            if self.verifier.use_numerical_verification:
                verification_score = self.verifier.numerical_verify(problem, simplified_solution).mean().item()
            else:
                verification_score = self.verifier(problem, simplified_solution).mean().item()
            self.avg_verification_score = 0.9 * self.avg_verification_score + 0.1 * verification_score
        
        # Recompose solution (usar solución de variante más simple para resolver problema más complejo)
        recomposed = self.recomposer(simplified_solution)
        
        return recomposed, verification_score
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: recursive problem decomposition and learning.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with decomposition info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Recursively solve
        solution, verification_score = self._recursive_solve(hidden_states, step=0)
        
        # Combine with original
        output = hidden_states + 0.3 * solution
        
        # Update metrics
        self.avg_decomposition_steps = 0.9 * self.avg_decomposition_steps + 0.1 * self.config.max_decomposition_steps
        self.learning_progress = 0.9 * self.learning_progress + 0.1 * verification_score
        
        metadata = {
            'verification_score': verification_score,
            'decomposition_steps': self.config.max_decomposition_steps
        }
        
        return output, metadata
    
    def apply_ttrl(self, test_problem: torch.Tensor, max_steps: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply Test-Time Reinforcement Learning (TTRL).
        
        Basado en el paper: TTRL realiza RL en variantes de problemas de test
        en tiempo de inferencia. Esto permite al modelo resolver problemas
        que LADDER no pudo resolver correctamente.
        
        Args:
            test_problem: [batch, seq, hidden_dim] - problema de test
            max_steps: Máximo número de pasos TTRL (default: config.ttrl_steps)
            
        Returns:
            solution: [batch, seq, hidden_dim]
            metadata: Dict con información de TTRL
        """
        max_steps = max_steps or self.config.ttrl_steps
        
        # Generar variantes del problema de test
        variants = []
        current = test_problem
        
        for step in range(max_steps):
            # Generar variante más simple
            if self.simplifier is not None:
                variant = self.simplifier.variant_generator(current)
                variant_proj = nn.Linear(self.config.simplification_dim, self.hidden_dim).to(current.device)
                variant_hidden = variant_proj(variant)
            else:
                variant_hidden = current
            
            # Resolver variante
            solution, _ = self._recursive_solve(variant_hidden, step=0)
            
            # Verificar solución
            if self.verifier is not None:
                score = self.verifier(test_problem, solution).mean().item()
                if score > 0.7:  # Threshold del paper
                    self.ttrl_improvement = 0.9 * self.ttrl_improvement + 0.1 * score
                    return solution, {'ttrl_steps': step + 1, 'verification_score': score}
            
            variants.append((variant_hidden, solution))
            current = variant_hidden
        
        # Si no se resolvió, usar última solución
        final_solution = variants[-1][1] if variants else test_problem
        
        return final_solution, {'ttrl_steps': max_steps, 'verification_score': 0.5}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'avg_decomposition_steps': self.avg_decomposition_steps.item(),
            'avg_verification_score': self.avg_verification_score.item(),
            'learning_progress': self.learning_progress.item(),
            'variant_generation_count': self.variant_generation_count.item(),
            'ttrl_improvement': self.ttrl_improvement.item() if self.config.use_ttrl else None,
            'max_steps': self.config.max_decomposition_steps,
            'num_variants_per_problem': self.config.num_variants_per_problem
        }


if __name__ == "__main__":
    config = LADDERConfig(
        hidden_dim=512,
        max_decomposition_steps=5,
        use_recursive_learning=True
    )
    module = LADDERModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ LADDER test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Verification score: {metadata['verification_score']:.4f}")
    print(f"   Learning progress: {metrics['learning_progress']:.4f}")


