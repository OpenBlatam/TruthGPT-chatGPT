#!/usr/bin/env python3
"""
Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles
===============================================================================================

Chen, He, Yuan, Chen, Cai, Dai, Yu, Yu, Li, Chen, Zhou, Wang. May 2025. arXiv

Proponen un conjunto de puzzles sintéticos verificables, con generador + verificador,
para entrenar LLMs con RL y mejorar razonamiento lógico. Su modelo Qwen2.5-32B-Enigmata
sobrepasa a otros en benchmarks de "puzzles" y generaliza a tareas de matemática.

Técnica principal: Synthetic verifiable puzzles with generator + verifier for RL training.
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
class EnigmataConfig:
    """Configuración para Enigmata."""
    hidden_dim: int = 512
    puzzle_dim: int = 256
    use_puzzle_generator: bool = True
    use_puzzle_verifier: bool = True
    use_rl_training: bool = True
    puzzle_complexity: float = 0.5
    verification_threshold: float = 0.7


class PuzzleGenerator(nn.Module):
    """
    Genera puzzles sintéticos verificables.
    """
    
    def __init__(self, config: EnigmataConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.puzzle_dim = config.puzzle_dim
        
        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.puzzle_dim)
        )
        
        # Complexity controller
        self.complexity_controller = nn.Sequential(
            nn.Linear(config.puzzle_dim, config.puzzle_dim // 2),
            nn.GELU(),
            nn.Linear(config.puzzle_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized PuzzleGenerator")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic puzzle.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            puzzle: [batch, seq, puzzle_dim]
            complexity: [batch, seq]
        """
        # Generate puzzle
        puzzle = self.generator(hidden_states)  # [batch, seq, puzzle_dim]
        
        # Control complexity
        complexity = self.complexity_controller(puzzle).squeeze(-1)  # [batch, seq]
        
        # Adjust complexity
        complexity_target = self.config.puzzle_complexity
        complexity_adjusted = complexity * complexity_target + (1 - complexity) * (1 - complexity_target)
        
        return puzzle, complexity_adjusted


class PuzzleVerifier(nn.Module):
    """
    Verifica si una solución a un puzzle es correcta.
    """
    
    def __init__(self, config: EnigmataConfig):
        super().__init__()
        self.config = config
        self.puzzle_dim = config.puzzle_dim
        self.hidden_dim = config.hidden_dim
        
        # Verifier network
        self.verifier = nn.Sequential(
            nn.Linear(config.puzzle_dim + config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.verifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized PuzzleVerifier")
    
    def forward(self, puzzle: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
        """
        Verify puzzle solution.
        
        Args:
            puzzle: [batch, seq, puzzle_dim]
            solution: [batch, seq, hidden_dim]
            
        Returns:
            verification_score: [batch, seq]
        """
        # Combine puzzle and solution
        combined = torch.cat([puzzle, solution], dim=-1)  # [batch, seq, puzzle_dim + hidden_dim]
        
        # Verify
        score = self.verifier(combined).squeeze(-1)  # [batch, seq]
        
        return score


class PuzzleSolver(nn.Module):
    """
    Resuelve puzzles generados.
    """
    
    def __init__(self, config: EnigmataConfig):
        super().__init__()
        self.config = config
        self.puzzle_dim = config.puzzle_dim
        self.hidden_dim = config.hidden_dim
        
        # Solver network
        self.solver = nn.Sequential(
            nn.Linear(config.puzzle_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Initialize
        for module in self.solver:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized PuzzleSolver")
    
    def forward(self, puzzle: torch.Tensor) -> torch.Tensor:
        """
        Solve puzzle.
        
        Args:
            puzzle: [batch, seq, puzzle_dim]
            
        Returns:
            solution: [batch, seq, hidden_dim]
        """
        # Solve
        solution = self.solver(puzzle)  # [batch, seq, hidden_dim]
        
        return solution


class EnigmataModule(nn.Module):
    """
    Módulo Enigmata completo.
    """
    
    def __init__(self, config: EnigmataConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Components
        if config.use_puzzle_generator:
            self.generator = PuzzleGenerator(config)
        else:
            self.generator = None
        
        if config.use_puzzle_verifier:
            self.verifier = PuzzleVerifier(config)
        else:
            self.verifier = None
        
        self.solver = PuzzleSolver(config)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Metrics
        self.register_buffer('avg_puzzle_complexity', torch.tensor(0.5))
        self.register_buffer('avg_verification_score', torch.tensor(0.5))
        self.register_buffer('puzzle_solving_rate', torch.tensor(0.5))
        
        logger.info("Initialized EnigmataModule")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: generate puzzle, solve, verify.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with puzzle info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Generate puzzle
        if self.generator is not None:
            puzzle, complexity = self.generator(hidden_states)  # [batch, seq, puzzle_dim], [batch, seq]
            self.avg_puzzle_complexity = 0.9 * self.avg_puzzle_complexity + 0.1 * complexity.mean().item()
        else:
            # Use hidden states as puzzle
            puzzle_proj = nn.Linear(self.hidden_dim, self.config.puzzle_dim).to(hidden_states.device)
            puzzle = puzzle_proj(hidden_states)
            complexity = torch.ones(batch_size, seq_len, device=hidden_states.device) * self.config.puzzle_complexity
        
        # Solve puzzle
        solution = self.solver(puzzle)  # [batch, seq, hidden_dim]
        
        # Verify solution
        verification_score = None
        if self.verifier is not None:
            verification_score = self.verifier(puzzle, solution)  # [batch, seq]
            avg_score = verification_score.mean().item()
            self.avg_verification_score = 0.9 * self.avg_verification_score + 0.1 * avg_score
            
            # Update solving rate
            solving_rate = (verification_score > self.config.verification_threshold).float().mean().item()
            self.puzzle_solving_rate = 0.9 * self.puzzle_solving_rate + 0.1 * solving_rate
        
        # Project output
        output = self.output_projection(solution)
        
        # Combine with original
        output = hidden_states + 0.3 * output
        
        metadata = {
            'puzzle_complexity': complexity.mean().item(),
            'verification_score': verification_score.mean().item() if verification_score is not None else None,
            'solving_rate': self.puzzle_solving_rate.item()
        }
        
        return output, metadata
    
    def compute_rl_reward(self, puzzle: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute RL reward based on verification.
        
        Args:
            puzzle: [batch, seq, puzzle_dim]
            solution: [batch, seq, hidden_dim]
            
        Returns:
            rewards: [batch]
        """
        if self.verifier is not None:
            verification = self.verifier(puzzle, solution)  # [batch, seq]
            # Reward is verification score
            rewards = verification.mean(dim=1)  # [batch]
        else:
            # Default reward
            rewards = torch.ones(puzzle.size(0), device=puzzle.device) * 0.5
        
        return rewards
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'avg_puzzle_complexity': self.avg_puzzle_complexity.item(),
            'avg_verification_score': self.avg_verification_score.item(),
            'puzzle_solving_rate': self.puzzle_solving_rate.item()
        }


if __name__ == "__main__":
    config = EnigmataConfig(
        hidden_dim=512,
        use_puzzle_generator=True,
        use_puzzle_verifier=True
    )
    module = EnigmataModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ Enigmata test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Puzzle complexity: {metadata['puzzle_complexity']:.4f}")
    print(f"   Verification score: {metadata['verification_score']:.4f}")



