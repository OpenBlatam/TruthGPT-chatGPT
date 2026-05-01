#!/usr/bin/env python3
"""
SPOC: Boosting LLM Reasoning via Spontaneous Self-Correction
============================================================

Zhao, Xu, Wang, Chen, Jin, Tan, Yu, Zhao, He, Chandar, Zhu. Jun 2025. arXiv

Introducen SPOC: el modelo propone soluciones y las verifica "on the fly" en una
misma inferencia. Mejora significativamente benchmarks de matemática (MATH500, AMC23, AIME).

Técnica principal: Spontaneous self-correction during inference.
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
class SPOCConfig:
    """Configuración para SPOC."""
    hidden_dim: int = 512
    max_correction_iterations: int = 3
    correction_threshold: float = 0.7
    use_self_verification: bool = True
    use_iterative_refinement: bool = True
    correction_strength: float = 0.5


class SolutionProposer(nn.Module):
    """
    Propone soluciones iniciales.
    """
    
    def __init__(self, config: SPOCConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Proposer network
        self.proposer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Initialize
        for module in self.proposer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized SolutionProposer")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Propose initial solution.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            proposed_solution: [batch, seq, hidden_dim]
        """
        # Propose
        solution = self.proposer(hidden_states)
        
        return solution


class SelfVerifier(nn.Module):
    """
    Verifica soluciones propuestas.
    """
    
    def __init__(self, config: SPOCConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Verifier network
        self.verifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # problem + solution
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
        
        logger.info("Initialized SelfVerifier")
    
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


class SelfCorrector(nn.Module):
    """
    Corrige soluciones basándose en verificación.
    """
    
    def __init__(self, config: SPOCConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Corrector network
        self.corrector = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),  # solution + error_signal
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Initialize
        for module in self.corrector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized SelfCorrector")
    
    def forward(self, solution: torch.Tensor, error_signal: torch.Tensor) -> torch.Tensor:
        """
        Correct solution based on error signal.
        
        Args:
            solution: [batch, seq, hidden_dim]
            error_signal: [batch, seq, hidden_dim] - difference between problem and solution
            
        Returns:
            corrected_solution: [batch, seq, hidden_dim]
        """
        # Combine solution and error signal
        combined = torch.cat([solution, error_signal], dim=-1)  # [batch, seq, hidden_dim * 2]
        
        # Correct
        correction = self.corrector(combined)
        
        # Apply correction
        corrected = solution + self.config.correction_strength * correction
        
        return corrected


class SPOCModule(nn.Module):
    """
    Módulo SPOC completo.
    """
    
    def __init__(self, config: SPOCConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Components
        self.proposer = SolutionProposer(config)
        
        if config.use_self_verification:
            self.verifier = SelfVerifier(config)
        else:
            self.verifier = None
        
        if config.use_iterative_refinement:
            self.corrector = SelfCorrector(config)
        else:
            self.corrector = None
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Metrics
        self.register_buffer('avg_verification_score', torch.tensor(0.5))
        self.register_buffer('correction_iterations', torch.tensor(0.0))
        self.register_buffer('self_correction_rate', torch.tensor(0.0))
        
        logger.info(f"Initialized SPOCModule: max_iterations={config.max_correction_iterations}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: propose, verify, correct iteratively.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with correction info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Propose initial solution
        solution = self.proposer(hidden_states)  # [batch, seq, hidden_dim]
        
        # Iterative correction
        num_corrections = 0
        verification_score = 0.5
        
        for iteration in range(self.config.max_correction_iterations):
            # Verify solution
            if self.verifier is not None:
                verification = self.verifier(hidden_states, solution)  # [batch, seq]
                avg_verification = verification.mean().item()
                verification_score = avg_verification
                
                self.avg_verification_score = 0.9 * self.avg_verification_score + 0.1 * avg_verification
                
                # Check if correction needed
                if avg_verification >= self.config.correction_threshold:
                    break  # Solution is good enough
                
                # Compute error signal
                error_signal = hidden_states - solution  # [batch, seq, hidden_dim]
                
                # Correct solution
                if self.corrector is not None:
                    solution = self.corrector(solution, error_signal)
                    num_corrections += 1
            else:
                break
        
        # Update metrics
        self.correction_iterations = 0.9 * self.correction_iterations + 0.1 * num_corrections
        
        # Self-correction rate (how often we correct)
        correction_rate = num_corrections / self.config.max_correction_iterations
        self.self_correction_rate = 0.9 * self.self_correction_rate + 0.1 * correction_rate
        
        # Project output
        output = self.output_projection(solution)
        
        # Combine with original
        output = hidden_states + 0.3 * output
        
        metadata = {
            'verification_score': verification_score,
            'num_corrections': num_corrections,
            'correction_rate': correction_rate
        }
        
        return output, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'avg_verification_score': self.avg_verification_score.item(),
            'correction_iterations': self.correction_iterations.item(),
            'self_correction_rate': self.self_correction_rate.item(),
            'max_iterations': self.config.max_correction_iterations
        }


if __name__ == "__main__":
    config = SPOCConfig(
        hidden_dim=512,
        max_correction_iterations=3,
        use_self_verification=True
    )
    module = SPOCModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ SPOC test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Verification score: {metadata['verification_score']:.4f}")
    print(f"   Corrections: {metadata['num_corrections']}")



