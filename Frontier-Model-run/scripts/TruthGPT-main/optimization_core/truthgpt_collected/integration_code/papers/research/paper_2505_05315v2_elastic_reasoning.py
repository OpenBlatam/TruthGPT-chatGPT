#!/usr/bin/env python3
"""
Paper: 2505.05315v2 - Elastic Reasoning
========================================

Scalable Chain of Thoughts via Elastic Reasoning

Este paper propone un framework para reasoning escalable que separa explícitamente
el reasoning en dos fases: thinking y solution, con budgets independientes.

Técnica principal: Separate budgeting para thinking y solution phases.

Basado en: https://arxiv.org/html/2505.05315v2

DETALLES EXACTOS DEL PAPER:
- Separa reasoning en dos fases: thinking y solution
- Budget-constrained inference: |y| ≤ c donde c = t + s
- t = budget para thinking phase
- s = budget para solution phase
- GRPO training con budget-constrained rollout
- El modelo puede generalizar a budgets arbitrarios sin entrenamiento adicional
- Thinking típicamente representa >90% de los tokens
- Solution es conciso pero completo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants & Paper Reference Data ---

EPSILON = 1e-8
EMA_ALPHA = 0.1

PAPER_REFERENCE_METRICS = {
    'math500_pass1': 0.836,
    'math500_tokens': 1619,
    'aime2024_reduction': 0.321,
    'overall_reduction': 0.30,
    'convergence_steps': 150,
    'thinking_percentage': 0.90,
    'training_setup': {
        'base_models': ['DeepScaleR-1.5B-Preview', 'DeepCoder-14B-Preview'],
        'base_models_source': ['DeepSeekR1-Distill-Qwen-1.5B', 'DeepSeekR1-Distill-Qwen-14B'],
        'convergence_step': 150,
        'initial_pass1': 0.07,
        'final_pass1': 0.20,
        'evaluation_frequency': 10,
        'math_datasets': ['AIME (1984-2023)', 'AMC', 'Omni-Math', 'STILL'],
        'code_datasets': ['TACO', 'SYNTHETIC-1', 'LiveCodeBench (2023/05/01-2024/07/31)'],
        'math_eval_datasets': ['AIME 2024', 'MATH500', 'AMC', 'Olympiad-Bench', 'Minerva Math'],
        'code_eval_datasets': ['LiveCodeBench (2024/08/01-2025/02/01)', 'Codeforces', 'HumanEval+']
    },
    'results_exact': {
        'math500': {
            'e1_pass1': 0.836,
            'e1_tokens': 1619,
            'l1_exact_pass1': 0.799,
            'l1_exact_tokens': 1959,
            'l1_max_pass1': 0.836,
            'l1_max_tokens': 1796
        },
        'aime2024': {
            'e1_degradation': 0.060,
            'l1_max_degradation': 0.129,
            'l1_exact_degradation': 0.168,
            'token_reduction': 0.321
        },
        'code_results': {
            'e1_code_14b_improvement': 0.003,
            'token_reduction': 0.374,
            'original_tokens': 17815,
            'e1_tokens': 11145,
            'accuracy_below_4k': 0.10
        },
        'ablation': {
            'solution_gain': 0.087,
            'budget_05k_1k': (500, 1000),
            'budget_1k_1k': (1000, 1000)
        }
    },
    'budget_configurations': {
        'training': {
            'fixed_budget': (1000, 1000),
            'note': 'Fixed budget during GRPO training'
        },
        'inference': {
            'unconstrained': 24000,
            'constrained_examples': [
                {'thinking': 1000, 'solution': 1000, 'total': 2000},
                {'thinking': 500, 'solution': 1000, 'total': 1500},
                {'thinking': 2000, 'solution': 1000, 'total': 3000}
            ],
            'code_budgets': [
                {'thinking': 1000, 'solution': 1000, 'total': 2000},
                {'thinking': 2000, 'solution': 1000, 'total': 3000},
                {'thinking': 4000, 'solution': 1000, 'total': 5000}
            ]
        }
    },
    'token_usage_exact': {
        'e1_code_14b': {
            'original_tokens': 17815,
            'e1_tokens': 11145,
            'reduction': 0.374,
            'reduction_tokens': 6670
        },
        'e1_math_15b': {
            'aime2024_reduction': 0.321,
            'overall_reduction': 0.30
        }
    },
    'accuracy_thresholds': {
        'deepcoder_14b_preview': {
            'below_4k_budget': 0.10,
            'note': 'Even with separate budgeting'
        },
        'e1_code_14b': {
            'improvement_unconstrained': 0.003,
            'scalability': 'Performance improves steadily as budget increases'
        }
    },
    'special_tokens': {
        'thinking_start': '<think>',
        'thinking_end': '</think>',
        'structure': 'y = (<think> intermediate reasoning </think>, solution)',
        'note': 'Following prior work, we denote the reasoning phase using special tokens such as <think> and </think>'
    },
    'inference_algorithm_steps': [
        '1. The model begins generating within a <think> block',
        '2. If the model emits </think> before reaching the budget t, we transition immediately to the solution phase',
        '3. If the budget t is exhausted before </think> is emitted, we forcibly terminate the reasoning by appending </think>',
        '4. The model then continues generating the solution segment, up to a maximum of s tokens'
    ],
    'datasets_exact': {
        'math_training': {
            'datasets': ['AIME (1984-2023)', 'AMC', 'Omni-Math', 'STILL'],
            'note': 'Following same datasets as DeepScaleR and DeepCoder papers'
        },
        'code_training': {
            'datasets': ['TACO', 'SYNTHETIC-1', 'LiveCodeBench (2023/05/01-2024/07/31)'],
            'note': 'Date ranges specified exactly'
        },
        'math_evaluation': {
            'datasets': ['AIME 2024', 'MATH500', 'AMC', 'Olympiad-Bench', 'Minerva Math']
        },
        'code_evaluation': {
            'datasets': ['LiveCodeBench (2024/08/01-2025/02/01)', 'Codeforces', 'HumanEval+'],
            'note': 'Date ranges specified exactly'
        }
    },
    'base_models_exact': {
        'math': {
            'name': 'DeepScaleR-1.5B-Preview',
            'source': 'DeepSeekR1-Distill-Qwen-1.5B',
            'method': 'Fine-tuned through iterative context lengthening'
        },
        'code': {
            'name': 'DeepCoder-14B-Preview',
            'source': 'DeepSeekR1-Distill-Qwen-14B',
            'method': 'Fine-tuned through iterative context lengthening'
        }
    },
    'repository': {
        'url': 'https://github.com/SalesforceAIResearch/Elastic-Reasoning',
        'organization': 'Salesforce AI Research'
    },
    'key_observations': {
        'thinking_percentage': 'y^think typically accounts for over 90% of the total tokens',
        'solution_character': 'y^solution provides a concise summary and final answer',
        'key_insight': 'Even when the reasoning phase is forcibly terminated (e.g., by inserting </think>), the model is still capable of producing a coherent—and often correct—solution',
        'generalization': 'Training with a fixed budget constraint (e.g., (1K, 1K)) enables the model to generalize effectively to a wide range of budget configurations',
        'solution_improvement': 'The improvement in solution generation plays a central role in this generalization, allowing the model to adapt even when the available thinking tokens are reduced'
    },
    'mathematical_notation': {
        'output_structure': 'y = (y^think, y^solution)',
        'budget_constraint': '|y| ≤ c',
        'budget_components': 'c = t + s',
        'inference_budget': 'c_i = t_i + s*',
        'thinking_tokens': 'y^think',
        'solution_tokens': 'y^solution'
    }
}


@dataclass
class ElasticReasoningConfig:
    """Configuración para Elastic Reasoning."""
    hidden_dim: int = 512
    thinking_budget: int = 1000  # t*: fixed thinking budget during training
    solution_budget: int = 100   # s*: fixed solution budget during training
    use_budget_constrained_rollout: bool = True
    enable_adaptive_budgeting: bool = True
    dropout_rate: float = 0.1


class ElasticReasoningModule(nn.Module):
    """
    Módulo de Elastic Reasoning basado exactamente en el paper 2505.05315v2.
    
    Técnica:
    - Separa reasoning en thinking y solution phases
    - Budget-constrained rollout durante training
    - Separate budgeting para inference
    """
    
    def __init__(self, config: ElasticReasoningConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Thinking phase encoder
        self.thinking_encoder = self._build_encoder()
        
        # Solution phase encoder
        self.solution_encoder = self._build_encoder()
        
        # Transition predictor (predicts when to transition from thinking to solution)
        self.transition_predictor = nn.Linear(config.hidden_dim, 1)
        
        self._init_weights()
        
        # Metrics (Buffers for persistence without gradients)
        self.register_buffer('thinking_tokens_used', torch.tensor(0.0))
        self.register_buffer('solution_tokens_used', torch.tensor(0.0))
        self.register_buffer('avg_thinking_budget', torch.tensor(float(config.thinking_budget)))
        self.register_buffer('avg_solution_budget', torch.tensor(float(config.solution_budget)))
        self.register_buffer('transition_accuracy', torch.tensor(0.5))
        
        logger.info(f"Initialized ElasticReasoningModule: t*={config.thinking_budget}, s*={config.solution_budget}")

    def _build_encoder(self) -> nn.Sequential:
        """Construye el encoder secuencial para las fases de reasoning."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        )

    def _init_weights(self):
        """Inicializa los pesos usando Xavier uniform."""
        for module in [self.thinking_encoder, self.solution_encoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.transition_predictor.weight)
        if self.transition_predictor.bias is not None:
            nn.init.zeros_(self.transition_predictor.bias)

    def budget_constrained_rollout(self, hidden_states: torch.Tensor, 
                                  thinking_budget: Optional[int] = None,
                                  solution_budget: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Budget-constrained rollout (Section 3.2.2).
        
        Algoritmo:
        1. Genera thinking tokens hasta `thinking_budget` o `</think>`.
        2. Genera solution tokens hasta `solution_budget`.
        3. Constraint: |y| ≤ t + s
        """
        # 1. Validation
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, hidden_dim], got {hidden_states.dim()}D")
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Hidden dimension mismatch: input {hidden_dim} vs config {self.hidden_dim}")
        
        # 2. Determine Budgets
        t_budget = thinking_budget if thinking_budget is not None else self.config.thinking_budget
        s_budget = solution_budget if solution_budget is not None else self.config.solution_budget
        
        if t_budget <= 0 or s_budget <= 0:
            raise ValueError(f"Budgets must be positive. Got t={t_budget}, s={s_budget}")
        
        # 3. Phase Segmentation (Simulation on existing sequence)
        # In a real generation loop, this would stop generation. Here we slice the tensor.
        thinking_end = min(t_budget, seq_len)
        solution_start = thinking_end
        solution_end = min(solution_start + s_budget, seq_len)
        
        # 4. Process Thinking Phase
        thinking_states = hidden_states[:, :thinking_end, :]
        thinking_output = self.thinking_encoder(thinking_states)
        
        # Transition prediction (auxiliary task)
        transition_scores = self.transition_predictor(thinking_output).squeeze(-1)
        transition_probs = torch.sigmoid(transition_scores)
        
        # 5. Process Solution Phase
        if solution_start < seq_len:
            solution_states = hidden_states[:, solution_start:solution_end, :]
            solution_output = self.solution_encoder(solution_states)
        else:
            solution_output = torch.zeros(
                batch_size, 0, self.hidden_dim, 
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        # 6. Combine Outputs
        output = torch.zeros_like(hidden_states)
        output[:, :thinking_end, :] = thinking_output
        
        sol_len = solution_output.size(1)
        if sol_len > 0:
            output[:, solution_start:solution_start+sol_len, :] = solution_output
        
        # 7. Update Metrics
        self._update_metrics(thinking_end, sol_len, t_budget, s_budget)
        
        metadata = self._create_metadata(
            thinking_end, sol_len, t_budget, s_budget, transition_probs
        )
        
        return output, metadata

    def _update_metrics(self, t_used: int, s_used: int, t_budget: int, s_budget: int):
        """Actualiza las métricas usando Exponential Moving Average (EMA)."""
        alpha = EMA_ALPHA
        self.thinking_tokens_used = (1 - alpha) * self.thinking_tokens_used + alpha * float(t_used)
        self.solution_tokens_used = (1 - alpha) * self.solution_tokens_used + alpha * float(s_used)
        self.avg_thinking_budget = (1 - alpha) * self.avg_thinking_budget + alpha * float(t_budget)
        self.avg_solution_budget = (1 - alpha) * self.avg_solution_budget + alpha * float(s_budget)

    def _create_metadata(self, t_used: int, s_used: int, t_budget: int, s_budget: int, 
                        transition_probs: torch.Tensor) -> Dict[str, Any]:
        """Genera metadatos detallados de la ejecución."""
        total_used = t_used + s_used
        total_budget = t_budget + s_budget
        return {
            'thinking_tokens': t_used,
            'solution_tokens': s_used,
            'total_tokens': total_used,
            'thinking_budget': t_budget,
            'solution_budget': s_budget,
            'total_budget': total_budget,
            'budget_satisfied': total_used <= total_budget,
            'budget_utilization': total_used / (total_budget + EPSILON),
            'thinking_ratio': t_used / (total_used + EPSILON),
            'transition_probs_mean': transition_probs.mean().item(),
            'transition_probs_std': transition_probs.std().item()
        }

    def forward(self, hidden_states: torch.Tensor,
                thinking_budget: Optional[int] = None,
                solution_budget: Optional[int] = None,
                is_training: Optional[bool] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass handling both training and inference modes."""
        
        training_mode = is_training if is_training is not None else self.training
        
        if training_mode and self.config.use_budget_constrained_rollout:
            return self.budget_constrained_rollout(
                hidden_states,
                thinking_budget=self.config.thinking_budget,
                solution_budget=self.config.solution_budget
            )
        
        # Inference or dynamic budget mode
        return self.budget_constrained_rollout(hidden_states, thinking_budget, solution_budget)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current runtime metrics."""
        thinking_ratio = self.thinking_tokens_used.item() / (
            self.thinking_tokens_used.item() + self.solution_tokens_used.item() + EPSILON
        )
        return {
            'runtime_metrics': {
                'thinking_tokens_used': self.thinking_tokens_used.item(),
                'solution_tokens_used': self.solution_tokens_used.item(),
                'avg_thinking_budget': self.avg_thinking_budget.item(),
                'avg_solution_budget': self.avg_solution_budget.item(),
                'thinking_ratio': thinking_ratio,
            },
            'paper_reference': PAPER_REFERENCE_METRICS
        }


class ElasticReasoningModuleWrapper(nn.Module):
    """Wrapper para integración con TruthGPT."""
    
    def __init__(self, config: ElasticReasoningConfig):
        super().__init__()
        self.module = ElasticReasoningModule(config)
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass simplificado para integración."""
        output, _ = self.module(hidden_states, **kwargs)
        return output
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        return self.module.get_metrics()


if __name__ == "__main__":
    config = ElasticReasoningConfig(
        hidden_dim=512,
        thinking_budget=1000,
        solution_budget=100
    )
    
    module = ElasticReasoningModule(config)
    x = torch.randn(2, 1200, config.hidden_dim)
    
    # Training mode
    output_train, meta_train = module(x, is_training=True)
    print(f"✅ Training mode: Thinking={meta_train['thinking_tokens']}, Solution={meta_train['solution_tokens']}")
    
    # Inference mode with arbitrary budget
    output_inf, meta_inf = module(x, thinking_budget=500, solution_budget=100, is_training=False)
    print(f"✅ Inference mode: Thinking={meta_inf['thinking_tokens']}, Solution={meta_inf['solution_tokens']}")
    
    metrics = module.get_metrics()
    print(f"✅ Metrics: {metrics}")


