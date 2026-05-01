"""
Reptile Algorithm
=================

Reptile meta-learning implementation (first-order meta-optimization).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any, List

from ..config import MetaLearningConfig

logger = logging.getLogger(__name__)

class Reptile:
    """Reptile meta-learning algorithm implementation."""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.meta_optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
        logger.info("✅ Reptile initialized")
    
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reptile update: interpolate weights toward task-adapted solutions."""
        total_eval_loss = 0.0
        original_params = [p.clone().detach() for p in self.model.parameters()]
        
        adapted_weights_list = []
        task_losses = []
        
        for task in tasks:
            # Inner loop: update weights permanently for this task simulation
            adapted_params = self._inner_loop(task)
            adapted_weights_list.append(adapted_params)
            
            # Evaluate (just for stats)
            with torch.no_grad():
                l = self._evaluate_performance(task)
                task_losses.append(l)
                total_eval_loss += l
                
            # Reset for next task
            self._set_params(original_params)
            
        # Meta-update: Direction of average shift
        for i, param in enumerate(self.model.parameters()):
            # Aggregate shift from all tasks
            total_shift = torch.zeros_like(param)
            for adapted in adapted_weights_list:
                total_shift += (adapted[i] - param.data)
            
            # Apply update via pseudo-gradient (negative direction for optimizer)
            param.grad = -total_shift / len(tasks)
            
        self.meta_optimizer.step()
        
        return {
            'meta_loss': total_eval_loss / len(tasks),
            'task_losses': [l.item() for l in task_losses]
        }
        
    def _inner_loop(self, task: Dict[str, Any]) -> List[torch.Tensor]:
        """Perform SGD steps on the support set."""
        opt = optim.SGD(self.model.parameters(), lr=self.config.inner_lr)
        data, labels = task['support_data'], task['support_labels']
        
        for _ in range(self.config.inner_steps):
            opt.zero_grad()
            out = self.model(data)
            loss = nn.CrossEntropyLoss()(out, labels)
            loss.backward()
            opt.step()
            
        return [p.clone().detach() for p in self.model.parameters()]

    def _evaluate_performance(self, task: Dict[str, Any]) -> torch.Tensor:
        """Evaluate current model state on query set."""
        data, labels = task['query_data'], task['query_labels']
        out = self.model(data)
        return nn.CrossEntropyLoss()(out, labels)

    def _set_params(self, params: List[torch.Tensor]):
        """Helper to restore weights."""
        for p, target in zip(self.model.parameters(), params):
            p.data.copy_(target)
            
    def adapt_to_task(self, task: Dict[str, Any], steps: int = None) -> nn.Module:
        """Clone and adapt model to a specific task."""
        m_copy = type(self.model)()
        m_copy.load_state_dict(self.model.state_dict())
        
        opt = optim.SGD(m_copy.parameters(), lr=self.config.inner_lr)
        data, labels = task['support_data'], task['support_labels']
        
        for _ in range(steps or self.config.inner_steps):
            opt.zero_grad()
            nn.CrossEntropyLoss()(m_copy(data), labels).backward()
            opt.step()
            
        return m_copy

