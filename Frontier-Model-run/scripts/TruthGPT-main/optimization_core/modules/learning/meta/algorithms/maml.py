"""
MAML Algorithm
==============

Model-Agnostic Meta-Learning implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any, List

from ..config import MetaLearningConfig

logger = logging.getLogger(__name__)

class MAML:
    """Model-Agnostic Meta-Learning engine."""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.meta_optimizer = optim.Adam(model.parameters(), lr=config.meta_lr)
        
        # Store initial state for reference
        self.initial_params = [p.clone().detach() for p in model.parameters()]
        logger.info("✅ MAML initialized")
    
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Outer loop update: minimize query loss across a batch of tasks."""
        meta_loss = 0.0
        task_losses = []
        original_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for task in tasks:
            # 1. Inner loop adaptation
            adapted_state = self._inner_loop_adaptation(task)
            
            # 2. Evaluation on query set
            loss = self._evaluate_query_loss(task, adapted_state)
            task_losses.append(loss)
            meta_loss += loss
            
            # Reset model to original for next task in batch
            self.model.load_state_dict(original_weights)
            
        avg_meta_loss = meta_loss / len(tasks)
        
        self.meta_optimizer.zero_grad()
        avg_meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            'meta_loss': avg_meta_loss.item(),
            'task_losses': [l.item() for l in task_losses],
            'num_tasks': len(tasks)
        }
        
    def adapt_to_task(self, task: Dict[str, Any], steps: int = None) -> nn.Module:
        """Create an adapted model instance for a specific task."""
        # Create a transient copy
        m_copy = type(self.model)()
        m_copy.load_state_dict(self.model.state_dict())
        
        inner_opt = optim.SGD(m_copy.parameters(), lr=self.config.inner_lr)
        s_data, s_labels = task['support_data'], task['support_labels']
        
        for _ in range(steps or self.config.inner_steps):
            inner_opt.zero_grad()
            out = m_copy(s_data)
            loss = nn.CrossEntropyLoss()(out, s_labels)
            loss.backward()
            inner_opt.step()
            
        return m_copy

    def _inner_loop_adaptation(self, task: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Perform temporary gradient steps on support set."""
        inner_opt = optim.SGD(self.model.parameters(), lr=self.config.inner_lr)
        s_data, s_labels = task['support_data'], task['support_labels']
        
        for _ in range(self.config.inner_steps):
            inner_opt.zero_grad()
            out = self.model(s_data)
            loss = nn.CrossEntropyLoss()(out, s_labels)
            loss.backward()
            inner_opt.step()
            
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def _evaluate_query_loss(self, task: Dict[str, Any], state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss on query set using adapted weights."""
        self.model.load_state_dict(state)
        q_data, q_labels = task['query_data'], task['query_labels']
        # Note: We need grad here for the meta-update (outer loop)
        out = self.model(q_data)
        return nn.CrossEntropyLoss()(out, q_labels)

