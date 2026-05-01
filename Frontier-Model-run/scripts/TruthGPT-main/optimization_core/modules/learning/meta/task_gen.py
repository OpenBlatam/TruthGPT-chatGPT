"""
Task Generator
==============

Generates few-shot learning tasks for meta-learning.
"""
import torch
import logging
from typing import Dict, Any, List, Tuple

from .config import MetaLearningConfig
from .enums import TaskDistribution

logger = logging.getLogger(__name__)

class TaskGenerator:
    """Generate meta-learning tasks with support and query sets."""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.task_counter = 0
        self.task_history = []
        logger.info("✅ Task Generator initialized")
    
    def generate_task(self, input_dim: int = 100, output_dim: int = 10) -> Dict[str, Any]:
        """Generate a single meta-learning task."""
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        support_data, support_labels = self._generate_data(input_dim, output_dim, self.config.support_size)
        query_data, query_labels = self._generate_data(input_dim, output_dim, self.config.query_size)
        
        task = {
            'task_id': task_id,
            'support_data': support_data,
            'support_labels': support_labels,
            'query_data': query_data,
            'query_labels': query_labels,
            'num_ways': self.config.num_ways,
            'input_dim': input_dim,
            'output_dim': output_dim
        }
        
        self.task_history.append(task)
        return task
    
    def generate_task_batch(self, batch_size: int, input_dim: int = 100, output_dim: int = 10) -> List[Dict[str, Any]]:
        """Generate a batch of tasks."""
        return [self.generate_task(input_dim, output_dim) for _ in range(batch_size)]

    def _generate_data(self, input_dim: int, output_dim: int, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthetic data generation based on task distribution."""
        dist = self.config.task_distribution
        
        if dist == TaskDistribution.UNIFORM:
            data = torch.randn(num_samples, input_dim)
            labels = torch.randint(0, output_dim, (num_samples,))
            
        elif dist == TaskDistribution.GAUSSIAN:
            # Multi-class gaussian blobs
            data_list = []
            labels_list = []
            samples_per_class = max(1, num_samples // self.config.num_ways)
            
            for class_id in range(self.config.num_ways):
                mean = torch.randn(input_dim)
                cov = torch.eye(input_dim) * 0.1
                class_data = torch.distributions.MultivariateNormal(mean, cov).sample((samples_per_class,))
                class_labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
                data_list.append(class_data)
                labels_list.append(class_labels)
                
            data = torch.cat(data_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            
            # Shuffle resulting set
            perm = torch.randperm(len(data))
            data, labels = data[perm], labels[perm]
            
        else:
            data = torch.randn(num_samples, input_dim)
            labels = torch.randint(0, output_dim, (num_samples,))
            
        return data, labels

