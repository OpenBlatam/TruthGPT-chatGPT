"""
Mock components for testing TruthGPT optimization core
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import time
import random

class MockOptimizer:
    """Mock optimizer for testing"""
    
    def __init__(self, name: str = "MockOptimizer", learning_rate: float = 0.001):
        self.name = name
        self.learning_rate = learning_rate
        self.step_count = 0
        self.optimization_history = []
        
    def step(self, loss: torch.Tensor) -> Dict[str, Any]:
        """Mock optimization step"""
        self.step_count += 1
        self.optimization_history.append({
            'step': self.step_count,
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'learning_rate': self.learning_rate,
            'timestamp': time.time()
        })
        return {'optimized': True, 'step': self.step_count}
    
    def zero_grad(self):
        """Mock gradient zeroing"""
        pass
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'total_steps': self.step_count,
            'current_lr': self.learning_rate,
            'history_length': len(self.optimization_history)
        }

class MockModel(nn.Module):
    """Mock model for testing"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 1024, output_size: int = 512):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
        self.forward_count = 0
        self.performance_metrics = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mock forward pass"""
        start_time = time.time()
        self.forward_count += 1
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        end_time = time.time()
        self.performance_metrics.append({
            'forward_time': end_time - start_time,
            'input_shape': x.shape,
            'forward_count': self.forward_count
        })
        
        return x
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'forward_count': self.forward_count,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }

class MockAttention(nn.Module):
    """Mock attention mechanism for testing"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.attention_count = 0
        self.attention_weights = []
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mock attention forward pass"""
        self.attention_count += 1
        
        batch_size, seq_len, d_model = query.shape
        
        # Simple attention computation
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # Mock attention weights
        attention_weights = torch.softmax(torch.randn(batch_size, self.n_heads, seq_len, seq_len), dim=-1)
        self.attention_weights.append(attention_weights.mean().item())
        
        # Mock output
        output = self.out_linear(v)
        
        return output, attention_weights
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get attention statistics"""
        return {
            'attention_count': self.attention_count,
            'avg_attention_weight': sum(self.attention_weights) / len(self.attention_weights) if self.attention_weights else 0,
            'd_model': self.d_model,
            'n_heads': self.n_heads
        }

class MockMLP(nn.Module):
    """Mock MLP for testing"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 2048, output_size: int = 512):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
        self.forward_count = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mock MLP forward pass"""
        self.forward_count += 1
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x
    
    def get_mlp_stats(self) -> Dict[str, Any]:
        """Get MLP statistics"""
        return {
            'forward_count': self.forward_count,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }

class MockDataset:
    """Mock dataset for testing"""
    
    def __init__(self, size: int = 1000, input_size: int = 512, output_size: int = 512):
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        
        # Generate synthetic data
        self.data = []
        for i in range(size):
            self.data.append({
                'input': torch.randn(input_size),
                'target': torch.randn(output_size),
                'index': i
            })
        
        self.current_index = 0
        
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]
    
    def get_batch(self, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Get a batch of data"""
        batch = []
        for _ in range(batch_size):
            batch.append(self.data[self.current_index % self.size])
            self.current_index += 1
        
        return {
            'input': torch.stack([item['input'] for item in batch]),
            'target': torch.stack([item['target'] for item in batch])
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'size': self.size,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'current_index': self.current_index
        }

class MockKVCache:
    """Mock KV cache for testing"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def put(self, key: str, value: torch.Tensor) -> bool:
        """Put value in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
        return True
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get value from cache"""
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }





