"""
Test data factory for generating synthetic datasets and test cases
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random

class TestDataFactory:
    """Factory for creating test data and synthetic datasets"""
    
    @staticmethod
    def create_random_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create random tensor with specified shape and dtype"""
        return torch.randn(shape, dtype=dtype)
    
    @staticmethod
    def create_attention_data(batch_size: int = 2, seq_len: int = 128, d_model: int = 512) -> Dict[str, torch.Tensor]:
        """Create attention test data"""
        return {
            'query': torch.randn(batch_size, seq_len, d_model),
            'key': torch.randn(batch_size, seq_len, d_model),
            'value': torch.randn(batch_size, seq_len, d_model),
            'mask': torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
        }
    
    @staticmethod
    def create_mlp_data(batch_size: int = 2, seq_len: int = 128, d_model: int = 512) -> torch.Tensor:
        """Create MLP test data"""
        return torch.randn(batch_size, seq_len, d_model)
    
    @staticmethod
    def create_optimization_data(
        num_params: int = 1000,
        num_epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Create optimization test data"""
        return {
            'parameters': [torch.randn(num_params, requires_grad=True)],
            'losses': [random.uniform(0.1, 10.0) for _ in range(num_epochs)],
            'gradients': [torch.randn(num_params) for _ in range(num_epochs)],
            'learning_rates': [0.001 * (0.9 ** i) for i in range(num_epochs)],
            'batch_size': batch_size
        }
    
    @staticmethod
    def create_kv_cache_data(
        batch_size: int = 2,
        seq_len: int = 128,
        d_model: int = 512,
        num_layers: int = 6
    ) -> Dict[str, torch.Tensor]:
        """Create KV cache test data"""
        return {
            'keys': torch.randn(batch_size, num_layers, seq_len, d_model),
            'values': torch.randn(batch_size, num_layers, seq_len, d_model),
            'positions': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        }
    
    @staticmethod
    def create_transformer_data(
        batch_size: int = 2,
        seq_len: int = 128,
        d_model: int = 512,
        vocab_size: int = 10000
    ) -> Dict[str, torch.Tensor]:
        """Create transformer test data"""
        return {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'token_type_ids': torch.zeros(batch_size, seq_len, dtype=torch.long)
        }
    
    @staticmethod
    def create_quantization_data(
        tensor_shape: Tuple[int, ...] = (32, 128),
        num_bits: int = 8
    ) -> Dict[str, Any]:
        """Create quantization test data"""
        return {
            'tensor': torch.randn(tensor_shape),
            'scale': torch.tensor(0.1),
            'zero_point': torch.tensor(0),
            'num_bits': num_bits,
            'quantized': torch.randint(0, 2**num_bits, tensor_shape, dtype=torch.uint8)
        }
    
    @staticmethod
    def create_benchmark_data(
        model_sizes: List[Tuple[int, int, int]] = [(512, 512, 2048), (1024, 1024, 4096)],
        sequence_lengths: List[int] = [128, 256, 512]
    ) -> List[Dict[str, Any]]:
        """Create benchmark test data"""
        benchmark_data = []
        for d_model, hidden_size, vocab_size in model_sizes:
            for seq_len in sequence_lengths:
                benchmark_data.append({
                    'd_model': d_model,
                    'hidden_size': hidden_size,
                    'vocab_size': vocab_size,
                    'seq_len': seq_len,
                    'input_data': torch.randint(0, vocab_size, (2, seq_len)),
                    'expected_shape': (2, seq_len, d_model)
                })
        return benchmark_data
    
    @staticmethod
    def create_error_cases() -> List[Dict[str, Any]]:
        """Create error test cases"""
        return [
            {
                'name': 'invalid_tensor_shape',
                'data': torch.randn(1, 2, 3, 4, 5),  # Too many dimensions
                'expected_error': ValueError
            },
            {
                'name': 'negative_dimensions',
                'data': torch.randn(-1, 128),  # Negative dimension
                'expected_error': RuntimeError
            },
            {
                'name': 'mismatched_batch_sizes',
                'query': torch.randn(2, 128, 512),
                'key': torch.randn(3, 128, 512),  # Different batch size
                'expected_error': RuntimeError
            }
        ]
    
    @staticmethod
    def create_performance_data(
        sizes: List[int] = [128, 256, 512, 1024, 2048]
    ) -> List[Dict[str, Any]]:
        """Create performance test data"""
        performance_data = []
        for size in sizes:
            performance_data.append({
                'size': size,
                'tensor': torch.randn(size, size),
                'expected_time': size * size * 1e-6,  # Rough estimate
                'memory_usage': size * size * 4  # 4 bytes per float32
            })
        return performance_data





