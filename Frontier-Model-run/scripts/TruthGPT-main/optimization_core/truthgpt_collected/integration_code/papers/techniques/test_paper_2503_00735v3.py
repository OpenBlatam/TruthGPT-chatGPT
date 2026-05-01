#!/usr/bin/env python3
"""
End-to-end tests for paper_2503_00735v3.py
"""

import unittest
import torch
import torch.nn as nn
import logging
import sys
import os

# Add current directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paper_2503_00735v3 import (
    Paper2503_00735v3Config,
    Paper2503_00735v3Module,
    EfficientFlashAttention,
    TruthGPT_Paper2503_00735v3_Integration
)

logging.basicConfig(level=logging.INFO)

class TestPaper2503_00735v3(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 64  # Small dim for tests
        self.num_heads = 4
        self.chunk_size = 8
        self.seq_len = 32
        self.batch_size = 2
        
        self.config = Paper2503_00735v3Config(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            chunk_size=self.chunk_size,
            use_efficient_attention=True
        )
        self.module = Paper2503_00735v3Module(self.config)

    def test_config_defaults(self):
        """Test default configuration values."""
        config = Paper2503_00735v3Config()
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.chunk_size, 64)
        self.assertTrue(config.use_efficient_attention)

    def test_efficient_flash_attention_shapes(self):
        """Test EfficientFlashAttention output shapes."""
        attention = EfficientFlashAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            chunk_size=self.chunk_size
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        output = attention(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))

    def test_chunked_attention_logic(self):
        """Test that chunked attention produces valid output and updates metrics."""
        attention = EfficientFlashAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            chunk_size=self.chunk_size
        )
        
        # Create Q, K, V manually to test the internal method if needed, 
        # but calling forward is easier and covers it.
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Initial metric check
        self.assertEqual(attention.chunk_utilization.item(), 0.0)
        
        output = attention(x)
        
        # Check metric updated
        metrics = attention.get_metrics()
        self.assertGreater(metrics['chunk_utilization'], 0.0)
        
        # Check output validity
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_attention_mask(self):
        """Test EfficientFlashAttention with attention mask."""
        attention = EfficientFlashAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            chunk_size=self.chunk_size
        )
        
        # Mocking the internal call since forward() in the provided code 
        # doesn't seem to take attention_mask explicitly in the public interface 
        # (it does project Q, K, V then calls chunked_attention).
        # Looking at the code: 
        # forward(self, x) -> calls chunked_attention(Q, K, V)
        # chunked_attention takes optional attention_mask.
        # However, the forward method in EfficientFlashAttention implementation 
        # DOES NOT accept an attention_mask argument and DOES NOT pass it to chunked_attention.
        # It seems the class supports it internally but the forward doesn't expose it.
        # I will test chunked_attention directly.
        
        Q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.hidden_dim // self.num_heads)
        K = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.hidden_dim // self.num_heads)
        V = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.hidden_dim // self.num_heads)
        
        mask = torch.ones(self.batch_size, self.seq_len)
        # Mask out last token
        mask[:, -1] = 0
        
        output = attention.chunked_attention(Q, K, V, attention_mask=mask)
        self.assertEqual(output.shape, Q.shape)

    def test_full_module_forward(self):
        """Test the full Paper2503_00735v3Module forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        output = self.module(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_integration_wrapper(self):
        """Test the integration with a mock base model."""
        
        class MockBaseModel(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear = nn.Linear(hidden_dim, hidden_dim)
            
            def forward(self, x):
                return self.linear(x)
        
        base_model = MockBaseModel(self.hidden_dim)
        integration = TruthGPT_Paper2503_00735v3_Integration(base_model, self.config)
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        output = integration(x)
        
        self.assertEqual(output.shape, x.shape)

    def test_different_sequence_lengths(self):
        """Test with sequence length not divisible by chunk size."""
        # chunk_size is 8. Try seq_len 10.
        seq_len = 10
        x = torch.randn(self.batch_size, seq_len, self.hidden_dim)
        output = self.module(x)
        self.assertEqual(output.shape, (self.batch_size, seq_len, self.hidden_dim))

if __name__ == '__main__':
    unittest.main()



