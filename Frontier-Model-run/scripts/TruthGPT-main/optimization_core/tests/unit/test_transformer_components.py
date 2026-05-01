"""
Unit tests for transformer components
Tests transformer blocks, layers, and related optimizations
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.mock_components import MockModel, MockAttention, MockMLP
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestTransformerBlocks(unittest.TestCase):
    """Test suite for transformer blocks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        self.config = TestUtils.create_test_config()
        
    def test_transformer_block_creation(self):
        """Test transformer block creation"""
        # Create mock transformer block
        class MockTransformerBlock(nn.Module):
            def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
                super().__init__()
                self.attention = MockAttention(d_model, n_heads)
                self.mlp = MockMLP(d_model, d_ff, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = x + self.dropout(attn_out)
                x = self.norm1(x)
                
                # MLP
                mlp_out = self.mlp(x)
                x = x + self.dropout(mlp_out)
                x = self.norm2(x)
                
                return x
        
        block = MockTransformerBlock()
        self.assertIsNotNone(block)
        
        # Test forward pass
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        output = block(data)
        
        self.assertEqual(output.shape, data.shape)
        
    def test_transformer_block_with_attention_mask(self):
        """Test transformer block with attention mask"""
        class MockTransformerBlock(nn.Module):
            def __init__(self, d_model=512, n_heads=8):
                super().__init__()
                self.attention = MockAttention(d_model, n_heads)
                self.mlp = MockMLP(d_model, 2048, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
            def forward(self, x, mask=None):
                attn_out, weights = self.attention(x, x, x, mask)
                x = x + attn_out
                x = self.norm1(x)
                
                mlp_out = self.mlp(x)
                x = x + mlp_out
                x = self.norm2(x)
                
                return x, weights
        
        block = MockTransformerBlock()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        mask = torch.ones(2, 128, 128, dtype=torch.bool)
        
        output, weights = block(data, mask)
        
        self.assertEqual(output.shape, data.shape)
        self.assertEqual(weights.shape, (2, 8, 128, 128))
        
    def test_transformer_block_gradient_flow(self):
        """Test transformer block gradient flow"""
        class MockTransformerBlock(nn.Module):
            def __init__(self, d_model=256):
                super().__init__()
                self.attention = MockAttention(d_model, 8)
                self.mlp = MockMLP(d_model, 1024, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
            def forward(self, x):
                attn_out, _ = self.attention(x, x, x)
                x = x + attn_out
                x = self.norm1(x)
                
                mlp_out = self.mlp(x)
                x = x + mlp_out
                x = self.norm2(x)
                
                return x
        
        block = MockTransformerBlock()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        data.requires_grad_(True)
        
        output = block(data)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(data.grad)
        self.assertTrue(TestAssertions.assert_gradient_flow([data.grad]))
        
    def test_transformer_block_performance(self):
        """Test transformer block performance"""
        class MockTransformerBlock(nn.Module):
            def __init__(self, d_model=512):
                super().__init__()
                self.attention = MockAttention(d_model, 8)
                self.mlp = MockMLP(d_model, 2048, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
            def forward(self, x):
                attn_out, _ = self.attention(x, x, x)
                x = x + attn_out
                x = self.norm1(x)
                
                mlp_out = self.mlp(x)
                x = x + mlp_out
                x = self.norm2(x)
                
                return x
        
        block = MockTransformerBlock()
        data = self.test_data.create_mlp_data(batch_size=4, seq_len=256, d_model=512)
        
        # Profile performance
        self.profiler.start_profile("transformer_block")
        output = block(data)
        metrics = self.profiler.end_profile()
        
        self.assertLess(metrics['execution_time'], 2.0)
        self.assertLess(metrics['memory_used'], 200)
        
    def test_transformer_block_different_sizes(self):
        """Test transformer block with different sizes"""
        sizes = [(128, 256), (256, 512), (512, 1024)]
        
        for d_model, d_ff in sizes:
            with self.subTest(size=(d_model, d_ff)):
                class MockTransformerBlock(nn.Module):
                    def __init__(self, d_model, d_ff):
                        super().__init__()
                        self.attention = MockAttention(d_model, 8)
                        self.mlp = MockMLP(d_model, d_ff, d_model)
                        self.norm1 = nn.LayerNorm(d_model)
                        self.norm2 = nn.LayerNorm(d_model)
                        
                    def forward(self, x):
                        attn_out, _ = self.attention(x, x, x)
                        x = x + attn_out
                        x = self.norm1(x)
                        
                        mlp_out = self.mlp(x)
                        x = x + mlp_out
                        x = self.norm2(x)
                        
                        return x
                
                block = MockTransformerBlock(d_model, d_ff)
                data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=d_model)
                output = block(data)
                
                self.assertEqual(output.shape, data.shape)

class TestLayerNormalization(unittest.TestCase):
    """Test suite for layer normalization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_layer_norm_basic(self):
        """Test basic layer normalization"""
        layer_norm = nn.LayerNorm(512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        output = layer_norm(data)
        
        self.assertEqual(output.shape, data.shape)
        
        # Check normalization properties
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        
        # Should be approximately normalized
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-5))
        
    def test_layer_norm_different_sizes(self):
        """Test layer normalization with different sizes"""
        sizes = [128, 256, 512, 1024]
        
        for size in sizes:
            with self.subTest(size=size):
                layer_norm = nn.LayerNorm(size)
                data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=size)
                output = layer_norm(data)
                
                self.assertEqual(output.shape, data.shape)
                
    def test_layer_norm_gradient_flow(self):
        """Test layer normalization gradient flow"""
        layer_norm = nn.LayerNorm(256)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        data.requires_grad_(True)
        
        output = layer_norm(data)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(data.grad)
        self.assertTrue(TestAssertions.assert_gradient_flow([data.grad]))
        
    def test_layer_norm_performance(self):
        """Test layer normalization performance"""
        layer_norm = nn.LayerNorm(512)
        data = self.test_data.create_mlp_data(batch_size=4, seq_len=256, d_model=512)
        
        # Profile performance
        profiler = PerformanceProfiler()
        profiler.start_profile("layer_norm")
        output = layer_norm(data)
        metrics = profiler.end_profile()
        
        self.assertLess(metrics['execution_time'], 0.1)
        self.assertLess(metrics['memory_used'], 50)

class TestPositionalEncoding(unittest.TestCase):
    """Test suite for positional encoding"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_sinusoidal_positional_encoding(self):
        """Test sinusoidal positional encoding"""
        class SinusoidalPositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   (-torch.log(torch.tensor(10000.0)) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe.unsqueeze(0))
                
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        
        pos_encoding = SinusoidalPositionalEncoding(512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        output = pos_encoding(data)
        
        self.assertEqual(output.shape, data.shape)
        
    def test_learned_positional_encoding(self):
        """Test learned positional encoding"""
        class LearnedPositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                self.embedding = nn.Embedding(max_len, d_model)
                
            def forward(self, x):
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
                pos_embeddings = self.embedding(positions)
                return x + pos_embeddings
        
        pos_encoding = LearnedPositionalEncoding(512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        output = pos_encoding(data)
        
        self.assertEqual(output.shape, data.shape)
        
    def test_positional_encoding_different_lengths(self):
        """Test positional encoding with different sequence lengths"""
        class SinusoidalPositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   (-torch.log(torch.tensor(10000.0)) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe.unsqueeze(0))
                
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        
        pos_encoding = SinusoidalPositionalEncoding(512)
        seq_lengths = [64, 128, 256, 512]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                data = self.test_data.create_mlp_data(batch_size=2, seq_len=seq_len, d_model=512)
                output = pos_encoding(data)
                
                self.assertEqual(output.shape, data.shape)

class TestTransformerOptimizations(unittest.TestCase):
    """Test suite for transformer optimizations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_attention_optimization(self):
        """Test attention optimization"""
        class OptimizedAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                
                self.q_linear = nn.Linear(d_model, d_model, bias=False)
                self.k_linear = nn.Linear(d_model, d_model, bias=False)
                self.v_linear = nn.Linear(d_model, d_model, bias=False)
                self.out_linear = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value, mask=None):
                batch_size, seq_len, d_model = query.shape
                
                # Linear transformations
                q = self.q_linear(query)
                k = self.k_linear(key)
                v = self.v_linear(value)
                
                # Reshape for multi-head attention
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                # Scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                attention_weights = torch.softmax(scores, dim=-1)
                attention_output = torch.matmul(attention_weights, v)
                
                # Reshape and output
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, d_model)
                
                return self.out_linear(attention_output), attention_weights
        
        attention = OptimizedAttention(512, 8)
        data = self.test_data.create_attention_data()
        
        # Profile attention
        self.profiler.start_profile("optimized_attention")
        output, weights = attention(data['query'], data['key'], data['value'])
        metrics = self.profiler.end_profile()
        
        self.assertEqual(output.shape, data['query'].shape)
        self.assertLess(metrics['execution_time'], 1.0)
        
    def test_mlp_optimization(self):
        """Test MLP optimization"""
        class OptimizedMLP(nn.Module):
            def __init__(self, d_model, d_ff, dropout=0.1):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        mlp = OptimizedMLP(512, 2048)
        data = self.test_data.create_mlp_data()
        
        # Profile MLP
        self.profiler.start_profile("optimized_mlp")
        output = mlp(data)
        metrics = self.profiler.end_profile()
        
        self.assertEqual(output.shape, data.shape)
        self.assertLess(metrics['execution_time'], 0.5)
        
    def test_transformer_block_optimization(self):
        """Test complete transformer block optimization"""
        class OptimizedTransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                self.attention = OptimizedAttention(d_model, n_heads)
                self.mlp = OptimizedMLP(d_model, d_ff, dropout)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x, mask=None):
                # Self-attention with residual connection
                attn_out, weights = self.attention(x, x, x, mask)
                x = x + self.dropout(attn_out)
                x = self.norm1(x)
                
                # MLP with residual connection
                mlp_out = self.mlp(x)
                x = x + self.dropout(mlp_out)
                x = self.norm2(x)
                
                return x, weights
        
        class OptimizedAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                
                self.q_linear = nn.Linear(d_model, d_model, bias=False)
                self.k_linear = nn.Linear(d_model, d_model, bias=False)
                self.v_linear = nn.Linear(d_model, d_model, bias=False)
                self.out_linear = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value, mask=None):
                batch_size, seq_len, d_model = query.shape
                
                q = self.q_linear(query)
                k = self.k_linear(key)
                v = self.v_linear(value)
                
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                attention_weights = torch.softmax(scores, dim=-1)
                attention_output = torch.matmul(attention_weights, v)
                
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, d_model)
                
                return self.out_linear(attention_output), attention_weights
        
        class OptimizedMLP(nn.Module):
            def __init__(self, d_model, d_ff, dropout=0.1):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        block = OptimizedTransformerBlock(512, 8, 2048)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        # Profile complete block
        self.profiler.start_profile("optimized_transformer_block")
        output, weights = block(data)
        metrics = self.profiler.end_profile()
        
        self.assertEqual(output.shape, data.shape)
        self.assertLess(metrics['execution_time'], 2.0)

if __name__ == '__main__':
    unittest.main()





