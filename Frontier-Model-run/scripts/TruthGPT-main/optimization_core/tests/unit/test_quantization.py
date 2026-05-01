"""
Unit tests for quantization components
Tests quantization techniques, quantized layers, and optimization
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestQuantizationBasics(unittest.TestCase):
    """Test suite for basic quantization functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_linear_quantization(self):
        """Test linear quantization"""
        class QuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.num_bits = num_bits
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
                
                # Quantization parameters
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                # Quantize weights
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.scale + self.zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                
                # Apply quantized weights
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        quantized_linear = QuantizedLinear(512, 1024, num_bits=8)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        output = quantized_linear(data)
        
        self.assertEqual(output.shape, (2, 128, 1024))
        
    def test_activation_quantization(self):
        """Test activation quantization"""
        class QuantizedActivation(nn.Module):
            def __init__(self, num_bits=8):
                super().__init__()
                self.num_bits = num_bits
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                qmin = 0
                qmax = 2 ** self.num_bits - 1
                
                # Quantize activation
                x_quantized = torch.clamp(
                    torch.round(x / self.scale + self.zero_point),
                    qmin, qmax
                )
                
                # Dequantize
                x_dequantized = (x_quantized - self.zero_point) * self.scale
                
                return x_dequantized
        
        quantized_activation = QuantizedActivation(num_bits=8)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        output = quantized_activation(data)
        
        self.assertEqual(output.shape, data.shape)
        
    def test_quantization_error(self):
        """Test quantization error"""
        class QuantizationError:
            def __init__(self, num_bits=8):
                self.num_bits = num_bits
                
            def quantize(self, x, scale=1.0, zero_point=0.0):
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                x_quantized = torch.clamp(
                    torch.round(x / scale + zero_point),
                    qmin, qmax
                )
                
                x_dequantized = (x_quantized - zero_point) * scale
                
                return x_dequantized, x_quantized
            
            def compute_error(self, original, quantized):
                mse = torch.mean((original - quantized) ** 2)
                return mse.item()
        
        quantizer = QuantizationError(num_bits=8)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        
        quantized, q_values = quantizer.quantize(data)
        error = quantizer.compute_error(data, quantized)
        
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)
        
    def test_different_quantization_bits(self):
        """Test quantization with different bit widths"""
        class QuantizationError:
            def __init__(self, num_bits):
                self.num_bits = num_bits
                
            def quantize(self, x, scale=1.0, zero_point=0.0):
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                x_quantized = torch.clamp(
                    torch.round(x / scale + zero_point),
                    qmin, qmax
                )
                
                x_dequantized = (x_quantized - zero_point) * scale
                return x_dequantized
            
            def compute_error(self, original, quantized):
                return torch.mean((original - quantized) ** 2).item()
        
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        bit_widths = [4, 6, 8, 16]
        errors = []
        
        for bits in bit_widths:
            quantizer = QuantizationError(bits)
            quantized = quantizer.quantize(data)
            error = quantizer.compute_error(data, quantized)
            errors.append(error)
        
        # Higher bit widths should have lower errors
        for i in range(1, len(errors)):
            self.assertLessEqual(errors[i], errors[i-1])

class TestQuantizedLayers(unittest.TestCase):
    """Test suite for quantized layers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_quantized_linear_layer(self):
        """Test quantized linear layer"""
        class QuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.num_bits = num_bits
                
                # Original weights
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
                
                # Quantization parameters
                self.weight_scale = nn.Parameter(torch.tensor(1.0))
                self.weight_zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                # Quantize weights
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.weight_scale + self.weight_zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.weight_zero_point) * self.weight_scale
                
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        quantized_linear = QuantizedLinear(512, 1024, num_bits=8)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        output = quantized_linear(data)
        
        self.assertEqual(output.shape, (2, 128, 1024))
        
    def test_quantized_conv_layer(self):
        """Test quantized convolutional layer"""
        class QuantizedConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, num_bits=8):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.num_bits = num_bits
                
                # Original weights
                self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
                self.bias = nn.Parameter(torch.randn(out_channels))
                
                # Quantization parameters
                self.weight_scale = nn.Parameter(torch.tensor(1.0))
                self.weight_zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                # Quantize weights
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.weight_scale + self.weight_zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.weight_zero_point) * self.weight_scale
                
                return torch.nn.functional.conv2d(x, weight_dequantized, self.bias)
        
        quantized_conv = QuantizedConv2d(3, 64, 3, num_bits=8)
        data = torch.randn(2, 3, 32, 32)
        
        output = quantized_conv(data)
        
        self.assertEqual(output.shape, (2, 64, 30, 30))
        
    def test_quantized_attention(self):
        """Test quantized attention mechanism"""
        class QuantizedAttention(nn.Module):
            def __init__(self, d_model, n_heads, num_bits=8):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.num_bits = num_bits
                
                self.q_linear = QuantizedLinear(d_model, d_model, num_bits)
                self.k_linear = QuantizedLinear(d_model, d_model, num_bits)
                self.v_linear = QuantizedLinear(d_model, d_model, num_bits)
                self.out_linear = QuantizedLinear(d_model, d_model, num_bits)
                
            def forward(self, query, key, value):
                batch_size, seq_len, d_model = query.shape
                
                q = self.q_linear(query)
                k = self.k_linear(key)
                v = self.v_linear(value)
                
                # Reshape for multi-head attention
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                # Scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attention_weights = torch.softmax(scores, dim=-1)
                attention_output = torch.matmul(attention_weights, v)
                
                # Reshape and output
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, d_model)
                
                return self.out_linear(attention_output), attention_weights
        
        class QuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
                self.num_bits = num_bits
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.scale + self.zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        attention = QuantizedAttention(512, 8, num_bits=8)
        data = self.test_data.create_attention_data()
        
        output, weights = attention(data['query'], data['key'], data['value'])
        
        self.assertEqual(output.shape, data['query'].shape)
        self.assertEqual(weights.shape, (2, 8, 128, 128))

class TestQuantizationOptimization(unittest.TestCase):
    """Test suite for quantization optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_quantization_performance(self):
        """Test quantization performance"""
        class QuantizedModel(nn.Module):
            def __init__(self, d_model=512, num_bits=8):
                super().__init__()
                self.linear1 = QuantizedLinear(d_model, 1024, num_bits)
                self.linear2 = QuantizedLinear(1024, d_model, num_bits)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x
        
        class QuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
                self.num_bits = num_bits
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.scale + self.zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        # Test different quantization levels
        bit_widths = [4, 6, 8, 16]
        results = []
        
        for bits in bit_widths:
            model = QuantizedModel(num_bits=bits)
            data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
            
            # Profile performance
            self.profiler.start_profile(f"quantized_model_{bits}bit")
            output = model(data)
            metrics = self.profiler.end_profile()
            
            results.append({
                'bits': bits,
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used']
            })
            
            self.assertEqual(output.shape, data.shape)
        
        # Analyze performance scaling
        for result in results:
            self.assertLess(result['execution_time'], 1.0)
            self.assertLess(result['memory_used'], 100)
            
    def test_quantization_memory_usage(self):
        """Test quantization memory usage"""
        class MemoryEfficientQuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.num_bits = num_bits
                
                # Store quantized weights directly
                self.weight_quantized = nn.Parameter(
                    torch.randint(-(2**(num_bits-1)), 2**(num_bits-1), (out_features, in_features))
                )
                self.bias = nn.Parameter(torch.randn(out_features))
                
                # Quantization parameters
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                # Dequantize weights on-the-fly
                weight_dequantized = (self.weight_quantized - self.zero_point) * self.scale
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        # Test memory usage with different bit widths
        bit_widths = [4, 6, 8, 16]
        memory_usage = []
        
        for bits in bit_widths:
            linear = MemoryEfficientQuantizedLinear(512, 1024, num_bits=bits)
            
            # Calculate theoretical memory usage
            weight_memory = 512 * 1024 * bits / 8  # bytes
            bias_memory = 1024 * 32 / 8  # bytes (float32)
            total_memory = weight_memory + bias_memory
            
            memory_usage.append({
                'bits': bits,
                'weight_memory': weight_memory,
                'total_memory': total_memory
            })
        
        # Verify memory scaling
        for i in range(1, len(memory_usage)):
            prev = memory_usage[i-1]
            curr = memory_usage[i]
            self.assertGreaterEqual(curr['total_memory'], prev['total_memory'])
            
    def test_quantization_accuracy(self):
        """Test quantization accuracy"""
        class AccuracyTestModel(nn.Module):
            def __init__(self, d_model=256, num_bits=8):
                super().__init__()
                self.linear1 = nn.Linear(d_model, 512)
                self.linear2 = nn.Linear(512, d_model)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x
        
        class QuantizedAccuracyTestModel(nn.Module):
            def __init__(self, d_model=256, num_bits=8):
                super().__init__()
                self.linear1 = QuantizedLinear(d_model, 512, num_bits)
                self.linear2 = QuantizedLinear(512, d_model, num_bits)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x
        
        class QuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
                self.num_bits = num_bits
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.scale + self.zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        # Create models
        original_model = AccuracyTestModel()
        quantized_model = QuantizedAccuracyTestModel(num_bits=8)
        
        # Copy weights from original to quantized
        quantized_model.linear1.weight.data = original_model.linear1.weight.data.clone()
        quantized_model.linear1.bias.data = original_model.linear1.bias.data.clone()
        quantized_model.linear2.weight.data = original_model.linear2.weight.data.clone()
        quantized_model.linear2.bias.data = original_model.linear2.bias.data.clone()
        
        # Test with same input
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        
        with torch.no_grad():
            original_output = original_model(data)
            quantized_output = quantized_model(data)
        
        # Calculate accuracy
        mse = torch.mean((original_output - quantized_output) ** 2)
        relative_error = mse / torch.mean(original_output ** 2)
        
        self.assertIsInstance(mse.item(), float)
        self.assertIsInstance(relative_error.item(), float)
        self.assertGreater(relative_error.item(), 0)

class TestQuantizationIntegration(unittest.TestCase):
    """Test suite for quantization integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_quantized_transformer_block(self):
        """Test quantized transformer block"""
        class QuantizedTransformerBlock(nn.Module):
            def __init__(self, d_model=512, n_heads=8, d_ff=2048, num_bits=8):
                super().__init__()
                self.attention = QuantizedAttention(d_model, n_heads, num_bits)
                self.mlp = QuantizedMLP(d_model, d_ff, num_bits)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(0.1)
                
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
        
        class QuantizedAttention(nn.Module):
            def __init__(self, d_model, n_heads, num_bits=8):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                
                self.q_linear = QuantizedLinear(d_model, d_model, num_bits)
                self.k_linear = QuantizedLinear(d_model, d_model, num_bits)
                self.v_linear = QuantizedLinear(d_model, d_model, num_bits)
                self.out_linear = QuantizedLinear(d_model, d_model, num_bits)
                
            def forward(self, query, key, value):
                batch_size, seq_len, d_model = query.shape
                
                q = self.q_linear(query)
                k = self.k_linear(key)
                v = self.v_linear(value)
                
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attention_weights = torch.softmax(scores, dim=-1)
                attention_output = torch.matmul(attention_weights, v)
                
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, d_model)
                
                return self.out_linear(attention_output), attention_weights
        
        class QuantizedMLP(nn.Module):
            def __init__(self, d_model, d_ff, num_bits=8):
                super().__init__()
                self.linear1 = QuantizedLinear(d_model, d_ff, num_bits)
                self.linear2 = QuantizedLinear(d_ff, d_model, num_bits)
                self.activation = nn.GELU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x
        
        class QuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
                self.num_bits = num_bits
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.scale + self.zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        block = QuantizedTransformerBlock(num_bits=8)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        output = block(data)
        
        self.assertEqual(output.shape, data.shape)
        
    def test_quantization_workflow(self):
        """Test complete quantization workflow"""
        class QuantizationWorkflow:
            def __init__(self, model, num_bits=8):
                self.model = model
                self.num_bits = num_bits
                self.quantized_layers = []
                
            def quantize_layer(self, layer):
                """Quantize a single layer"""
                if isinstance(layer, nn.Linear):
                    return self._quantize_linear(layer)
                return layer
                
            def _quantize_linear(self, linear_layer):
                """Quantize linear layer"""
                class QuantizedLinear(nn.Module):
                    def __init__(self, original_layer, num_bits):
                        super().__init__()
                        self.weight = original_layer.weight
                        self.bias = original_layer.bias
                        self.num_bits = num_bits
                        self.scale = nn.Parameter(torch.tensor(1.0))
                        self.zero_point = nn.Parameter(torch.tensor(0.0))
                        
                    def forward(self, x):
                        qmin = -(2 ** (self.num_bits - 1))
                        qmax = 2 ** (self.num_bits - 1) - 1
                        
                        weight_quantized = torch.clamp(
                            torch.round(self.weight / self.scale + self.zero_point),
                            qmin, qmax
                        )
                        weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                        
                        return torch.nn.functional.linear(x, weight_dequantized, self.bias)
                
                return QuantizedLinear(linear_layer, self.num_bits)
            
            def apply_quantization(self):
                """Apply quantization to model"""
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        quantized = self.quantize_layer(module)
                        setattr(self.model, name, quantized)
                        self.quantized_layers.append(name)
        
        # Create test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(512, 1024)
                self.linear2 = nn.Linear(1024, 512)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        workflow = QuantizationWorkflow(model, num_bits=8)
        workflow.apply_quantization()
        
        # Test quantized model
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        output = model(data)
        
        self.assertEqual(output.shape, data.shape)
        self.assertGreater(len(workflow.quantized_layers), 0)

if __name__ == '__main__':
    unittest.main()





