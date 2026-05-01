#!/usr/bin/env python3
"""
Distance-Based Attention Mechanism - Example Usage
=================================================

This script demonstrates how to use the distance-based attention mechanism
in various scenarios, from basic usage to advanced TruthGPT integration.

Run this script to see the distance-based attention mechanism in action.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Any

# Import our modules
from distance_based_attention import (
    DistanceBasedAttention,
    MultiHeadDistanceAttention,
    DistanceAttentionConfig,
    DistanceType,
    create_distance_attention
)
from truthgpt_distance_attention_integration import (
    TruthGPTDistanceAttentionModel,
    TruthGPTAttentionConfig,
    DistanceAttentionOptimizer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_basic_usage():
    """Example 1: Basic distance-based attention usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Distance-Based Attention")
    print("="*60)
    
    # Create a simple distance-based attention mechanism
    attention = DistanceBasedAttention(
        hidden_size=256,
        num_heads=8,
        distance_type=DistanceType.L1,
        lambda_param=1.0
    )
    
    # Create test input
    batch_size, seq_len, hidden_size = 2, 32, 256
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output, attn_weights = attention(x, output_attentions=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {attn_weights.sum(dim=-1).mean():.4f}")
    print(f"Attention weights range: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")

def example_2_different_distance_metrics():
    """Example 2: Comparing different distance metrics."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Different Distance Metrics Comparison")
    print("="*60)
    
    # Test data
    batch_size, seq_len, hidden_size = 2, 16, 128
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test different distance metrics
    distance_types = [DistanceType.L1, DistanceType.L2, DistanceType.LP, DistanceType.COSINE]
    
    results = {}
    
    for dist_type in distance_types:
        print(f"\nTesting {dist_type.value} distance...")
        
        config = DistanceAttentionConfig(
            hidden_size=hidden_size,
            num_heads=4,
            distance_type=dist_type.value,
            lambda_param=1.0
        )
        
        attention = create_distance_attention(config)
        
        # Forward pass
        start_time = time.time()
        output, attn_weights = attention(x, output_attentions=True)
        forward_time = time.time() - start_time
        
        # Calculate statistics
        attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
        attn_sparsity = (attn_weights < 0.01).float().mean()
        
        results[dist_type.value] = {
            'forward_time': forward_time,
            'attention_entropy': attn_entropy.item(),
            'attention_sparsity': attn_sparsity.item(),
            'output_norm': output.norm().item()
        }
        
        print(f"  Forward time: {forward_time:.4f}s")
        print(f"  Attention entropy: {attn_entropy:.4f}")
        print(f"  Attention sparsity: {attn_sparsity:.4f}")
        print(f"  Output norm: {output.norm():.4f}")
    
    # Summary
    print(f"\nSummary:")
    for dist_type, metrics in results.items():
        print(f"  {dist_type}: {metrics['forward_time']:.4f}s, entropy={metrics['attention_entropy']:.4f}")

def example_3_multi_head_with_residuals():
    """Example 3: Multi-head attention with residual connections."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-Head Attention with Residual Connections")
    print("="*60)
    
    # Create multi-head attention with residuals
    attention = MultiHeadDistanceAttention(
        hidden_size=512,
        num_heads=8,
        distance_type=DistanceType.L2,
        lambda_param=1.0,
        use_residual=True,
        use_layer_norm=True,
        dropout=0.1
    )
    
    # Test data
    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, 512)
    
    print(f"Input shape: {x.shape}")
    print(f"Input norm: {x.norm():.4f}")
    
    # Forward pass
    output, attn_weights = attention(x, output_attentions=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {output.norm():.4f}")
    print(f"Residual connection preserved: {torch.allclose(output, x, atol=1e-1)}")

def example_4_learnable_lambda():
    """Example 4: Learnable lambda parameter."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Learnable Lambda Parameter")
    print("="*60)
    
    # Create attention with learnable lambda
    attention = DistanceBasedAttention(
        hidden_size=256,
        num_heads=4,
        distance_type=DistanceType.L1,
        lambda_param=1.0
    )
    
    # Enable learnable lambda
    attention.set_learnable_lambda(True)
    
    print(f"Initial lambda: {attention.learnable_lambda.item():.4f}")
    
    # Create test data
    x = torch.randn(2, 16, 256, requires_grad=True)
    
    # Forward pass
    output, _ = attention(x, output_attentions=False)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Lambda gradient: {attention.learnable_lambda.grad.item():.4f}")
    print(f"Lambda after gradient step: {attention.learnable_lambda.item():.4f}")

def example_5_truthgpt_integration():
    """Example 5: TruthGPT integration."""
    print("\n" + "="*60)
    print("EXAMPLE 5: TruthGPT Integration")
    print("="*60)
    
    # Create TruthGPT configuration
    config = TruthGPTAttentionConfig(
        hidden_size=256,
        num_heads=4,
        num_layers=3,
        vocab_size=1000,
        distance_type="l1",
        lambda_param=1.0,
        enable_performance_monitoring=True
    )
    
    # Create model
    model = TruthGPTDistanceAttentionModel(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test data
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of attention layers: {len(outputs['attentions'])}")
    print(f"Number of hidden states: {len(outputs['hidden_states'])}")
    
    # Test gradient flow
    loss = outputs['logits'].sum()
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    # Performance statistics
    if config.enable_performance_monitoring:
        perf_stats = model.get_performance_stats()
        print(f"Performance statistics available for {len(perf_stats)} layers")

def example_6_attention_analysis():
    """Example 6: Attention pattern analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Attention Pattern Analysis")
    print("="*60)
    
    # Create attention mechanism
    attention = DistanceBasedAttention(
        hidden_size=128,
        num_heads=4,
        distance_type=DistanceType.L1,
        lambda_param=1.0
    )
    
    # Create test data with patterns
    batch_size, seq_len = 1, 16
    x = torch.randn(batch_size, seq_len, 128)
    
    # Add some structure to the input
    x[0, :8, :] += 1.0  # First half has different pattern
    x[0, 8:, :] -= 1.0  # Second half has different pattern
    
    # Forward pass
    output, attn_weights = attention(x, output_attentions=True)
    
    # Analyze attention patterns
    attn_weights = attn_weights[0]  # Remove batch dimension
    
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum per head: {attn_weights.sum(dim=-1).mean(dim=0)}")
    
    # Calculate attention entropy per head
    entropy_per_head = []
    for head in range(attn_weights.shape[0]):
        head_weights = attn_weights[head]
        entropy = -(head_weights * torch.log(head_weights + 1e-8)).sum(dim=-1).mean()
        entropy_per_head.append(entropy.item())
    
    print(f"Attention entropy per head: {entropy_per_head}")
    
    # Find most attended positions
    avg_attention = attn_weights.mean(dim=0)  # Average across heads
    most_attended = avg_attention.sum(dim=0).argsort(descending=True)[:5]
    print(f"Most attended positions: {most_attended.tolist()}")

def example_7_performance_comparison():
    """Example 7: Performance comparison with standard attention."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Performance Comparison")
    print("="*60)
    
    # Test parameters
    batch_size, seq_len, hidden_size = 4, 64, 256
    num_heads = 8
    num_runs = 10
    
    # Test data
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Standard multi-head attention
    standard_attention = nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=num_heads,
        dropout=0.1,
        batch_first=True
    )
    
    # Distance-based attention
    distance_attention = DistanceBasedAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        distance_type=DistanceType.L1,
        lambda_param=1.0
    )
    
    # Benchmark standard attention
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = standard_attention(x, x, x)[0]
    standard_time = (time.time() - start_time) / num_runs
    
    # Benchmark distance-based attention
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = distance_attention(x, output_attentions=False)
    distance_time = (time.time() - start_time) / num_runs
    
    print(f"Standard attention time: {standard_time:.4f}s")
    print(f"Distance-based attention time: {distance_time:.4f}s")
    print(f"Speedup ratio: {standard_time / distance_time:.2f}x")
    
    # Memory usage comparison (approximate)
    print(f"\nMemory usage comparison:")
    print(f"Standard attention: ~{batch_size * seq_len * seq_len * num_heads * 4 / 1024**2:.2f} MB")
    print(f"Distance-based attention: ~{batch_size * seq_len * seq_len * num_heads * 4 / 1024**2:.2f} MB")

def main():
    """Run all examples."""
    print("Distance-Based Attention Mechanism - Examples")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_different_distance_metrics()
        example_3_multi_head_with_residuals()
        example_4_learnable_lambda()
        example_5_truthgpt_integration()
        example_6_attention_analysis()
        example_7_performance_comparison()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    main()






