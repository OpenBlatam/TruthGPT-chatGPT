"""
TruthGPT Phase 0 Test Suite

This module contains comprehensive tests for the TruthGPT Phase 0 specifications,
including core transformer optimizations, attention mechanisms, and basic performance enhancements.
"""

import pytest
import torch
import numpy as np
from truthgpt_specs.phase0 import (
    create_phase0_optimizer,
    Phase0Config,
    TransformerOptimizer,
    AttentionMechanism,
    FeedForwardNetwork
)


class TestPhase0Core:
    """Test suite for Phase 0 core functionality."""
    
    def test_phase0_config_creation(self):
        """Test Phase 0 configuration creation."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            vocab_size=50000,
            max_sequence_length=4096
        )
        
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 2048
        assert config.vocab_size == 50000
        assert config.max_sequence_length == 4096
    
    def test_phase0_optimizer_creation(self):
        """Test Phase 0 optimizer creation."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_gradient_checkpointing=True,
            use_mixed_precision=True
        )
        
        optimizer = create_phase0_optimizer(config)
        assert optimizer is not None
        assert optimizer.config == config
    
    def test_transformer_optimizer(self):
        """Test transformer optimizer functionality."""
        config = Phase0Config(d_model=512, n_heads=8, n_layers=6)
        optimizer = TransformerOptimizer(config)
        
        # Create dummy model
        model = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        # Test optimization
        optimized_model = optimizer.optimize_model(model)
        assert optimized_model is not None
        assert optimized_model != model  # Should be optimized
    
    def test_attention_mechanism(self):
        """Test attention mechanism functionality."""
        config = Phase0Config(d_model=512, n_heads=8)
        attention = AttentionMechanism(config)
        
        # Test attention computation
        batch_size = 2
        seq_len = 128
        input_tensor = torch.randn(batch_size, seq_len, 512)
        
        output = attention(input_tensor)
        assert output.shape == input_tensor.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_feed_forward_network(self):
        """Test feed-forward network functionality."""
        config = Phase0Config(d_model=512, d_ff=2048)
        ff_network = FeedForwardNetwork(config)
        
        # Test feed-forward computation
        batch_size = 2
        seq_len = 128
        input_tensor = torch.randn(batch_size, seq_len, 512)
        
        output = ff_network(input_tensor)
        assert output.shape == input_tensor.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_gradient_checkpointing=True
        )
        
        optimizer = create_phase0_optimizer(config)
        assert optimizer.config.use_gradient_checkpointing is True
    
    def test_mixed_precision(self):
        """Test mixed precision functionality."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_mixed_precision=True
        )
        
        optimizer = create_phase0_optimizer(config)
        assert optimizer.config.use_mixed_precision is True
    
    def test_flash_attention(self):
        """Test Flash Attention functionality."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_flash_attention=True
        )
        
        optimizer = create_phase0_optimizer(config)
        assert optimizer.config.use_flash_attention is True
    
    def test_dynamic_batching(self):
        """Test dynamic batching functionality."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_dynamic_batching=True
        )
        
        optimizer = create_phase0_optimizer(config)
        assert optimizer.config.use_dynamic_batching is True


class TestPhase0Performance:
    """Test suite for Phase 0 performance metrics."""
    
    def test_memory_usage(self):
        """Test memory usage optimization."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_gradient_checkpointing=True
        )
        
        optimizer = create_phase0_optimizer(config)
        
        # Test memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        model = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        optimized_model = optimizer.optimize_model(model)
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_usage = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_usage < 1024 * 1024 * 1024  # Less than 1GB
    
    def test_training_speed(self):
        """Test training speed optimization."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_mixed_precision=True
        )
        
        optimizer = create_phase0_optimizer(config)
        
        # Test training speed
        model = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        optimized_model = optimizer.optimize_model(model)
        
        # Test forward pass speed
        input_tensor = torch.randn(2, 128, 512)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output = optimized_model(input_tensor, input_tensor)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Inference should be fast
        assert inference_time < 1.0  # Less than 1 second
    
    def test_throughput(self):
        """Test throughput optimization."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_dynamic_batching=True
        )
        
        optimizer = create_phase0_optimizer(config)
        
        # Test throughput
        model = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        optimized_model = optimizer.optimize_model(model)
        
        # Test batch processing
        batch_sizes = [1, 2, 4, 8]
        throughputs = []
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 128, 512)
            
            import time
            start_time = time.time()
            
            with torch.no_grad():
                output = optimized_model(input_tensor, input_tensor)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            throughput = batch_size / inference_time
            throughputs.append(throughput)
        
        # Throughput should increase with batch size
        assert throughputs[-1] > throughputs[0]
    
    def test_accuracy_preservation(self):
        """Test accuracy preservation during optimization."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_mixed_precision=True
        )
        
        optimizer = create_phase0_optimizer(config)
        
        # Test accuracy preservation
        model = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        optimized_model = optimizer.optimize_model(model)
        
        # Test with same input
        input_tensor = torch.randn(2, 128, 512)
        
        with torch.no_grad():
            original_output = model(input_tensor, input_tensor)
            optimized_output = optimized_model(input_tensor, input_tensor)
        
        # Outputs should be similar (within tolerance)
        difference = torch.abs(original_output - optimized_output).mean()
        assert difference < 0.1  # Less than 10% difference


class TestPhase0Integration:
    """Test suite for Phase 0 integration."""
    
    def test_end_to_end_optimization(self):
        """Test end-to-end optimization pipeline."""
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            use_flash_attention=True,
            use_dynamic_batching=True
        )
        
        optimizer = create_phase0_optimizer(config)
        
        # Create and optimize model
        model = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        optimized_model = optimizer.optimize_model(model)
        
        # Test complete pipeline
        input_tensor = torch.randn(2, 128, 512)
        
        with torch.no_grad():
            output = optimized_model(input_tensor, input_tensor)
        
        assert output is not None
        assert output.shape == input_tensor.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            vocab_size=50000,
            max_sequence_length=4096
        )
        
        assert valid_config.is_valid()
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            invalid_config = Phase0Config(
                d_model=0,  # Invalid
                n_heads=8,
                n_layers=6
            )
    
    def test_migration_compatibility(self):
        """Test migration compatibility."""
        # Test that Phase 0 is compatible with future phases
        config = Phase0Config(
            d_model=512,
            n_heads=8,
            n_layers=6
        )
        
        optimizer = create_phase0_optimizer(config)
        
        # Test that optimizer can be migrated
        assert hasattr(optimizer, 'migrate_to_altair')
        assert hasattr(optimizer, 'migrate_to_bellatrix')
        assert hasattr(optimizer, 'migrate_to_capella')


if __name__ == "__main__":
    pytest.main([__file__])





