"""
Test suite for Optimization Core Components
Comprehensive tests for all optimization core modules and components
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock, call
import tempfile
import os
from pathlib import Path

# Import optimization core components
from __init__ import (
    # CUDA Kernels
    OptimizedLayerNorm, OptimizedRMSNorm, CUDAOptimizations,
    
    # Triton Optimizations
    TritonLayerNorm, TritonOptimizations,
    
    # Enhanced GRPO
    EnhancedGRPOTrainer, EnhancedGRPOArgs, KalmanFilter,
    
    # MCTS Optimization
    MCTSOptimizer, MCTSOptimizationArgs, create_mcts_optimizer,
    
    # Parallel Training
    EnhancedPPOActor, ParallelTrainingConfig, create_parallel_actor,
    
    # Experience Buffer
    ReplayBuffer, Experience, PrioritizedExperienceReplay, create_experience_buffer,
    
    # Advanced Losses
    GRPOLoss, EnhancedGRPOLoss, AdversarialLoss, CurriculumLoss, create_loss_function,
    
    # Reward Functions
    GRPORewardFunction, AdaptiveRewardFunction, MultiObjectiveRewardFunction, create_reward_function,
    
    # Advanced Normalization
    AdvancedRMSNorm, LlamaRMSNorm, CRMSNorm, AdvancedNormalizationOptimizations,
    create_advanced_rms_norm, create_llama_rms_norm, create_crms_norm,
    
    # Positional Encodings
    RotaryEmbedding, LlamaRotaryEmbedding, FixedLlamaRotaryEmbedding, AliBi,
    SinusoidalPositionalEmbedding, PositionalEncodingOptimizations,
    create_rotary_embedding, create_llama_rotary_embedding, create_alibi, create_sinusoidal_embedding,
    
    # Enhanced MLP
    SwiGLU, GatedMLP, MixtureOfExperts, AdaptiveMLP, EnhancedMLPOptimizations,
    create_swiglu, create_gated_mlp, create_mixture_of_experts, create_adaptive_mlp,
    
    # Advanced Kernel Fusion
    FusedLayerNormLinear, FusedAttentionMLP, KernelFusionOptimizer, create_kernel_fusion_optimizer,
    
    # Advanced Quantization
    QuantizedLinear, QuantizedLayerNorm, AdvancedQuantizationOptimizer, create_quantization_optimizer,
    
    # Memory Pooling
    TensorPool, ActivationCache, MemoryPoolingOptimizer, create_memory_pooling_optimizer,
    get_global_tensor_pool, get_global_activation_cache,
    
    # Enhanced CUDA Kernels
    AdvancedCUDAConfig, FusedKernelOptimizer, MemoryCoalescingOptimizer,
    QuantizationKernelOptimizer, EnhancedCUDAOptimizations, create_enhanced_cuda_optimizer,
    
    # Optimization Registry
    OptimizationRegistry, apply_optimizations, get_optimization_config,
    register_optimization, get_optimization_report,
    
    # Advanced Optimization Registry
    AdvancedOptimizationConfig, get_advanced_optimization_config,
    apply_advanced_optimizations, get_advanced_optimization_report,
    
    # Memory Optimizations
    MemoryOptimizer, MemoryOptimizationConfig, create_memory_optimizer,
    
    # Computational Optimizations
    FusedAttention, BatchOptimizer, ComputationalOptimizer, create_computational_optimizer,
    
    # Optimization Profiles
    OptimizationProfile, get_optimization_profiles, apply_optimization_profile
)


class TestCUDAOptimizations(unittest.TestCase):
    """Test cases for CUDA optimization components."""
    
    def test_optimized_layer_norm(self):
        """Test OptimizedLayerNorm functionality."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Test basic functionality
        layer_norm = OptimizedLayerNorm(128)
        self.assertIsInstance(layer_norm, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128, device='cuda')
        output = layer_norm(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.device, x.device)
    
    def test_optimized_rms_norm(self):
        """Test OptimizedRMSNorm functionality."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Test basic functionality
        rms_norm = OptimizedRMSNorm(128)
        self.assertIsInstance(rms_norm, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128, device='cuda')
        output = rms_norm(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.device, x.device)
    
    def test_cuda_optimizations(self):
        """Test CUDAOptimizations class."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        cuda_ops = CUDAOptimizations()
        self.assertIsInstance(cuda_ops, CUDAOptimizations)
        
        # Test optimization methods
        model = nn.Linear(128, 64)
        optimized_model = cuda_ops.optimize_model(model)
        self.assertIsInstance(optimized_model, nn.Module)


class TestTritonOptimizations(unittest.TestCase):
    """Test cases for Triton optimization components."""
    
    def test_triton_layer_norm(self):
        """Test TritonLayerNorm functionality."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        try:
            layer_norm = TritonLayerNorm(128)
            self.assertIsInstance(layer_norm, nn.Module)
            
            # Test forward pass
            x = torch.randn(32, 128, device='cuda')
            output = layer_norm(x)
            self.assertEqual(output.shape, x.shape)
        except ImportError:
            self.skipTest("Triton not available")
    
    def test_triton_optimizations(self):
        """Test TritonOptimizations class."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        try:
            triton_ops = TritonOptimizations()
            self.assertIsInstance(triton_ops, TritonOptimizations)
        except ImportError:
            self.skipTest("Triton not available")


class TestEnhancedGRPO(unittest.TestCase):
    """Test cases for Enhanced GRPO components."""
    
    def test_enhanced_grpo_trainer(self):
        """Test EnhancedGRPOTrainer functionality."""
        # Test trainer initialization
        args = EnhancedGRPOArgs()
        trainer = EnhancedGRPOTrainer(args)
        self.assertIsInstance(trainer, EnhancedGRPOTrainer)
    
    def test_enhanced_grpo_args(self):
        """Test EnhancedGRPOArgs dataclass."""
        args = EnhancedGRPOArgs()
        self.assertIsInstance(args, EnhancedGRPOArgs)
    
    def test_kalman_filter(self):
        """Test KalmanFilter functionality."""
        kalman = KalmanFilter()
        self.assertIsInstance(kalman, KalmanFilter)
        
        # Test filter operations
        state = np.array([1.0, 0.0])
        measurement = np.array([1.1, 0.1])
        
        filtered_state = kalman.update(state, measurement)
        self.assertIsInstance(filtered_state, np.ndarray)


class TestMCTSOptimization(unittest.TestCase):
    """Test cases for MCTS optimization components."""
    
    def test_mcts_optimizer(self):
        """Test MCTSOptimizer functionality."""
        args = MCTSOptimizationArgs()
        optimizer = MCTSOptimizer(args)
        self.assertIsInstance(optimizer, MCTSOptimizer)
    
    def test_mcts_optimization_args(self):
        """Test MCTSOptimizationArgs dataclass."""
        args = MCTSOptimizationArgs()
        self.assertIsInstance(args, MCTSOptimizationArgs)
    
    def test_create_mcts_optimizer(self):
        """Test create_mcts_optimizer factory function."""
        optimizer = create_mcts_optimizer()
        self.assertIsInstance(optimizer, MCTSOptimizer)


class TestParallelTraining(unittest.TestCase):
    """Test cases for parallel training components."""
    
    def test_enhanced_ppo_actor(self):
        """Test EnhancedPPOActor functionality."""
        config = ParallelTrainingConfig()
        actor = EnhancedPPOActor(config)
        self.assertIsInstance(actor, EnhancedPPOActor)
    
    def test_parallel_training_config(self):
        """Test ParallelTrainingConfig dataclass."""
        config = ParallelTrainingConfig()
        self.assertIsInstance(config, ParallelTrainingConfig)
    
    def test_create_parallel_actor(self):
        """Test create_parallel_actor factory function."""
        actor = create_parallel_actor()
        self.assertIsInstance(actor, EnhancedPPOActor)


class TestExperienceBuffer(unittest.TestCase):
    """Test cases for experience buffer components."""
    
    def test_replay_buffer(self):
        """Test ReplayBuffer functionality."""
        buffer = ReplayBuffer(capacity=1000)
        self.assertIsInstance(buffer, ReplayBuffer)
        
        # Test adding experiences
        experience = Experience(
            state=torch.randn(10),
            action=torch.tensor(1),
            reward=1.0,
            next_state=torch.randn(10),
            done=False
        )
        buffer.add(experience)
        self.assertEqual(len(buffer), 1)
    
    def test_experience_dataclass(self):
        """Test Experience dataclass."""
        experience = Experience(
            state=torch.randn(10),
            action=torch.tensor(1),
            reward=1.0,
            next_state=torch.randn(10),
            done=False
        )
        self.assertIsInstance(experience, Experience)
        self.assertEqual(experience.reward, 1.0)
        self.assertFalse(experience.done)
    
    def test_prioritized_experience_replay(self):
        """Test PrioritizedExperienceReplay functionality."""
        buffer = PrioritizedExperienceReplay(capacity=1000)
        self.assertIsInstance(buffer, PrioritizedExperienceReplay)
    
    def test_create_experience_buffer(self):
        """Test create_experience_buffer factory function."""
        buffer = create_experience_buffer()
        self.assertIsInstance(buffer, ReplayBuffer)


class TestAdvancedLosses(unittest.TestCase):
    """Test cases for advanced loss functions."""
    
    def test_grpo_loss(self):
        """Test GRPOLoss functionality."""
        loss_fn = GRPOLoss()
        self.assertIsInstance(loss_fn, GRPOLoss)
        
        # Test loss calculation
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        loss = loss_fn(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_enhanced_grpo_loss(self):
        """Test EnhancedGRPOLoss functionality."""
        loss_fn = EnhancedGRPOLoss()
        self.assertIsInstance(loss_fn, EnhancedGRPOLoss)
    
    def test_adversarial_loss(self):
        """Test AdversarialLoss functionality."""
        loss_fn = AdversarialLoss()
        self.assertIsInstance(loss_fn, AdversarialLoss)
    
    def test_curriculum_loss(self):
        """Test CurriculumLoss functionality."""
        loss_fn = CurriculumLoss()
        self.assertIsInstance(loss_fn, CurriculumLoss)
    
    def test_create_loss_function(self):
        """Test create_loss_function factory function."""
        loss_fn = create_loss_function("grpo")
        self.assertIsInstance(loss_fn, GRPOLoss)


class TestRewardFunctions(unittest.TestCase):
    """Test cases for reward functions."""
    
    def test_grpo_reward_function(self):
        """Test GRPORewardFunction functionality."""
        reward_fn = GRPORewardFunction()
        self.assertIsInstance(reward_fn, GRPORewardFunction)
        
        # Test reward calculation
        state = torch.randn(10)
        action = torch.tensor(1)
        reward = reward_fn(state, action)
        self.assertIsInstance(reward, torch.Tensor)
    
    def test_adaptive_reward_function(self):
        """Test AdaptiveRewardFunction functionality."""
        reward_fn = AdaptiveRewardFunction()
        self.assertIsInstance(reward_fn, AdaptiveRewardFunction)
    
    def test_multi_objective_reward_function(self):
        """Test MultiObjectiveRewardFunction functionality."""
        reward_fn = MultiObjectiveRewardFunction()
        self.assertIsInstance(reward_fn, MultiObjectiveRewardFunction)
    
    def test_create_reward_function(self):
        """Test create_reward_function factory function."""
        reward_fn = create_reward_function("grpo")
        self.assertIsInstance(reward_fn, GRPORewardFunction)


class TestAdvancedNormalization(unittest.TestCase):
    """Test cases for advanced normalization components."""
    
    def test_advanced_rms_norm(self):
        """Test AdvancedRMSNorm functionality."""
        norm = AdvancedRMSNorm(128)
        self.assertIsInstance(norm, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = norm(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_llama_rms_norm(self):
        """Test LlamaRMSNorm functionality."""
        norm = LlamaRMSNorm(128)
        self.assertIsInstance(norm, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = norm(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_crms_norm(self):
        """Test CRMSNorm functionality."""
        norm = CRMSNorm(128)
        self.assertIsInstance(norm, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = norm(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_advanced_normalization_optimizations(self):
        """Test AdvancedNormalizationOptimizations class."""
        optimizations = AdvancedNormalizationOptimizations()
        self.assertIsInstance(optimizations, AdvancedNormalizationOptimizations)
    
    def test_create_advanced_rms_norm(self):
        """Test create_advanced_rms_norm factory function."""
        norm = create_advanced_rms_norm(128)
        self.assertIsInstance(norm, AdvancedRMSNorm)
    
    def test_create_llama_rms_norm(self):
        """Test create_llama_rms_norm factory function."""
        norm = create_llama_rms_norm(128)
        self.assertIsInstance(norm, LlamaRMSNorm)
    
    def test_create_crms_norm(self):
        """Test create_crms_norm factory function."""
        norm = create_crms_norm(128)
        self.assertIsInstance(norm, CRMSNorm)


class TestPositionalEncodings(unittest.TestCase):
    """Test cases for positional encoding components."""
    
    def test_rotary_embedding(self):
        """Test RotaryEmbedding functionality."""
        embedding = RotaryEmbedding(128)
        self.assertIsInstance(embedding, RotaryEmbedding)
        
        # Test forward pass
        x = torch.randn(32, 10, 128)
        output = embedding(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_llama_rotary_embedding(self):
        """Test LlamaRotaryEmbedding functionality."""
        embedding = LlamaRotaryEmbedding(128)
        self.assertIsInstance(embedding, LlamaRotaryEmbedding)
        
        # Test forward pass
        x = torch.randn(32, 10, 128)
        output = embedding(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_fixed_llama_rotary_embedding(self):
        """Test FixedLlamaRotaryEmbedding functionality."""
        embedding = FixedLlamaRotaryEmbedding(128)
        self.assertIsInstance(embedding, FixedLlamaRotaryEmbedding)
        
        # Test forward pass
        x = torch.randn(32, 10, 128)
        output = embedding(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_alibi(self):
        """Test AliBi functionality."""
        alibi = AliBi(128)
        self.assertIsInstance(alibi, AliBi)
        
        # Test forward pass
        x = torch.randn(32, 10, 128)
        output = alibi(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_sinusoidal_positional_embedding(self):
        """Test SinusoidalPositionalEmbedding functionality."""
        embedding = SinusoidalPositionalEmbedding(128)
        self.assertIsInstance(embedding, SinusoidalPositionalEmbedding)
        
        # Test forward pass
        x = torch.randn(32, 10, 128)
        output = embedding(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_positional_encoding_optimizations(self):
        """Test PositionalEncodingOptimizations class."""
        optimizations = PositionalEncodingOptimizations()
        self.assertIsInstance(optimizations, PositionalEncodingOptimizations)
    
    def test_create_rotary_embedding(self):
        """Test create_rotary_embedding factory function."""
        embedding = create_rotary_embedding(128)
        self.assertIsInstance(embedding, RotaryEmbedding)
    
    def test_create_llama_rotary_embedding(self):
        """Test create_llama_rotary_embedding factory function."""
        embedding = create_llama_rotary_embedding(128)
        self.assertIsInstance(embedding, LlamaRotaryEmbedding)
    
    def test_create_alibi(self):
        """Test create_alibi factory function."""
        alibi = create_alibi(128)
        self.assertIsInstance(alibi, AliBi)
    
    def test_create_sinusoidal_embedding(self):
        """Test create_sinusoidal_embedding factory function."""
        embedding = create_sinusoidal_embedding(128)
        self.assertIsInstance(embedding, SinusoidalPositionalEmbedding)


class TestEnhancedMLP(unittest.TestCase):
    """Test cases for enhanced MLP components."""
    
    def test_swiglu(self):
        """Test SwiGLU functionality."""
        mlp = SwiGLU(128, 256)
        self.assertIsInstance(mlp, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_gated_mlp(self):
        """Test GatedMLP functionality."""
        mlp = GatedMLP(128, 256)
        self.assertIsInstance(mlp, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_mixture_of_experts(self):
        """Test MixtureOfExperts functionality."""
        moe = MixtureOfExperts(128, 256, num_experts=4)
        self.assertIsInstance(moe, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = moe(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_adaptive_mlp(self):
        """Test AdaptiveMLP functionality."""
        mlp = AdaptiveMLP(128, 256)
        self.assertIsInstance(mlp, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_enhanced_mlp_optimizations(self):
        """Test EnhancedMLPOptimizations class."""
        optimizations = EnhancedMLPOptimizations()
        self.assertIsInstance(optimizations, EnhancedMLPOptimizations)
    
    def test_create_swiglu(self):
        """Test create_swiglu factory function."""
        mlp = create_swiglu(128, 256)
        self.assertIsInstance(mlp, SwiGLU)
    
    def test_create_gated_mlp(self):
        """Test create_gated_mlp factory function."""
        mlp = create_gated_mlp(128, 256)
        self.assertIsInstance(mlp, GatedMLP)
    
    def test_create_mixture_of_experts(self):
        """Test create_mixture_of_experts factory function."""
        moe = create_mixture_of_experts(128, 256, num_experts=4)
        self.assertIsInstance(moe, MixtureOfExperts)
    
    def test_create_adaptive_mlp(self):
        """Test create_adaptive_mlp factory function."""
        mlp = create_adaptive_mlp(128, 256)
        self.assertIsInstance(mlp, AdaptiveMLP)


class TestAdvancedKernelFusion(unittest.TestCase):
    """Test cases for advanced kernel fusion components."""
    
    def test_fused_layer_norm_linear(self):
        """Test FusedLayerNormLinear functionality."""
        fused_layer = FusedLayerNormLinear(128, 64)
        self.assertIsInstance(fused_layer, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = fused_layer(x)
        self.assertEqual(output.shape, (32, 64))
    
    def test_fused_attention_mlp(self):
        """Test FusedAttentionMLP functionality."""
        fused_mlp = FusedAttentionMLP(128, 256)
        self.assertIsInstance(fused_mlp, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = fused_mlp(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_kernel_fusion_optimizer(self):
        """Test KernelFusionOptimizer functionality."""
        optimizer = KernelFusionOptimizer()
        self.assertIsInstance(optimizer, KernelFusionOptimizer)
    
    def test_create_kernel_fusion_optimizer(self):
        """Test create_kernel_fusion_optimizer factory function."""
        optimizer = create_kernel_fusion_optimizer()
        self.assertIsInstance(optimizer, KernelFusionOptimizer)


class TestAdvancedQuantization(unittest.TestCase):
    """Test cases for advanced quantization components."""
    
    def test_quantized_linear(self):
        """Test QuantizedLinear functionality."""
        linear = QuantizedLinear(128, 64)
        self.assertIsInstance(linear, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = linear(x)
        self.assertEqual(output.shape, (32, 64))
    
    def test_quantized_layer_norm(self):
        """Test QuantizedLayerNorm functionality."""
        norm = QuantizedLayerNorm(128)
        self.assertIsInstance(norm, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = norm(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_advanced_quantization_optimizer(self):
        """Test AdvancedQuantizationOptimizer functionality."""
        optimizer = AdvancedQuantizationOptimizer()
        self.assertIsInstance(optimizer, AdvancedQuantizationOptimizer)
    
    def test_create_quantization_optimizer(self):
        """Test create_quantization_optimizer factory function."""
        optimizer = create_quantization_optimizer()
        self.assertIsInstance(optimizer, AdvancedQuantizationOptimizer)


class TestMemoryPooling(unittest.TestCase):
    """Test cases for memory pooling components."""
    
    def test_tensor_pool(self):
        """Test TensorPool functionality."""
        pool = TensorPool()
        self.assertIsInstance(pool, TensorPool)
        
        # Test tensor operations
        tensor = torch.randn(32, 128)
        pool.put(tensor)
        retrieved = pool.get((32, 128))
        self.assertEqual(retrieved.shape, tensor.shape)
    
    def test_activation_cache(self):
        """Test ActivationCache functionality."""
        cache = ActivationCache()
        self.assertIsInstance(cache, ActivationCache)
        
        # Test caching operations
        key = "test_key"
        value = torch.randn(32, 128)
        cache.put(key, value)
        retrieved = cache.get(key)
        self.assertTrue(torch.equal(retrieved, value))
    
    def test_memory_pooling_optimizer(self):
        """Test MemoryPoolingOptimizer functionality."""
        optimizer = MemoryPoolingOptimizer()
        self.assertIsInstance(optimizer, MemoryPoolingOptimizer)
    
    def test_create_memory_pooling_optimizer(self):
        """Test create_memory_pooling_optimizer factory function."""
        optimizer = create_memory_pooling_optimizer()
        self.assertIsInstance(optimizer, MemoryPoolingOptimizer)
    
    def test_get_global_tensor_pool(self):
        """Test get_global_tensor_pool function."""
        pool = get_global_tensor_pool()
        self.assertIsInstance(pool, TensorPool)
    
    def test_get_global_activation_cache(self):
        """Test get_global_activation_cache function."""
        cache = get_global_activation_cache()
        self.assertIsInstance(cache, ActivationCache)


class TestEnhancedCUDAKernels(unittest.TestCase):
    """Test cases for enhanced CUDA kernel components."""
    
    def test_advanced_cuda_config(self):
        """Test AdvancedCUDAConfig functionality."""
        config = AdvancedCUDAConfig()
        self.assertIsInstance(config, AdvancedCUDAConfig)
    
    def test_fused_kernel_optimizer(self):
        """Test FusedKernelOptimizer functionality."""
        optimizer = FusedKernelOptimizer()
        self.assertIsInstance(optimizer, FusedKernelOptimizer)
    
    def test_memory_coalescing_optimizer(self):
        """Test MemoryCoalescingOptimizer functionality."""
        optimizer = MemoryCoalescingOptimizer()
        self.assertIsInstance(optimizer, MemoryCoalescingOptimizer)
    
    def test_quantization_kernel_optimizer(self):
        """Test QuantizationKernelOptimizer functionality."""
        optimizer = QuantizationKernelOptimizer()
        self.assertIsInstance(optimizer, QuantizationKernelOptimizer)
    
    def test_enhanced_cuda_optimizations(self):
        """Test EnhancedCUDAOptimizations functionality."""
        optimizations = EnhancedCUDAOptimizations()
        self.assertIsInstance(optimizations, EnhancedCUDAOptimizations)
    
    def test_create_enhanced_cuda_optimizer(self):
        """Test create_enhanced_cuda_optimizer factory function."""
        optimizer = create_enhanced_cuda_optimizer()
        self.assertIsInstance(optimizer, EnhancedCUDAOptimizations)


class TestOptimizationRegistry(unittest.TestCase):
    """Test cases for optimization registry components."""
    
    def test_optimization_registry(self):
        """Test OptimizationRegistry functionality."""
        registry = OptimizationRegistry()
        self.assertIsInstance(registry, OptimizationRegistry)
    
    def test_apply_optimizations(self):
        """Test apply_optimizations function."""
        model = nn.Linear(128, 64)
        optimized_model = apply_optimizations(model)
        self.assertIsInstance(optimized_model, nn.Module)
    
    def test_get_optimization_config(self):
        """Test get_optimization_config function."""
        config = get_optimization_config()
        self.assertIsInstance(config, dict)
    
    def test_register_optimization(self):
        """Test register_optimization function."""
        def test_optimization(model):
            return model
        
        register_optimization("test_opt", test_optimization)
        # Should not raise exceptions
    
    def test_get_optimization_report(self):
        """Test get_optimization_report function."""
        report = get_optimization_report()
        self.assertIsInstance(report, dict)


class TestAdvancedOptimizationRegistry(unittest.TestCase):
    """Test cases for advanced optimization registry components."""
    
    def test_advanced_optimization_config(self):
        """Test AdvancedOptimizationConfig functionality."""
        config = AdvancedOptimizationConfig()
        self.assertIsInstance(config, AdvancedOptimizationConfig)
    
    def test_get_advanced_optimization_config(self):
        """Test get_advanced_optimization_config function."""
        config = get_advanced_optimization_config()
        self.assertIsInstance(config, AdvancedOptimizationConfig)
    
    def test_apply_advanced_optimizations(self):
        """Test apply_advanced_optimizations function."""
        model = nn.Linear(128, 64)
        optimized_model = apply_advanced_optimizations(model)
        self.assertIsInstance(optimized_model, nn.Module)
    
    def test_get_advanced_optimization_report(self):
        """Test get_advanced_optimization_report function."""
        report = get_advanced_optimization_report()
        self.assertIsInstance(report, dict)


class TestMemoryOptimizations(unittest.TestCase):
    """Test cases for memory optimization components."""
    
    def test_memory_optimizer(self):
        """Test MemoryOptimizer functionality."""
        optimizer = MemoryOptimizer()
        self.assertIsInstance(optimizer, MemoryOptimizer)
    
    def test_memory_optimization_config(self):
        """Test MemoryOptimizationConfig functionality."""
        config = MemoryOptimizationConfig()
        self.assertIsInstance(config, MemoryOptimizationConfig)
    
    def test_create_memory_optimizer(self):
        """Test create_memory_optimizer factory function."""
        optimizer = create_memory_optimizer()
        self.assertIsInstance(optimizer, MemoryOptimizer)


class TestComputationalOptimizations(unittest.TestCase):
    """Test cases for computational optimization components."""
    
    def test_fused_attention(self):
        """Test FusedAttention functionality."""
        attention = FusedAttention(128, num_heads=8)
        self.assertIsInstance(attention, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 10, 128)
        output = attention(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_batch_optimizer(self):
        """Test BatchOptimizer functionality."""
        optimizer = BatchOptimizer()
        self.assertIsInstance(optimizer, BatchOptimizer)
    
    def test_computational_optimizer(self):
        """Test ComputationalOptimizer functionality."""
        optimizer = ComputationalOptimizer()
        self.assertIsInstance(optimizer, ComputationalOptimizer)
    
    def test_create_computational_optimizer(self):
        """Test create_computational_optimizer factory function."""
        optimizer = create_computational_optimizer()
        self.assertIsInstance(optimizer, ComputationalOptimizer)


class TestOptimizationProfiles(unittest.TestCase):
    """Test cases for optimization profile components."""
    
    def test_optimization_profile(self):
        """Test OptimizationProfile functionality."""
        profile = OptimizationProfile()
        self.assertIsInstance(profile, OptimizationProfile)
    
    def test_get_optimization_profiles(self):
        """Test get_optimization_profiles function."""
        profiles = get_optimization_profiles()
        self.assertIsInstance(profiles, dict)
        self.assertGreater(len(profiles), 0)
    
    def test_apply_optimization_profile(self):
        """Test apply_optimization_profile function."""
        model = nn.Linear(128, 64)
        optimized_model = apply_optimization_profile(model, "balanced")
        self.assertIsInstance(optimized_model, nn.Module)


class TestIntegration(unittest.TestCase):
    """Integration tests for optimization core components."""
    
    def test_component_integration(self):
        """Test integration between different components."""
        # Create a complex model
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = AdvancedRMSNorm(128)
                self.linear = QuantizedLinear(128, 64)
                self.attention = FusedAttention(64, num_heads=8)
                self.mlp = SwiGLU(64, 128)
            
            def forward(self, x):
                x = self.norm(x)
                x = self.linear(x)
                x = self.attention(x)
                x = self.mlp(x)
                return x
        
        model = ComplexModel()
        
        # Test forward pass
        x = torch.randn(32, 10, 128)
        output = model(x)
        self.assertEqual(output.shape, (32, 10, 128))
    
    def test_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Create model
        model = nn.Linear(128, 64)
        
        # Apply various optimizations
        optimized_model = apply_optimizations(model)
        self.assertIsInstance(optimized_model, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 128)
        output = optimized_model(x)
        self.assertEqual(output.shape, (32, 64))
    
    def test_memory_optimization_integration(self):
        """Test memory optimization integration."""
        # Create memory optimizer
        memory_optimizer = create_memory_optimizer()
        
        # Create model
        model = nn.Linear(128, 64)
        
        # Apply memory optimizations
        optimized_model = memory_optimizer.optimize_model(model)
        self.assertIsInstance(optimized_model, nn.Module)
    
    def test_quantization_integration(self):
        """Test quantization integration."""
        # Create quantization optimizer
        quant_optimizer = create_quantization_optimizer()
        
        # Create model
        model = nn.Linear(128, 64)
        
        # Apply quantization
        optimized_model = quant_optimizer.optimize_model(model)
        self.assertIsInstance(optimized_model, nn.Module)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCUDAOptimizations,
        TestTritonOptimizations,
        TestEnhancedGRPO,
        TestMCTSOptimization,
        TestParallelTraining,
        TestExperienceBuffer,
        TestAdvancedLosses,
        TestRewardFunctions,
        TestAdvancedNormalization,
        TestPositionalEncodings,
        TestEnhancedMLP,
        TestAdvancedKernelFusion,
        TestAdvancedQuantization,
        TestMemoryPooling,
        TestEnhancedCUDAKernels,
        TestOptimizationRegistry,
        TestAdvancedOptimizationRegistry,
        TestMemoryOptimizations,
        TestComputationalOptimizations,
        TestOptimizationProfiles,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

