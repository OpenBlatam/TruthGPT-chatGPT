"""
Optimization Core Module for TruthGPT
Advanced performance optimizations and CUDA/Triton kernels
Enhanced with MCTS, parallel training, and advanced optimization techniques
"""

from .cuda_kernels import OptimizedLayerNorm, OptimizedRMSNorm, CUDAOptimizations
from .triton_optimizations import TritonLayerNorm, TritonOptimizations  
from .enhanced_grpo import EnhancedGRPOTrainer, EnhancedGRPOArgs, KalmanFilter
from .mcts_optimization import MCTSOptimizer, MCTSOptimizationArgs, create_mcts_optimizer
from .parallel_training import EnhancedPPOActor, ParallelTrainingConfig, create_parallel_actor
from .experience_buffer import ReplayBuffer, Experience, PrioritizedExperienceReplay, create_experience_buffer
from .advanced_losses import GRPOLoss, EnhancedGRPOLoss, AdversarialLoss, CurriculumLoss, create_loss_function
from .reward_functions import GRPORewardFunction, AdaptiveRewardFunction, MultiObjectiveRewardFunction, create_reward_function
from .advanced_normalization import AdvancedRMSNorm, LlamaRMSNorm, CRMSNorm, AdvancedNormalizationOptimizations, create_advanced_rms_norm, create_llama_rms_norm, create_crms_norm
from .positional_encodings import RotaryEmbedding, LlamaRotaryEmbedding, FixedLlamaRotaryEmbedding, AliBi, SinusoidalPositionalEmbedding, PositionalEncodingOptimizations, create_rotary_embedding, create_llama_rotary_embedding, create_alibi, create_sinusoidal_embedding
from .enhanced_mlp import SwiGLU, GatedMLP, MixtureOfExperts, AdaptiveMLP, EnhancedMLPOptimizations, create_swiglu, create_gated_mlp, create_mixture_of_experts, create_adaptive_mlp
from .advanced_kernel_fusion import (
    FusedLayerNormLinear, FusedAttentionMLP, KernelFusionOptimizer,
    create_kernel_fusion_optimizer
)
from .advanced_quantization import (
    QuantizedLinear, QuantizedLayerNorm, AdvancedQuantizationOptimizer,
    create_quantization_optimizer
)
from .memory_pooling import (
    TensorPool, ActivationCache, MemoryPoolingOptimizer,
    create_memory_pooling_optimizer, get_global_tensor_pool, get_global_activation_cache
)
from .enhanced_cuda_kernels import (
    AdvancedCUDAConfig, FusedKernelOptimizer, MemoryCoalescingOptimizer,
    QuantizationKernelOptimizer, EnhancedCUDAOptimizations, create_enhanced_cuda_optimizer
)
from .ultra_optimization_core import (
    UltraOptimizedLayerNorm, AdaptiveQuantization, DynamicKernelFusion,
    IntelligentMemoryManager, UltraOptimizationCore, create_ultra_optimization_core
)
from .super_optimization_core import (
    SuperOptimizedAttention, AdaptiveComputationTime, SuperOptimizedMLP,
    ProgressiveOptimization, SuperOptimizationCore, create_super_optimization_core
)
from .meta_optimization_core import (
    SelfOptimizingLayerNorm, AdaptiveOptimizationScheduler, DynamicComputationGraph,
    MetaOptimizationCore, create_meta_optimization_core
)
from .hyper_optimization_core import (
    HyperOptimizedLinear, NeuralArchitectureOptimizer, AdvancedGradientOptimizer,
    HyperOptimizationCore, create_hyper_optimization_core
)
from .quantum_optimization_core import (
    QuantumInspiredLinear, QuantumAttention, QuantumLayerNorm,
    QuantumOptimizationCore, create_quantum_optimization_core
)
from .neural_architecture_search import (
    ArchitectureGene, ArchitectureChromosome, NeuralArchitectureSearchOptimizer,
    NASOptimizationCore, create_nas_optimization_core
)
from .enhanced_optimization_core import (
    AdaptivePrecisionOptimizer, DynamicKernelFusionOptimizer, IntelligentMemoryManager,
    SelfOptimizingComponent, EnhancedOptimizedLayerNorm, EnhancedOptimizationCore,
    create_enhanced_optimization_core
)
from .ultra_enhanced_optimization_core import (
    NeuralCodeOptimizer, AdaptiveAlgorithmSelector, PredictiveOptimizer,
    SelfEvolvingKernel, RealTimeProfiler, UltraEnhancedOptimizationCore,
    create_ultra_enhanced_optimization_core
)
from .mega_enhanced_optimization_core import (
    AIOptimizationAgent, QuantumNeuralFusion, EvolutionaryOptimizer,
    HardwareAwareOptimizer, MegaEnhancedOptimizationCore,
    create_mega_enhanced_optimization_core
)
from .supreme_optimization_core import (
    NeuralArchitectureOptimizer, DynamicComputationGraph, SelfModifyingOptimizer,
    QuantumComputingSimulator, SupremeOptimizationCore,
    create_supreme_optimization_core
)
from .transcendent_optimization_core import (
    ConsciousnessSimulator, MultidimensionalOptimizer, TemporalOptimizer,
    TranscendentOptimizationCore, create_transcendent_optimization_core
)
from .rl_pruning import RLPruning, RLPruningAgent, RLPruningOptimizations, create_rl_pruning, create_rl_pruning_agent
from .optimization_registry import OptimizationRegistry, apply_optimizations, get_optimization_config, register_optimization, get_optimization_report
from .advanced_optimization_registry_v2 import AdvancedOptimizationConfig, get_advanced_optimization_config, apply_advanced_optimizations, get_advanced_optimization_report
from .enhanced_mcts_optimizer import EnhancedMCTSWithBenchmarks, EnhancedMCTSBenchmarkArgs, create_enhanced_mcts_with_benchmarks, benchmark_mcts_comparison
from .olympiad_benchmarks import OlympiadBenchmarkSuite, OlympiadBenchmarkConfig, OlympiadProblem, ProblemCategory, DifficultyLevel, get_olympiad_benchmark_config, create_olympiad_benchmark_suite
from .memory_optimizations import MemoryOptimizer, MemoryOptimizationConfig, create_memory_optimizer
from .computational_optimizations import FusedAttention, BatchOptimizer, ComputationalOptimizer, create_computational_optimizer
from .optimization_profiles import OptimizationProfile, get_optimization_profiles, apply_optimization_profile

from .hybrid_optimization_core import (
    HybridOptimizationCore, HybridOptimizationConfig, CandidateSelector,
    HybridOptimizationStrategy, HybridRLOptimizer, PolicyNetwork, ValueNetwork,
    OptimizationEnvironment, create_hybrid_optimization_core
)

from .enhanced_parameter_optimizer import (
    EnhancedParameterOptimizer, EnhancedParameterConfig,
    create_enhanced_parameter_optimizer, optimize_model_parameters
)

from .parameter_optimization_utils import (
    calculate_parameter_efficiency, optimize_learning_rate_schedule,
    optimize_rl_parameters, optimize_temperature_parameters,
    optimize_quantization_parameters, optimize_memory_parameters,
    generate_model_specific_optimizations, benchmark_parameter_optimization
)

__all__ = [
    'OptimizedLayerNorm',
    'OptimizedRMSNorm',
    'CUDAOptimizations',

    'TritonLayerNorm',
    'TritonOptimizations',
    'EnhancedGRPOTrainer',
    'EnhancedGRPOArgs',
    'KalmanFilter',
    'MCTSOptimizer',
    'MCTSOptimizationArgs',
    'create_mcts_optimizer',
    'EnhancedPPOActor',
    'ParallelTrainingConfig',
    'create_parallel_actor',
    'ReplayBuffer',
    'Experience',
    'PrioritizedExperienceReplay',
    'create_experience_buffer',
    'GRPOLoss',
    'EnhancedGRPOLoss',
    'AdversarialLoss',
    'CurriculumLoss',
    'create_loss_function',
    'GRPORewardFunction',
    'AdaptiveRewardFunction',
    'MultiObjectiveRewardFunction',
    'create_reward_function',
    'AdvancedRMSNorm',
    'LlamaRMSNorm',
    'CRMSNorm',
    'AdvancedNormalizationOptimizations',
    'create_advanced_rms_norm',
    'create_llama_rms_norm',
    'create_crms_norm',
    'RotaryEmbedding',
    'LlamaRotaryEmbedding',
    'FixedLlamaRotaryEmbedding',
    'AliBi',
    'SinusoidalPositionalEmbedding',
    'PositionalEncodingOptimizations',
    'create_rotary_embedding',
    'create_llama_rotary_embedding',
    'create_alibi',
    'create_sinusoidal_embedding',
    'SwiGLU',
    'GatedMLP',
    'MixtureOfExperts',
    'AdaptiveMLP',
    'EnhancedMLPOptimizations',
    'create_swiglu',
    'create_gated_mlp',
    'create_mixture_of_experts',
    'create_adaptive_mlp',
    'RLPruning',
    'RLPruningAgent',
    'RLPruningOptimizations',
    'create_rl_pruning',
    'create_rl_pruning_agent',
    'OptimizationRegistry',
    'apply_optimizations',
    'get_optimization_config',
    'register_optimization',
    'get_optimization_report',
    'AdvancedOptimizationConfig',
    'get_advanced_optimization_config',
    'apply_advanced_optimizations',
    'get_advanced_optimization_report',
    'EnhancedMCTSWithBenchmarks',
    'EnhancedMCTSBenchmarkArgs',
    'create_enhanced_mcts_with_benchmarks',
    'benchmark_mcts_comparison',
    'OlympiadBenchmarkSuite',
    'OlympiadBenchmarkConfig',
    'OlympiadProblem',
    'ProblemCategory',
    'DifficultyLevel',
    'get_olympiad_benchmark_config',
    'create_olympiad_benchmark_suite',
    'MemoryOptimizer',
    'MemoryOptimizationConfig',
    'create_memory_optimizer',
    'FusedAttention',
    'BatchOptimizer',
    'ComputationalOptimizer',
    'create_computational_optimizer',
    'OptimizationProfile',
    'get_optimization_profiles',
    'apply_optimization_profile',
    'FusedLayerNormLinear',
    'FusedAttentionMLP',
    'KernelFusionOptimizer',
    'create_kernel_fusion_optimizer',
    'QuantizedLinear',
    'QuantizedLayerNorm',
    'AdvancedQuantizationOptimizer',
    'create_quantization_optimizer',
    'TensorPool',
    'ActivationCache',
    'MemoryPoolingOptimizer',
    'create_memory_pooling_optimizer',
    'get_global_tensor_pool',
    'get_global_activation_cache',
    'AdvancedCUDAConfig',
    'FusedKernelOptimizer',
    'MemoryCoalescingOptimizer',
    'QuantizationKernelOptimizer',
    'EnhancedCUDAOptimizations',
    'create_enhanced_cuda_optimizer',
    'UltraOptimizedLayerNorm',
    'AdaptiveQuantization',
    'DynamicKernelFusion',
    'IntelligentMemoryManager',
    'UltraOptimizationCore',
    'create_ultra_optimization_core',
    'SuperOptimizedAttention',
    'AdaptiveComputationTime',
    'SuperOptimizedMLP',
    'ProgressiveOptimization',
    'SuperOptimizationCore',
    'create_super_optimization_core',
    'SelfOptimizingLayerNorm',
    'AdaptiveOptimizationScheduler',
    'DynamicComputationGraph',
    'MetaOptimizationCore',
    'create_meta_optimization_core',
    'HyperOptimizedLinear',
    'NeuralArchitectureOptimizer',
    'AdvancedGradientOptimizer',
    'HyperOptimizationCore',
    'create_hyper_optimization_core',
    'QuantumInspiredLinear',
    'QuantumAttention',
    'QuantumLayerNorm',
    'QuantumOptimizationCore',
    'create_quantum_optimization_core',
    'ArchitectureGene',
    'ArchitectureChromosome',
    'NeuralArchitectureSearchOptimizer',
    'NASOptimizationCore',
    'create_nas_optimization_core',
    'AdaptivePrecisionOptimizer',
    'DynamicKernelFusionOptimizer',
    'IntelligentMemoryManager',
    'SelfOptimizingComponent',
    'EnhancedOptimizedLayerNorm',
    'EnhancedOptimizationCore',
    'create_enhanced_optimization_core',
    'NeuralCodeOptimizer',
    'AdaptiveAlgorithmSelector',
    'PredictiveOptimizer',
    'SelfEvolvingKernel',
    'RealTimeProfiler',
    'UltraEnhancedOptimizationCore',
    'create_ultra_enhanced_optimization_core',
    'AIOptimizationAgent',
    'QuantumNeuralFusion',
    'EvolutionaryOptimizer',
    'HardwareAwareOptimizer',
    'MegaEnhancedOptimizationCore',
    'create_mega_enhanced_optimization_core',
    'NeuralArchitectureOptimizer',
    'DynamicComputationGraph',
    'SelfModifyingOptimizer',
    'QuantumComputingSimulator',
    'SupremeOptimizationCore',
    'create_supreme_optimization_core',
    'ConsciousnessSimulator',
    'MultidimensionalOptimizer',
    'TemporalOptimizer',
    'TranscendentOptimizationCore',
    'create_transcendent_optimization_core',
    'FusedMultiHeadAttention',
    'AttentionFusionOptimizer',
    'create_attention_fusion_optimizer',
    'AdvancedTritonOptimizations',
    'create_advanced_triton_optimizer',
    'AdvancedMemoryManager',
    'KernelFusionOptimizer',
    'ComputeOptimizer',
    'AdvancedCUDAOptimizations',
    'create_advanced_cuda_optimizer',
    'FusedLayerNormLinear',
    'FusedAttentionMLP',
    'AdvancedKernelFusionOptimizer',
    'create_kernel_fusion_optimizer',
    'QuantizedLinear',
    'QuantizedLayerNorm',
    'MixedPrecisionOptimizer',
    'AdvancedQuantizationOptimizer',
    'create_quantization_optimizer',
    'TensorPool',
    'ActivationCache',
    'GradientCache',
    'MemoryPoolingOptimizer',
    'create_memory_pooling_optimizer',
    
    'HybridOptimizationCore',
    'HybridOptimizationConfig',
    'CandidateSelector',
    'HybridOptimizationStrategy',
    'HybridRLOptimizer',
    'PolicyNetwork',
    'ValueNetwork',
    'OptimizationEnvironment',
    'create_hybrid_optimization_core',
    
    'EnhancedParameterOptimizer',
    'EnhancedParameterConfig',
    'create_enhanced_parameter_optimizer',
    'optimize_model_parameters',
    
    'calculate_parameter_efficiency',
    'optimize_learning_rate_schedule',
    'optimize_rl_parameters',
    'optimize_temperature_parameters',
    'optimize_quantization_parameters',
    'optimize_memory_parameters',
    'generate_model_specific_optimizations',
    'benchmark_parameter_optimization'
]

__version__ = "13.1.0"
