"""
Performance-related constants for TruthGPT Optimization Core
"""

from .enums import OptimizationLevel, OptimizationFramework, OptimizationTechnique

# Speed improvement constants
SPEED_IMPROVEMENTS = {
    OptimizationLevel.BASIC: 10.0,
    OptimizationLevel.ADVANCED: 50.0,
    OptimizationLevel.EXPERT: 100.0,
    OptimizationLevel.MASTER: 500.0,
    OptimizationLevel.LEGENDARY: 1000.0,
    OptimizationLevel.TRANSCENDENT: 10000.0,
    OptimizationLevel.DIVINE: 100000.0,
    OptimizationLevel.OMNIPOTENT: 1000000.0,
    OptimizationLevel.INFINITE: 10000000.0,
    OptimizationLevel.ULTIMATE: 100000000.0,
    OptimizationLevel.SUPREME: 1000000000.0,
    OptimizationLevel.REFACTORED: 10000000000.0
}

# Memory reduction constants
MEMORY_REDUCTIONS = {
    OptimizationLevel.BASIC: 0.1,
    OptimizationLevel.ADVANCED: 0.3,
    OptimizationLevel.EXPERT: 0.5,
    OptimizationLevel.MASTER: 0.7,
    OptimizationLevel.LEGENDARY: 0.8,
    OptimizationLevel.TRANSCENDENT: 0.9,
    OptimizationLevel.DIVINE: 0.95,
    OptimizationLevel.OMNIPOTENT: 0.98,
    OptimizationLevel.INFINITE: 0.99,
    OptimizationLevel.ULTIMATE: 0.995,
    OptimizationLevel.SUPREME: 0.999,
    OptimizationLevel.REFACTORED: 0.9999
}

# Energy efficiency constants
ENERGY_EFFICIENCIES = {
    OptimizationLevel.BASIC: 2.0,
    OptimizationLevel.ADVANCED: 5.0,
    OptimizationLevel.EXPERT: 10.0,
    OptimizationLevel.MASTER: 50.0,
    OptimizationLevel.LEGENDARY: 100.0,
    OptimizationLevel.TRANSCENDENT: 500.0,
    OptimizationLevel.DIVINE: 1000.0,
    OptimizationLevel.OMNIPOTENT: 5000.0,
    OptimizationLevel.INFINITE: 10000.0,
    OptimizationLevel.ULTIMATE: 50000.0,
    OptimizationLevel.SUPREME: 100000.0,
    OptimizationLevel.REFACTORED: 1000000.0
}

# Accuracy preservation constants
ACCURACY_PRESERVATIONS = {
    OptimizationLevel.BASIC: 0.99,
    OptimizationLevel.ADVANCED: 0.98,
    OptimizationLevel.EXPERT: 0.97,
    OptimizationLevel.MASTER: 0.96,
    OptimizationLevel.LEGENDARY: 0.95,
    OptimizationLevel.TRANSCENDENT: 0.94,
    OptimizationLevel.DIVINE: 0.93,
    OptimizationLevel.OMNIPOTENT: 0.92,
    OptimizationLevel.INFINITE: 0.91,
    OptimizationLevel.ULTIMATE: 0.90,
    OptimizationLevel.SUPREME: 0.89,
    OptimizationLevel.REFACTORED: 0.88
}

# Framework benefits
FRAMEWORK_BENEFITS = {
    OptimizationFramework.PYTORCH: 0.3,
    OptimizationFramework.TENSORFLOW: 0.25,
    OptimizationFramework.JAX: 0.2,
    OptimizationFramework.ONNX: 0.15,
    OptimizationFramework.TORCHSCRIPT: 0.1,
    OptimizationFramework.TRT: 0.05,
    OptimizationFramework.OPENVINO: 0.05,
    OptimizationFramework.COREML: 0.05,
    OptimizationFramework.TFLITE: 0.05,
    OptimizationFramework.QUANTIZATION: 0.1,
    OptimizationFramework.DISTRIBUTED: 0.2,
    OptimizationFramework.PARALLEL: 0.15
}

# Optimization techniques benefits
TECHNIQUE_BENEFITS = {
    OptimizationTechnique.JIT_COMPILATION: 0.2,
    OptimizationTechnique.QUANTIZATION: 0.3,
    OptimizationTechnique.MIXED_PRECISION: 0.15,
    OptimizationTechnique.INDUCTOR: 0.25,
    OptimizationTechnique.DYNAMO: 0.2,
    OptimizationTechnique.AUTOGRAD: 0.1,
    OptimizationTechnique.DISTRIBUTED: 0.3,
    OptimizationTechnique.FX: 0.15,
    OptimizationTechnique.AMP: 0.1,
    OptimizationTechnique.COMPILE: 0.2,
    OptimizationTechnique.XLA: 0.25,
    OptimizationTechnique.GRAPPLER: 0.2,
    OptimizationTechnique.TF_QUANTIZATION: 0.3,
    OptimizationTechnique.TF_DISTRIBUTED: 0.3,
    OptimizationTechnique.TF_FUNCTION: 0.15,
    OptimizationTechnique.TF_MIXED_PRECISION: 0.1,
    OptimizationTechnique.TF_KERAS: 0.1,
    OptimizationTechnique.TF_AUTOGRAPH: 0.15,
    OptimizationTechnique.TF_TPU: 0.2,
    OptimizationTechnique.TF_GPU: 0.15,
    OptimizationTechnique.QUANTUM_NEURAL: 0.4,
    OptimizationTechnique.QUANTUM_ENTANGLEMENT: 0.35,
    OptimizationTechnique.QUANTUM_SUPERPOSITION: 0.3,
    OptimizationTechnique.QUANTUM_INTERFERENCE: 0.25,
    OptimizationTechnique.QUANTUM_TUNNELING: 0.2,
    OptimizationTechnique.QUANTUM_COHERENCE: 0.15,
    OptimizationTechnique.QUANTUM_DECOHERENCE: 0.1,
    OptimizationTechnique.NEURAL_NETWORK: 0.2,
    OptimizationTechnique.DEEP_LEARNING: 0.25,
    OptimizationTechnique.MACHINE_LEARNING: 0.15,
    OptimizationTechnique.ARTIFICIAL_INTELLIGENCE: 0.3,
    OptimizationTechnique.AI_ENGINE: 0.25,
    OptimizationTechnique.TRUTHGPT_AI: 0.35,
    OptimizationTechnique.CROSS_FRAMEWORK_FUSION: 0.4,
    OptimizationTechnique.UNIFIED_QUANTIZATION: 0.35,
    OptimizationTechnique.HYBRID_DISTRIBUTED: 0.3,
    OptimizationTechnique.CROSS_PLATFORM: 0.25,
    OptimizationTechnique.FRAMEWORK_AGNOSTIC: 0.2,
    OptimizationTechnique.UNIVERSAL_COMPILATION: 0.3,
    OptimizationTechnique.CROSS_BACKEND: 0.25,
    OptimizationTechnique.ATTENTION_OPTIMIZATION: 0.3,
    OptimizationTechnique.TRANSFORMER_OPTIMIZATION: 0.35,
    OptimizationTechnique.EMBEDDING_OPTIMIZATION: 0.25,
    OptimizationTechnique.POSITIONAL_ENCODING: 0.2,
    OptimizationTechnique.MLP_OPTIMIZATION: 0.25,
    OptimizationTechnique.LAYER_NORM_OPTIMIZATION: 0.2,
    OptimizationTechnique.DROPOUT_OPTIMIZATION: 0.15,
    OptimizationTechnique.ACTIVATION_OPTIMIZATION: 0.2
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'min_speed_improvement': 1.1,
    'min_memory_reduction': 0.01,
    'min_energy_efficiency': 1.1,
    'min_accuracy_preservation': 0.8,
    'max_optimization_time': 600.0,  # seconds
    'max_memory_usage': 16.0,  # GB
    'max_cpu_usage': 95.0,  # percentage
    'max_gpu_usage': 95.0,  # percentage
    'min_cache_hit_rate': 0.8,
    'min_throughput': 1.0,  # samples/second
    'max_latency': 1000.0,  # milliseconds
    'min_bandwidth': 1.0,  # GB/s
    'min_compute_efficiency': 0.5,
    'min_memory_efficiency': 0.5,
    'min_energy_efficiency': 0.5
}

__all__ = [
    'SPEED_IMPROVEMENTS',
    'MEMORY_REDUCTIONS',
    'ENERGY_EFFICIENCIES',
    'ACCURACY_PRESERVATIONS',
    'FRAMEWORK_BENEFITS',
    'TECHNIQUE_BENEFITS',
    'PERFORMANCE_THRESHOLDS',
]


