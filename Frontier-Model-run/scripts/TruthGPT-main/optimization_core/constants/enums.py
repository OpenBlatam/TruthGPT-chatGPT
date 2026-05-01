"""
Enumeration definitions for TruthGPT Optimization Core
"""

from enum import Enum

class OptimizationFramework(Enum):
    """Supported optimization frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TRT = "tensorrt"
    OPENVINO = "openvino"
    COREML = "coreml"
    TFLITE = "tflite"
    QUANTIZATION = "quantization"
    DISTRIBUTED = "distributed"
    PARALLEL = "parallel"

class OptimizationLevel(Enum):
    """Optimization levels for TruthGPT."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"
    LEGENDARY = "legendary"
    TRANSCENDENT = "transcendent"
    DIVINE = "divine"
    OMNIPOTENT = "omnipotent"
    INFINITE = "infinite"
    ULTIMATE = "ultimate"
    SUPREME = "supreme"
    REFACTORED = "refactored"

class OptimizationType(Enum):
    """Types of optimizations."""
    SPEED = "speed"
    MEMORY = "memory"
    ENERGY = "energy"
    ACCURACY = "accuracy"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    META_LEARNING = "meta_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    GRADIENT = "gradient"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    CONVOLUTION = "convolution"
    RECURRENT = "recurrent"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    DROPOUT = "dropout"
    BATCH = "batch"
    SEQUENCE = "sequence"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CHANNEL = "channel"
    FREQUENCY = "frequency"
    SPECTRAL = "spectral"

class OptimizationTechnique(Enum):
    """Specific optimization techniques."""
    # PyTorch techniques
    JIT_COMPILATION = "jit_compilation"
    QUANTIZATION = "quantization"
    MIXED_PRECISION = "mixed_precision"
    INDUCTOR = "inductor"
    DYNAMO = "dynamo"
    AUTOGRAD = "autograd"
    DISTRIBUTED = "distributed"
    FX = "fx"
    AMP = "amp"
    COMPILE = "compile"
    
    # TensorFlow techniques
    XLA = "xla"
    GRAPPLER = "grappler"
    TF_QUANTIZATION = "tf_quantization"
    TF_DISTRIBUTED = "tf_distributed"
    TF_FUNCTION = "tf_function"
    TF_MIXED_PRECISION = "tf_mixed_precision"
    TF_KERAS = "tf_keras"
    TF_AUTOGRAPH = "tf_autograph"
    TF_TPU = "tf_tpu"
    TF_GPU = "tf_gpu"
    
    # Quantum techniques
    QUANTUM_NEURAL = "quantum_neural"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    QUANTUM_INTERFERENCE = "quantum_interference"
    QUANTUM_TUNNELING = "quantum_tunneling"
    QUANTUM_COHERENCE = "quantum_coherence"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    
    # AI techniques
    NEURAL_NETWORK = "neural_network"
    DEEP_LEARNING = "deep_learning"
    MACHINE_LEARNING = "machine_learning"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    AI_ENGINE = "ai_engine"
    TRUTHGPT_AI = "truthgpt_ai"
    
    # Hybrid techniques
    CROSS_FRAMEWORK_FUSION = "cross_framework_fusion"
    UNIFIED_QUANTIZATION = "unified_quantization"
    HYBRID_DISTRIBUTED = "hybrid_distributed"
    CROSS_PLATFORM = "cross_platform"
    FRAMEWORK_AGNOSTIC = "framework_agnostic"
    UNIVERSAL_COMPILATION = "universal_compilation"
    CROSS_BACKEND = "cross_backend"
    
    # TruthGPT specific
    ATTENTION_OPTIMIZATION = "attention_optimization"
    TRANSFORMER_OPTIMIZATION = "transformer_optimization"
    EMBEDDING_OPTIMIZATION = "embedding_optimization"
    POSITIONAL_ENCODING = "positional_encoding"
    MLP_OPTIMIZATION = "mlp_optimization"
    LAYER_NORM_OPTIMIZATION = "layer_norm_optimization"
    DROPOUT_OPTIMIZATION = "dropout_optimization"
    ACTIVATION_OPTIMIZATION = "activation_optimization"

class OptimizationMetric(Enum):
    """Optimization metrics."""
    SPEED_IMPROVEMENT = "speed_improvement"
    MEMORY_REDUCTION = "memory_reduction"
    ACCURACY_PRESERVATION = "accuracy_preservation"
    ENERGY_EFFICIENCY = "energy_efficiency"
    PARAMETER_REDUCTION = "parameter_reduction"
    COMPRESSION_RATIO = "compression_ratio"
    CACHE_HIT_RATE = "cache_hit_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    BANDWIDTH = "bandwidth"
    COMPUTE_EFFICIENCY = "compute_efficiency"
    MEMORY_EFFICIENCY = "memory_efficiency"

class OptimizationResult(Enum):
    """Optimization result types."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    COMPUTE_ERROR = "compute_error"
    CONVERGENCE_ERROR = "convergence_error"
    ACCURACY_ERROR = "accuracy_error"

__all__ = [
    'OptimizationFramework',
    'OptimizationLevel',
    'OptimizationType',
    'OptimizationTechnique',
    'OptimizationMetric',
    'OptimizationResult',
]


