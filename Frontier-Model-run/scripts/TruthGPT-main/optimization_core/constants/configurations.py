"""
Configuration constants for TruthGPT Optimization Core
"""

from .enums import OptimizationLevel, OptimizationFramework, OptimizationTechnique, OptimizationType

# Default configurations
DEFAULT_CONFIGS = {
    'optimization_level': OptimizationLevel.BASIC,
    'framework': OptimizationFramework.PYTORCH,
    'technique': OptimizationTechnique.JIT_COMPILATION,
    'target_speed_improvement': 10.0,
    'target_memory_reduction': 0.1,
    'target_energy_efficiency': 2.0,
    'target_accuracy_preservation': 0.99,
    'max_optimization_time': 300.0,  # seconds
    'max_memory_usage': 8.0,  # GB
    'max_cpu_usage': 80.0,  # percentage
    'max_gpu_usage': 90.0,  # percentage
    'enable_quantization': True,
    'enable_pruning': False,
    'enable_distillation': False,
    'enable_nas': False,
    'enable_meta_learning': False,
    'enable_rl': False,
    'enable_evolutionary': False,
    'enable_bayesian': False,
    'enable_gradient': True,
    'enable_attention': True,
    'enable_transformer': True,
    'enable_convolution': False,
    'enable_recurrent': False,
    'enable_activation': True,
    'enable_normalization': True,
    'enable_dropout': True,
    'enable_batch': True,
    'enable_sequence': True,
    'enable_temporal': False,
    'enable_spatial': False,
    'enable_channel': False,
    'enable_frequency': False,
    'enable_spectral': False
}

# Optimization profiles
OPTIMIZATION_PROFILES = {
    'speed_focused': {
        'target_speed_improvement': 100.0,
        'target_memory_reduction': 0.1,
        'target_energy_efficiency': 2.0,
        'target_accuracy_preservation': 0.95,
        'techniques': [OptimizationTechnique.JIT_COMPILATION, OptimizationTechnique.INDUCTOR, OptimizationTechnique.DYNAMO]
    },
    'memory_focused': {
        'target_speed_improvement': 10.0,
        'target_memory_reduction': 0.8,
        'target_energy_efficiency': 5.0,
        'target_accuracy_preservation': 0.9,
        'techniques': [OptimizationTechnique.QUANTIZATION, OptimizationType.PRUNING, OptimizationType.DISTILLATION]
    },
    'energy_focused': {
        'target_speed_improvement': 20.0,
        'target_memory_reduction': 0.3,
        'target_energy_efficiency': 50.0,
        'target_accuracy_preservation': 0.95,
        'techniques': [OptimizationTechnique.MIXED_PRECISION, OptimizationTechnique.QUANTIZATION, OptimizationType.PRUNING]
    },
    'accuracy_focused': {
        'target_speed_improvement': 5.0,
        'target_memory_reduction': 0.1,
        'target_energy_efficiency': 2.0,
        'target_accuracy_preservation': 0.99,
        'techniques': [OptimizationTechnique.JIT_COMPILATION, OptimizationTechnique.INDUCTOR, OptimizationTechnique.DYNAMO]
    },
    'balanced': {
        'target_speed_improvement': 50.0,
        'target_memory_reduction': 0.5,
        'target_energy_efficiency': 10.0,
        'target_accuracy_preservation': 0.95,
        'techniques': [OptimizationTechnique.JIT_COMPILATION, OptimizationTechnique.QUANTIZATION, OptimizationTechnique.MIXED_PRECISION]
    },
    'extreme': {
        'target_speed_improvement': 1000.0,
        'target_memory_reduction': 0.9,
        'target_energy_efficiency': 100.0,
        'target_accuracy_preservation': 0.9,
        'techniques': [OptimizationTechnique.QUANTIZATION, OptimizationType.PRUNING, OptimizationType.DISTILLATION, OptimizationType.NEURAL_ARCHITECTURE_SEARCH]
    }
}

# Hardware configurations
HARDWARE_CONFIGS = {
    'cpu_only': {
        'gpu_enabled': False,
        'tpu_enabled': False,
        'cpu_cores': 8,
        'memory_gb': 16,
        'storage_gb': 100
    },
    'gpu_enabled': {
        'gpu_enabled': True,
        'tpu_enabled': False,
        'gpu_memory_gb': 8,
        'cpu_cores': 16,
        'memory_gb': 32,
        'storage_gb': 500
    },
    'tpu_enabled': {
        'gpu_enabled': False,
        'tpu_enabled': True,
        'tpu_cores': 8,
        'cpu_cores': 32,
        'memory_gb': 64,
        'storage_gb': 1000
    },
    'multi_gpu': {
        'gpu_enabled': True,
        'tpu_enabled': False,
        'gpu_count': 4,
        'gpu_memory_gb': 32,
        'cpu_cores': 64,
        'memory_gb': 128,
        'storage_gb': 2000
    },
    'distributed': {
        'gpu_enabled': True,
        'tpu_enabled': True,
        'node_count': 8,
        'gpu_count': 32,
        'tpu_cores': 64,
        'cpu_cores': 512,
        'memory_gb': 1024,
        'storage_gb': 10000
    }
}

# Software configurations
SOFTWARE_CONFIGS = {
    'pytorch': {
        'framework': OptimizationFramework.PYTORCH,
        'version': '2.0.0',
        'cuda_version': '11.8',
        'cudnn_version': '8.7',
        'dependencies': ['torch', 'torchvision', 'torchaudio', 'transformers']
    },
    'tensorflow': {
        'framework': OptimizationFramework.TENSORFLOW,
        'version': '2.13.0',
        'cuda_version': '11.8',
        'cudnn_version': '8.7',
        'dependencies': ['tensorflow', 'tensorflow-gpu', 'keras', 'tensorflow-hub']
    },
    'jax': {
        'framework': OptimizationFramework.JAX,
        'version': '0.4.0',
        'cuda_version': '11.8',
        'cudnn_version': '8.7',
        'dependencies': ['jax', 'jaxlib', 'flax', 'optax']
    },
    'onnx': {
        'framework': OptimizationFramework.ONNX,
        'version': '1.14.0',
        'dependencies': ['onnx', 'onnxruntime', 'onnxruntime-gpu']
    },
    'torchscript': {
        'framework': OptimizationFramework.TORCHSCRIPT,
        'version': '2.0.0',
        'dependencies': ['torch', 'torchscript']
    },
    'tensorrt': {
        'framework': OptimizationFramework.TRT,
        'version': '8.6.0',
        'cuda_version': '11.8',
        'dependencies': ['tensorrt', 'pycuda']
    },
    'openvino': {
        'framework': OptimizationFramework.OPENVINO,
        'version': '2023.0.0',
        'dependencies': ['openvino', 'openvino-dev']
    },
    'coreml': {
        'framework': OptimizationFramework.COREML,
        'version': '7.0.0',
        'dependencies': ['coremltools', 'onnx']
    },
    'tflite': {
        'framework': OptimizationFramework.TFLITE,
        'version': '2.13.0',
        'dependencies': ['tensorflow-lite', 'tensorflow-lite-gpu']
    }
}

# Model configurations
MODEL_CONFIGS = {
    'small': {
        'parameters': 1000000,
        'layers': 12,
        'hidden_size': 512,
        'attention_heads': 8,
        'vocab_size': 50000
    },
    'medium': {
        'parameters': 10000000,
        'layers': 24,
        'hidden_size': 1024,
        'attention_heads': 16,
        'vocab_size': 100000
    },
    'large': {
        'parameters': 100000000,
        'layers': 48,
        'hidden_size': 2048,
        'attention_heads': 32,
        'vocab_size': 200000
    },
    'xlarge': {
        'parameters': 1000000000,
        'layers': 96,
        'hidden_size': 4096,
        'attention_heads': 64,
        'vocab_size': 500000
    },
    'xxlarge': {
        'parameters': 10000000000,
        'layers': 192,
        'hidden_size': 8192,
        'attention_heads': 128,
        'vocab_size': 1000000
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'small': {
        'samples': 10000,
        'sequence_length': 128,
        'vocab_size': 50000,
        'batch_size': 32
    },
    'medium': {
        'samples': 100000,
        'sequence_length': 256,
        'vocab_size': 100000,
        'batch_size': 64
    },
    'large': {
        'samples': 1000000,
        'sequence_length': 512,
        'vocab_size': 200000,
        'batch_size': 128
    },
    'xlarge': {
        'samples': 10000000,
        'sequence_length': 1024,
        'vocab_size': 500000,
        'batch_size': 256
    },
    'xxlarge': {
        'samples': 100000000,
        'sequence_length': 2048,
        'vocab_size': 1000000,
        'batch_size': 512
    }
}

# Training configurations
TRAINING_CONFIGS = {
    'basic': {
        'epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam',
        'scheduler': 'cosine'
    },
    'advanced': {
        'epochs': 50,
        'learning_rate': 0.0001,
        'batch_size': 64,
        'optimizer': 'adamw',
        'scheduler': 'cosine_with_restarts'
    },
    'expert': {
        'epochs': 100,
        'learning_rate': 0.00001,
        'batch_size': 128,
        'optimizer': 'adamw',
        'scheduler': 'cosine_with_restarts',
        'warmup_steps': 1000
    },
    'master': {
        'epochs': 200,
        'learning_rate': 0.000001,
        'batch_size': 256,
        'optimizer': 'adamw',
        'scheduler': 'cosine_with_restarts',
        'warmup_steps': 5000,
        'gradient_clipping': 1.0
    },
    'legendary': {
        'epochs': 500,
        'learning_rate': 0.0000001,
        'batch_size': 512,
        'optimizer': 'adamw',
        'scheduler': 'cosine_with_restarts',
        'warmup_steps': 10000,
        'gradient_clipping': 0.5,
        'weight_decay': 0.01
    }
}

# Evaluation configurations
EVALUATION_CONFIGS = {
    'basic': {
        'metrics': ['accuracy', 'loss'],
        'test_split': 0.2,
        'validation_split': 0.1,
        'cross_validation': False
    },
    'advanced': {
        'metrics': ['accuracy', 'loss', 'f1', 'precision', 'recall'],
        'test_split': 0.2,
        'validation_split': 0.1,
        'cross_validation': True,
        'k_folds': 5
    },
    'expert': {
        'metrics': ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc', 'mcc'],
        'test_split': 0.2,
        'validation_split': 0.1,
        'cross_validation': True,
        'k_folds': 10,
        'stratified': True
    },
    'master': {
        'metrics': ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc', 'mcc', 'bleu', 'rouge'],
        'test_split': 0.2,
        'validation_split': 0.1,
        'cross_validation': True,
        'k_folds': 10,
        'stratified': True,
        'bootstrap': True,
        'bootstrap_samples': 1000
    },
    'legendary': {
        'metrics': ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc', 'mcc', 'bleu', 'rouge', 'meteor', 'bertscore'],
        'test_split': 0.2,
        'validation_split': 0.1,
        'cross_validation': True,
        'k_folds': 10,
        'stratified': True,
        'bootstrap': True,
        'bootstrap_samples': 10000,
        'monte_carlo': True,
        'monte_carlo_samples': 1000
    }
}

# Deployment configurations
DEPLOYMENT_CONFIGS = {
    'local': {
        'environment': 'local',
        'scaling': 'single',
        'monitoring': 'basic',
        'logging': 'basic'
    },
    'cloud': {
        'environment': 'cloud',
        'scaling': 'auto',
        'monitoring': 'advanced',
        'logging': 'advanced'
    },
    'edge': {
        'environment': 'edge',
        'scaling': 'fixed',
        'monitoring': 'basic',
        'logging': 'basic'
    },
    'distributed': {
        'environment': 'distributed',
        'scaling': 'horizontal',
        'monitoring': 'advanced',
        'logging': 'advanced'
    },
    'production': {
        'environment': 'production',
        'scaling': 'auto',
        'monitoring': 'comprehensive',
        'logging': 'comprehensive'
    }
}

# Monitoring configurations
MONITORING_CONFIGS = {
    'basic': {
        'metrics': ['cpu', 'memory', 'gpu'],
        'frequency': 60,  # seconds
        'retention': 7,  # days
        'alerts': False
    },
    'advanced': {
        'metrics': ['cpu', 'memory', 'gpu', 'disk', 'network'],
        'frequency': 30,  # seconds
        'retention': 30,  # days
        'alerts': True
    },
    'comprehensive': {
        'metrics': ['cpu', 'memory', 'gpu', 'disk', 'network', 'temperature', 'power'],
        'frequency': 10,  # seconds
        'retention': 90,  # days
        'alerts': True,
        'dashboards': True,
        'reports': True
    }
}

# Logging configurations
LOGGING_CONFIGS = {
    'basic': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'handlers': ['console'],
        'retention': 7  # days
    },
    'advanced': {
        'level': 'DEBUG',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d',
        'handlers': ['console', 'file'],
        'retention': 30  # days
    },
    'comprehensive': {
        'level': 'DEBUG',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d - %(funcName)s',
        'handlers': ['console', 'file', 'syslog'],
        'retention': 90  # days
    }
}

# Security configurations
SECURITY_CONFIGS = {
    'basic': {
        'encryption': False,
        'authentication': False,
        'authorization': False,
        'audit': False
    },
    'advanced': {
        'encryption': True,
        'authentication': True,
        'authorization': True,
        'audit': True
    },
    'comprehensive': {
        'encryption': True,
        'authentication': True,
        'authorization': True,
        'audit': True,
        'mfa': True,
        'rbac': True,
        'encryption_at_rest': True,
        'encryption_in_transit': True
    }
}

# Compliance configurations
COMPLIANCE_CONFIGS = {
    'basic': {
        'gdpr': False,
        'ccpa': False,
        'hipaa': False,
        'sox': False
    },
    'advanced': {
        'gdpr': True,
        'ccpa': True,
        'hipaa': False,
        'sox': False
    },
    'comprehensive': {
        'gdpr': True,
        'ccpa': True,
        'hipaa': True,
        'sox': True,
        'pci_dss': True,
        'iso27001': True
    }
}

# Quality configurations
QUALITY_CONFIGS = {
    'basic': {
        'testing': 'unit',
        'coverage': 0.8,
        'linting': True,
        'formatting': True
    },
    'advanced': {
        'testing': 'unit_integration',
        'coverage': 0.9,
        'linting': True,
        'formatting': True,
        'type_checking': True
    },
    'comprehensive': {
        'testing': 'unit_integration_e2e',
        'coverage': 0.95,
        'linting': True,
        'formatting': True,
        'type_checking': True,
        'security_scanning': True,
        'performance_testing': True
    }
}

# Documentation configurations
DOCUMENTATION_CONFIGS = {
    'basic': {
        'api_docs': True,
        'user_guide': True,
        'examples': True,
        'tutorials': False
    },
    'advanced': {
        'api_docs': True,
        'user_guide': True,
        'examples': True,
        'tutorials': True,
        'architecture': True,
        'design': True
    },
    'comprehensive': {
        'api_docs': True,
        'user_guide': True,
        'examples': True,
        'tutorials': True,
        'architecture': True,
        'design': True,
        'deployment': True,
        'troubleshooting': True,
        'faq': True,
        'changelog': True
    }
}

__all__ = [
    'DEFAULT_CONFIGS',
    'OPTIMIZATION_PROFILES',
    'HARDWARE_CONFIGS',
    'SOFTWARE_CONFIGS',
    'MODEL_CONFIGS',
    'DATASET_CONFIGS',
    'TRAINING_CONFIGS',
    'EVALUATION_CONFIGS',
    'DEPLOYMENT_CONFIGS',
    'MONITORING_CONFIGS',
    'LOGGING_CONFIGS',
    'SECURITY_CONFIGS',
    'COMPLIANCE_CONFIGS',
    'QUALITY_CONFIGS',
    'DOCUMENTATION_CONFIGS',
]


