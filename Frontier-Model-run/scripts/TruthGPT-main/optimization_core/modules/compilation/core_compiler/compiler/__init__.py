"""
TruthGPT Compiler Infrastructure
Advanced compilation and optimization system for TruthGPT models
TensorFlow-style architecture with comprehensive compiler support
"""

# Core compiler infrastructure
from .core.compiler_core import (
    CompilerCore, CompilationTarget, OptimizationLevel, CompilationResult,
    CompilationConfig, CompilationContext, CompilationError,
    create_compiler_core, compilation_context
)

# AOT (Ahead-of-Time) Compilation
from .aot.aot_compiler import (
    AOTCompiler, AOTCompilationConfig, AOTOptimizationStrategy,
    create_aot_compiler, aot_compilation_context
)

# JIT (Just-in-Time) Compilation
from .jit.jit_compiler import (
    JITCompiler, JITCompilationConfig, JITOptimizationStrategy,
    create_jit_compiler, jit_compilation_context
)

# MLIR Compilation Infrastructure
from .mlir.mlir_compiler import (
    MLIRCompiler, MLIRDialect, MLIROptimizationPass, MLIRCompilationResult,
    create_mlir_compiler, mlir_compilation_context
)

# Plugin System
from .plugin.plugin_system import (
    CompilerPlugin, PluginManager, PluginRegistry, PluginInterface,
    create_plugin_manager, plugin_compilation_context
)

# TensorFlow to TensorRT Compilation
from .tf2tensorrt.tf2tensorrt_compiler import (
    TF2TensorRTCompiler, TensorRTConfig, TensorRTOptimizationLevel,
    create_tf2tensorrt_compiler, tf2tensorrt_compilation_context
)

# TensorFlow to XLA Compilation
from .tf2xla.tf2xla_compiler import (
    TF2XLACompiler, XLAConfig, XLAOptimizationLevel,
    create_tf2xla_compiler, tf2xla_compilation_context
)

# Compiler Utilities
from .utils.compiler_utils import (
    CompilerUtils, CodeGenerator, OptimizationAnalyzer,
    create_compiler_utils, compiler_utils_context
)

# Runtime Compilation
from .runtime.runtime_compiler import (
    RuntimeCompiler, RuntimeCompilationConfig, RuntimeOptimizationStrategy,
    RuntimeCompilationResult, RuntimeTarget, RuntimeOptimizationLevel,
    create_runtime_compiler, runtime_compilation_context
)

# Kernel Compilation
from .kernels.kernel_compiler import (
    KernelCompiler, KernelOptimizationLevel, KernelCompilationResult,
    KernelTarget, KernelConfig, KernelOptimizationPass,
    create_kernel_compiler, kernel_compilation_context
)

__all__ = [
    # Core compiler infrastructure
    'CompilerCore',
    'CompilationTarget',
    'OptimizationLevel',
    'CompilationResult',
    'CompilationConfig',
    'CompilationContext',
    'CompilationError',
    'create_compiler_core',
    'compilation_context',
    
    # AOT Compilation
    'AOTCompiler',
    'AOTCompilationConfig',
    'AOTOptimizationStrategy',
    'create_aot_compiler',
    'aot_compilation_context',
    
    # JIT Compilation
    'JITCompiler',
    'JITCompilationConfig',
    'JITOptimizationStrategy',
    'create_jit_compiler',
    'jit_compilation_context',
    
    # MLIR Compilation
    'MLIRCompiler',
    'MLIRDialect',
    'MLIROptimizationPass',
    'MLIRCompilationResult',
    'create_mlir_compiler',
    'mlir_compilation_context',
    
    # Plugin System
    'CompilerPlugin',
    'PluginManager',
    'PluginRegistry',
    'PluginInterface',
    'create_plugin_manager',
    'plugin_compilation_context',
    
    # TensorFlow to TensorRT
    'TF2TensorRTCompiler',
    'TensorRTConfig',
    'TensorRTOptimizationLevel',
    'create_tf2tensorrt_compiler',
    'tf2tensorrt_compilation_context',
    
    # TensorFlow to XLA
    'TF2XLACompiler',
    'XLAConfig',
    'XLAOptimizationLevel',
    'create_tf2xla_compiler',
    'tf2xla_compilation_context',
    
    # Compiler Utilities
    'CompilerUtils',
    'CodeGenerator',
    'OptimizationAnalyzer',
    'create_compiler_utils',
    'compiler_utils_context',
    
    # Runtime Compilation
    'RuntimeCompiler',
    'RuntimeCompilationConfig',
    'RuntimeOptimizationStrategy',
    'RuntimeCompilationResult',
    'RuntimeTarget',
    'RuntimeOptimizationLevel',
    'create_runtime_compiler',
    'runtime_compilation_context',
    
    # Kernel Compilation
    'KernelCompiler',
    'KernelOptimizationLevel',
    'KernelCompilationResult',
    'KernelTarget',
    'KernelConfig',
    'KernelOptimizationPass',
    'create_kernel_compiler',
    'kernel_compilation_context'
]

# Import distributed and neural compilers
try:
    from .distributed.distributed_compiler import (
        DistributedCompiler,
        DistributedCompilationConfig,
        create_distributed_compiler,
    )
except ImportError:
    DistributedCompiler = None
    DistributedCompilationConfig = None
    create_distributed_compiler = None

try:
    from .neural.neural_compiler import (
        NeuralCompiler,
        NeuralCompilationConfig,
        create_neural_compiler,
    )
except ImportError:
    NeuralCompiler = None
    NeuralCompilationConfig = None
    create_neural_compiler = None


# Unified compiler factory
def create_compiler(compiler_type: str = "core", config: dict = None):
    """
    Unified factory function to create compilers.
    
    Args:
        compiler_type: Type of compiler to create. Options:
            - "core" - CompilerCore
            - "aot" - AOTCompiler
            - "jit" - JITCompiler
            - "mlir" - MLIRCompiler
            - "runtime" - RuntimeCompiler
            - "kernel" - KernelCompiler
            - "distributed" - DistributedCompiler
            - "neural" - NeuralCompiler
            - "tf2tensorrt" - TF2TensorRTCompiler
            - "tf2xla" - TF2XLACompiler
            - "plugin" - PluginManager
        config: Optional configuration dictionary
    
    Returns:
        The requested compiler instance
    """
    if config is None:
        config = {}
    
    compiler_type = compiler_type.lower()
    
    factory_map = {
        "core": create_compiler_core,
        "aot": create_aot_compiler,
        "jit": create_jit_compiler,
        "mlir": create_mlir_compiler,
        "runtime": create_runtime_compiler,
        "kernel": create_kernel_compiler,
        "distributed": create_distributed_compiler,
        "neural": create_neural_compiler,
        "tf2tensorrt": create_tf2tensorrt_compiler,
        "tf2xla": create_tf2xla_compiler,
        "plugin": create_plugin_manager,
    }
    
    if compiler_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown compiler type: '{compiler_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[compiler_type]
    if factory is None:
        raise ImportError(f"Compiler type '{compiler_type}' is not available (module not found)")
    
    return factory(config)


# Registry of all available compilers
COMPILER_REGISTRY = {
    "core": {
        "class": CompilerCore,
        "module": "compiler.core.compiler_core",
        "description": "Core compiler infrastructure",
        "factory": create_compiler_core,
    },
    "aot": {
        "class": AOTCompiler,
        "module": "compiler.aot.aot_compiler",
        "description": "Ahead-of-time compiler",
        "factory": create_aot_compiler,
    },
    "jit": {
        "class": JITCompiler,
        "module": "compiler.jit.jit_compiler",
        "description": "Just-in-time compiler",
        "factory": create_jit_compiler,
    },
    "mlir": {
        "class": MLIRCompiler,
        "module": "compiler.mlir.mlir_compiler",
        "description": "MLIR compiler",
        "factory": create_mlir_compiler,
    },
    "runtime": {
        "class": RuntimeCompiler,
        "module": "compiler.runtime.runtime_compiler",
        "description": "Runtime compiler",
        "factory": create_runtime_compiler,
    },
    "kernel": {
        "class": KernelCompiler,
        "module": "compiler.kernels.kernel_compiler",
        "description": "Kernel compiler",
        "factory": create_kernel_compiler,
    },
    "distributed": {
        "class": DistributedCompiler,
        "module": "compiler.distributed.distributed_compiler",
        "description": "Distributed compiler",
        "factory": create_distributed_compiler,
    },
    "neural": {
        "class": NeuralCompiler,
        "module": "compiler.neural.neural_compiler",
        "description": "Neural compiler",
        "factory": create_neural_compiler,
    },
    "tf2tensorrt": {
        "class": TF2TensorRTCompiler,
        "module": "compiler.tf2tensorrt.tf2tensorrt_compiler",
        "description": "TensorFlow to TensorRT compiler",
        "factory": create_tf2tensorrt_compiler,
    },
    "tf2xla": {
        "class": TF2XLACompiler,
        "module": "compiler.tf2xla.tf2xla_compiler",
        "description": "TensorFlow to XLA compiler",
        "factory": create_tf2xla_compiler,
    },
    "plugin": {
        "class": PluginManager,
        "module": "compiler.plugin.plugin_system",
        "description": "Plugin manager",
        "factory": create_plugin_manager,
    },
}


def list_available_compilers() -> list:
    """List all available compiler types."""
    return [k for k, v in COMPILER_REGISTRY.items() if v["factory"] is not None]


def get_compiler_info(compiler_type: str) -> dict:
    """
    Get information about a specific compiler.
    
    Args:
        compiler_type: Type of compiler
    
    Returns:
        Dictionary with compiler information
    """
    if compiler_type not in COMPILER_REGISTRY:
        raise ValueError(f"Unknown compiler type: {compiler_type}")
    
    registry_entry = COMPILER_REGISTRY[compiler_type]
    
    if registry_entry["factory"] is None:
        raise ImportError(f"Compiler type '{compiler_type}' is not available (module not found)")
    
    return {
        "type": compiler_type,
        "class": registry_entry["class"].__name__ if registry_entry["class"] else None,
        "module": registry_entry["module"],
        "description": registry_entry["description"],
    }


# Update __all__
__all__ = [
    # Core compiler infrastructure
    'CompilerCore',
    'CompilationTarget',
    'OptimizationLevel',
    'CompilationResult',
    'CompilationConfig',
    'CompilationContext',
    'CompilationError',
    'create_compiler_core',
    'compilation_context',
    
    # AOT Compilation
    'AOTCompiler',
    'AOTCompilationConfig',
    'AOTOptimizationStrategy',
    'create_aot_compiler',
    'aot_compilation_context',
    
    # JIT Compilation
    'JITCompiler',
    'JITCompilationConfig',
    'JITOptimizationStrategy',
    'create_jit_compiler',
    'jit_compilation_context',
    
    # MLIR Compilation
    'MLIRCompiler',
    'MLIRDialect',
    'MLIROptimizationPass',
    'MLIRCompilationResult',
    'create_mlir_compiler',
    'mlir_compilation_context',
    
    # Plugin System
    'CompilerPlugin',
    'PluginManager',
    'PluginRegistry',
    'PluginInterface',
    'create_plugin_manager',
    'plugin_compilation_context',
    
    # TensorFlow to TensorRT
    'TF2TensorRTCompiler',
    'TensorRTConfig',
    'TensorRTOptimizationLevel',
    'create_tf2tensorrt_compiler',
    'tf2tensorrt_compilation_context',
    
    # TensorFlow to XLA
    'TF2XLACompiler',
    'XLAConfig',
    'XLAOptimizationLevel',
    'create_tf2xla_compiler',
    'tf2xla_compilation_context',
    
    # Compiler Utilities
    'CompilerUtils',
    'CodeGenerator',
    'OptimizationAnalyzer',
    'create_compiler_utils',
    'compiler_utils_context',
    
    # Runtime Compilation
    'RuntimeCompiler',
    'RuntimeCompilationConfig',
    'RuntimeOptimizationStrategy',
    'RuntimeCompilationResult',
    'RuntimeTarget',
    'RuntimeOptimizationLevel',
    'create_runtime_compiler',
    'runtime_compilation_context',
    
    # Kernel Compilation
    'KernelCompiler',
    'KernelOptimizationLevel',
    'KernelCompilationResult',
    'KernelTarget',
    'KernelConfig',
    'KernelOptimizationPass',
    'create_kernel_compiler',
    'kernel_compilation_context',
    
    # Distributed Compilation
    'DistributedCompiler',
    'DistributedCompilationConfig',
    'create_distributed_compiler',
    
    # Neural Compilation
    'NeuralCompiler',
    'NeuralCompilationConfig',
    'create_neural_compiler',
    
    # Unified factory
    'create_compiler',
    
    # Registry
    'COMPILER_REGISTRY',
    'list_available_compilers',
    'get_compiler_info',
]

__version__ = "1.0.0"

