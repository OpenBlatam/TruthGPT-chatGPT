"""
Runtime Compilation for TruthGPT Compiler
Runtime compilation and adaptive optimization
"""

from .runtime_compiler import (
    RuntimeCompiler, RuntimeCompilationConfig, RuntimeOptimizationStrategy,
    RuntimeCompilationResult, RuntimeTarget, RuntimeOptimizationLevel,
    create_runtime_compiler, runtime_compilation_context
)

__all__ = [
    'RuntimeCompiler',
    'RuntimeCompilationConfig',
    'RuntimeOptimizationStrategy',
    'RuntimeCompilationResult',
    'RuntimeTarget',
    'RuntimeOptimizationLevel',
    'create_runtime_compiler',
    'runtime_compilation_context',
]






