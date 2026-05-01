"""
Neural Compiler Module for TruthGPT
Advanced neural-guided compilation with machine learning optimization
"""

from .neural_compiler import (
    NeuralCompiler, NeuralCompilationConfig, NeuralCompilationResult,
    NeuralCompilationMode, NeuralOptimizationStrategy, NeuralCompilationTarget,
    create_neural_compiler, neural_compilation_context
)

__all__ = [
    'NeuralCompiler',
    'NeuralCompilationConfig',
    'NeuralCompilationResult',
    'NeuralCompilationMode',
    'NeuralOptimizationStrategy',
    'NeuralCompilationTarget',
    'create_neural_compiler',
    'neural_compilation_context'
]

__version__ = "1.0.0"



