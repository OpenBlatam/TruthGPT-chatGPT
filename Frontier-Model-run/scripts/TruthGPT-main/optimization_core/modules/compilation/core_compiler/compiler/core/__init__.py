"""
Core Compiler Infrastructure for TruthGPT
Base classes and interfaces for all compiler components
"""

from .compiler_core import (
    CompilerCore, CompilationTarget, OptimizationLevel, CompilationResult,
    CompilationConfig, CompilationContext, CompilationError,
    create_compiler_core, compilation_context
)

from .compilation_pipeline import (
    CompilationPipeline, PipelineStage, PipelineResult,
    create_compilation_pipeline, pipeline_context
)

from .optimization_engine import (
    OptimizationEngine, OptimizationPass, OptimizationResult,
    create_optimization_engine, optimization_context
)

__all__ = [
    'CompilerCore',
    'CompilationTarget',
    'OptimizationLevel',
    'CompilationResult',
    'CompilationConfig',
    'CompilationContext',
    'CompilationError',
    'create_compiler_core',
    'compilation_context',
    'CompilationPipeline',
    'PipelineStage',
    'PipelineResult',
    'create_compilation_pipeline',
    'pipeline_context',
    'OptimizationEngine',
    'OptimizationPass',
    'OptimizationResult',
    'create_optimization_engine',
    'optimization_context'
]






