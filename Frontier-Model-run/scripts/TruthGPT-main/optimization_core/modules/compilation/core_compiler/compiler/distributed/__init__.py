"""
Distributed Compiler Module for TruthGPT
Advanced distributed compilation with multi-node optimization and load balancing
"""

from .distributed_compiler import (
    DistributedCompiler, DistributedCompilationConfig, DistributedCompilationResult,
    DistributedCompilationMode, LoadBalancingStrategy, DistributedCompilationTarget,
    create_distributed_compiler, distributed_compilation_context
)

__all__ = [
    'DistributedCompiler',
    'DistributedCompilationConfig',
    'DistributedCompilationResult',
    'DistributedCompilationMode',
    'LoadBalancingStrategy',
    'DistributedCompilationTarget',
    'create_distributed_compiler',
    'distributed_compilation_context'
]

__version__ = "1.0.0"



