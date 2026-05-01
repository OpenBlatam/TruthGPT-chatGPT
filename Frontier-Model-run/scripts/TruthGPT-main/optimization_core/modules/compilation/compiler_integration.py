"""
TruthGPT Compiler Integration Module
Bridges the compiler infrastructure with existing TruthGPT optimizers
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import numpy as np

from .core_compiler.compiler import (
    CompilerCore, CompilationConfig, CompilationResult, CompilationTarget, OptimizationLevel,
    create_compiler_core, AOTCompiler, JITCompiler, MLIRCompiler, TF2TensorRTCompiler, TF2XLACompiler,
    RuntimeCompiler, KernelCompiler, create_aot_compiler, create_jit_compiler, create_mlir_compiler,
    create_tf2tensorrt_compiler, create_tf2xla_compiler, create_runtime_compiler, create_kernel_compiler
)

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTCompilationConfig:
    """Configuration for TruthGPT compilation integration"""
    # Compiler selection
    primary_compiler: str = "aot"  # aot, jit, mlir, tensorrt, xla, runtime, kernel
    fallback_compilers: List[str] = None
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.EXTREME
    target_platform: CompilationTarget = CompilationTarget.GPU
    
    # TruthGPT-specific settings
    enable_truthgpt_optimizations: bool = True
    enable_quantum_optimizations: bool = False
    enable_neural_architecture_search: bool = True
    enable_meta_learning: bool = True
    
    # Performance settings
    enable_profiling: bool = True
    enable_benchmarking: bool = True
    enable_caching: bool = True
    
    # Integration settings
    auto_select_compiler: bool = True
    enable_compiler_fusion: bool = True
    enable_adaptive_compilation: bool = True

    def __post_init__(self):
        if self.fallback_compilers is None:
            self.fallback_compilers = ["jit", "mlir", "runtime"]

@dataclass
class TruthGPTCompilationResult:
    """Result of TruthGPT compilation integration"""
    success: bool
    primary_compiler_used: str
    compilation_results: Dict[str, CompilationResult] = None
    performance_metrics: Dict[str, float] = None
    optimization_report: Dict[str, Any] = None
    integration_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.compilation_results is None:
            self.compilation_results = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.optimization_report is None:
            self.optimization_report = {}
        if self.integration_metadata is None:
            self.integration_metadata = {}

class TruthGPTCompilerIntegration:
    """Main integration class for TruthGPT compiler infrastructure"""
    
    def __init__(self, config: TruthGPTCompilationConfig):
        self.config = config
        self.compilers = {}
        self.performance_history = {}
        self.optimization_cache = {}
        self._initialize_compilers()
        
    def _initialize_compilers(self):
        """Initialize all available compilers"""
        logger.info("Initializing TruthGPT compiler integration")
        
        # Core compilation config
        core_config = CompilationConfig(
            target=self.config.target_platform,
            optimization_level=self.config.optimization_level,
            enable_quantization=True,
            enable_fusion=True,
            enable_parallelization=True
        )
        
        # Initialize AOT compiler
        try:
            from .core_compiler.compiler.aot import AOTCompilationConfig, AOTTarget, AOTOptimizationLevel
            aot_config = AOTCompilationConfig(
                target=AOTTarget.CUDA if self.config.target_platform == CompilationTarget.GPU else AOTTarget.NATIVE,
                optimization_level=AOTOptimizationLevel.EXTREME,
                enable_inlining=True,
                enable_vectorization=True,
                enable_loop_unrolling=True
            )
            self.compilers["aot"] = create_aot_compiler(aot_config)
            logger.info("AOT compiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize AOT compiler: {e}")
        
        # Initialize JIT compiler
        try:
            from .core_compiler.compiler.jit import JITCompilationConfig, JITTarget, JITOptimizationLevel
            jit_config = JITCompilationConfig(
                target=JITTarget.NATIVE,
                optimization_level=JITOptimizationLevel.ADAPTIVE,
                enable_profiling=True,
                enable_hotspot_detection=True
            )
            self.compilers["jit"] = create_jit_compiler(jit_config)
            logger.info("JIT compiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize JIT compiler: {e}")
        
        # Initialize MLIR compiler
        try:
            self.compilers["mlir"] = create_mlir_compiler(core_config)
            logger.info("MLIR compiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MLIR compiler: {e}")
        
        # Initialize TensorRT compiler
        try:
            from .core_compiler.compiler.tf2tensorrt import TensorRTConfig, TensorRTOptimizationLevel, TensorRTPrecision
            tensorrt_config = TensorRTConfig(
                optimization_level=TensorRTOptimizationLevel.AGGRESSIVE,
                precision=TensorRTPrecision.FP16,
                enable_fp16=True,
                max_batch_size=32
            )
            self.compilers["tensorrt"] = create_tf2tensorrt_compiler(tensorrt_config)
            logger.info("TensorRT compiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorRT compiler: {e}")
        
        # Initialize XLA compiler
        try:
            from .core_compiler.compiler.tf2xla import XLAConfig, XLAOptimizationLevel, XLATarget
            xla_config = XLAConfig(
                target=XLATarget.GPU if self.config.target_platform == CompilationTarget.GPU else XLATarget.CPU,
                optimization_level=XLAOptimizationLevel.AGGRESSIVE,
                enable_fusion=True,
                enable_autotuning=True
            )
            self.compilers["xla"] = create_tf2xla_compiler(xla_config)
            logger.info("XLA compiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize XLA compiler: {e}")
        
        # Initialize Runtime compiler
        try:
            from .core_compiler.compiler.runtime import RuntimeCompilationConfig, RuntimeTarget, RuntimeOptimizationLevel
            runtime_config = RuntimeCompilationConfig(
                target=RuntimeTarget.NATIVE,
                optimization_level=RuntimeOptimizationLevel.ADAPTIVE,
                enable_profiling=True,
                enable_adaptive_optimization=True
            )
            self.compilers["runtime"] = create_runtime_compiler(runtime_config)
            logger.info("Runtime compiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Runtime compiler: {e}")
        
        # Initialize Kernel compiler
        try:
            from .core_compiler.compiler.kernels import KernelConfig, KernelTarget, KernelOptimizationLevel
            kernel_config = KernelConfig(
                target=KernelTarget.CUDA if self.config.target_platform == CompilationTarget.GPU else KernelTarget.CPU,
                optimization_level=KernelOptimizationLevel.AGGRESSIVE,
                enable_fusion=True,
                enable_vectorization=True
            )
            self.compilers["kernel"] = create_kernel_compiler(kernel_config)
            logger.info("Kernel compiler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kernel compiler: {e}")
        
        logger.info(f"Initialized {len(self.compilers)} compilers: {list(self.compilers.keys())}")
    
    def compile_truthgpt_model(self, model: Any, optimizer: Any = None, input_spec: Optional[Dict] = None) -> TruthGPTCompilationResult:
        """Compile TruthGPT model with integrated compiler infrastructure"""
        try:
            start_time = time.time()
            
            # Apply TruthGPT optimizations first if optimizer is provided
            if optimizer and self.config.enable_truthgpt_optimizations:
                logger.info("Applying TruthGPT optimizations")
                model = self._apply_truthgpt_optimizations(model, optimizer)
            
            # Select best compiler
            selected_compiler = self._select_compiler(model, input_spec)
            
            # Compile with selected compiler
            compilation_results = {}
            primary_result = None
            
            if selected_compiler in self.compilers:
                logger.info(f"Compiling with {selected_compiler} compiler")
                compiler = self.compilers[selected_compiler]
                result = compiler.compile(model, input_spec)
                compilation_results[selected_compiler] = result
                primary_result = result
                
                # If primary compilation fails, try fallback compilers
                if not result.success and self.config.fallback_compilers:
                    for fallback_compiler in self.config.fallback_compilers:
                        if fallback_compiler in self.compilers and fallback_compiler != selected_compiler:
                            logger.info(f"Trying fallback compiler: {fallback_compiler}")
                            fallback_result = self.compilers[fallback_compiler].compile(model, input_spec)
                            compilation_results[fallback_compiler] = fallback_result
                            
                            if fallback_result.success:
                                primary_result = fallback_result
                                selected_compiler = fallback_compiler
                                break
            
            # Generate performance metrics
            performance_metrics = self._generate_performance_metrics(compilation_results, time.time() - start_time)
            
            # Generate optimization report
            optimization_report = self._generate_optimization_report(model, compilation_results)
            
            # Update performance history
            self._update_performance_history(selected_compiler, performance_metrics)
            
            return TruthGPTCompilationResult(
                success=primary_result.success if primary_result else False,
                primary_compiler_used=selected_compiler,
                compilation_results=compilation_results,
                performance_metrics=performance_metrics,
                optimization_report=optimization_report,
                integration_metadata={
                    "compilation_time": time.time() - start_time,
                    "compilers_available": list(self.compilers.keys()),
                    "truthgpt_optimizations_applied": optimizer is not None
                }
            )
            
        except Exception as e:
            logger.error(f"TruthGPT compilation integration failed: {str(e)}")
            return TruthGPTCompilationResult(
                success=False,
                primary_compiler_used="none",
                integration_metadata={"error": str(e)}
            )
    
    def _apply_truthgpt_optimizations(self, model: Any, optimizer: Any) -> Any:
        """Apply TruthGPT-specific optimizations"""
        try:
            # Apply the optimizer's optimization methods
            if hasattr(optimizer, 'optimize'):
                optimized_model = optimizer.optimize(model)
                logger.info("Applied TruthGPT optimizer")
                return optimized_model
            elif hasattr(optimizer, 'enhance_model'):
                enhanced_model = optimizer.enhance_model(model)
                logger.info("Applied TruthGPT model enhancement")
                return enhanced_model
            else:
                logger.warning("Optimizer does not have standard optimization methods")
                return model
        except Exception as e:
            logger.warning(f"Failed to apply TruthGPT optimizations: {e}")
            return model
    
    def _select_compiler(self, model: Any, input_spec: Optional[Dict] = None) -> str:
        """Select the best compiler for the model"""
        if not self.config.auto_select_compiler:
            return self.config.primary_compiler
        
        # Simple heuristic-based selection
        model_size = self._estimate_model_size(model)
        target_platform = self.config.target_platform
        
        if target_platform == CompilationTarget.GPU:
            if model_size > 1000000:  # Large model
                return "tensorrt" if "tensorrt" in self.compilers else "aot"
            else:
                return "jit" if "jit" in self.compilers else "runtime"
        else:
            if model_size > 1000000:  # Large model
                return "aot" if "aot" in self.compilers else "mlir"
            else:
                return "jit" if "jit" in self.compilers else "runtime"
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate model size for compiler selection"""
        try:
            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
            else:
                return 100000  # Default estimate
        except:
            return 100000
    
    def _generate_performance_metrics(self, compilation_results: Dict[str, CompilationResult], total_time: float) -> Dict[str, float]:
        """Generate performance metrics from compilation results"""
        metrics = {
            "total_compilation_time": total_time,
            "compilers_used": len(compilation_results),
            "successful_compilations": sum(1 for result in compilation_results.values() if result.success)
        }
        
        for compiler_name, result in compilation_results.items():
            if result.success:
                metrics[f"{compiler_name}_compilation_time"] = result.compilation_time
                if hasattr(result, 'optimization_metrics'):
                    for key, value in result.optimization_metrics.items():
                        metrics[f"{compiler_name}_{key}"] = value
        
        return metrics
    
    def _generate_optimization_report(self, model: Any, compilation_results: Dict[str, CompilationResult]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            "model_info": {
                "model_type": type(model).__name__,
                "estimated_size": self._estimate_model_size(model)
            },
            "compilation_summary": {
                "total_compilers": len(compilation_results),
                "successful_compilers": sum(1 for result in compilation_results.values() if result.success),
                "best_compiler": self._get_best_compiler(compilation_results)
            },
            "compiler_results": {}
        }
        
        for compiler_name, result in compilation_results.items():
            report["compiler_results"][compiler_name] = {
                "success": result.success,
                "compilation_time": result.compilation_time,
                "optimization_metrics": getattr(result, 'optimization_metrics', {}),
                "errors": result.errors if hasattr(result, 'errors') else []
            }
        
        return report
    
    def _get_best_compiler(self, compilation_results: Dict[str, CompilationResult]) -> str:
        """Determine the best compiler based on results"""
        successful_results = {name: result for name, result in compilation_results.items() if result.success}
        
        if not successful_results:
            return "none"
        
        # Select based on compilation time (faster is better)
        best_compiler = min(successful_results.items(), key=lambda x: x[1].compilation_time)
        return best_compiler[0]
    
    def _update_performance_history(self, compiler_name: str, metrics: Dict[str, float]):
        """Update performance history for compiler selection"""
        if compiler_name not in self.performance_history:
            self.performance_history[compiler_name] = []
        
        self.performance_history[compiler_name].append({
            "timestamp": time.time(),
            "metrics": metrics
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history[compiler_name]) > 100:
            self.performance_history[compiler_name] = self.performance_history[compiler_name][-100:]
    
    def get_compiler_statistics(self) -> Dict[str, Any]:
        """Get statistics about compiler performance"""
        stats = {
            "available_compilers": list(self.compilers.keys()),
            "performance_history": {},
            "total_compilations": 0
        }
        
        for compiler_name, history in self.performance_history.items():
            if history:
                stats["performance_history"][compiler_name] = {
                    "total_compilations": len(history),
                    "avg_compilation_time": np.mean([entry["metrics"].get("total_compilation_time", 0) for entry in history]),
                    "success_rate": np.mean([1 if entry["metrics"].get("successful_compilations", 0) > 0 else 0 for entry in history])
                }
                stats["total_compilations"] += len(history)
        
        return stats
    
    def benchmark_compilers(self, model: Any, input_spec: Optional[Dict] = None, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark all available compilers"""
        logger.info(f"Benchmarking {len(self.compilers)} compilers with {iterations} iterations each")
        
        benchmark_results = {}
        
        for compiler_name, compiler in self.compilers.items():
            logger.info(f"Benchmarking {compiler_name} compiler")
            
            times = []
            successes = 0
            
            for i in range(iterations):
                start_time = time.time()
                try:
                    result = compiler.compile(model, input_spec)
                    compilation_time = time.time() - start_time
                    times.append(compilation_time)
                    
                    if result.success:
                        successes += 1
                        
                except Exception as e:
                    logger.warning(f"Benchmark iteration {i+1} failed for {compiler_name}: {e}")
                    times.append(float('inf'))
            
            if times:
                benchmark_results[compiler_name] = {
                    "avg_time": np.mean([t for t in times if t != float('inf')]),
                    "min_time": np.min([t for t in times if t != float('inf')]),
                    "max_time": np.max([t for t in times if t != float('inf')]),
                    "success_rate": successes / iterations,
                    "total_iterations": iterations
                }
        
        return benchmark_results

def create_truthgpt_compiler_integration(config: TruthGPTCompilationConfig) -> TruthGPTCompilerIntegration:
    """Create a TruthGPT compiler integration instance"""
    return TruthGPTCompilerIntegration(config)

def truthgpt_compilation_context(config: TruthGPTCompilationConfig):
    """Create a TruthGPT compilation context"""
    class TruthGPTCompilationContext:
        def __init__(self, cfg: TruthGPTCompilationConfig):
            self.config = cfg
            self.integration = None
            
        def __enter__(self):
            self.integration = create_truthgpt_compiler_integration(self.config)
            logger.info("TruthGPT compiler integration context started")
            return self.integration
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            logger.info("TruthGPT compiler integration context ended")
    
    return TruthGPTCompilationContext(config)






