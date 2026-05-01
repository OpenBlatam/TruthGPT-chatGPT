"""
Test Suite for TruthGPT Compiler
Comprehensive testing framework for all compiler components
"""

from .test_compiler_core import (
    TestCompilerCore, TestCompilationConfig, TestCompilationResult,
    test_compiler_core, test_compilation_context
)

from .test_aot_compiler import (
    TestAOTCompiler, TestAOTCompilation, TestAOTOptimization,
    test_aot_compiler, test_aot_compilation_context
)

from .test_jit_compiler import (
    TestJITCompiler, TestJITCompilation, TestJITOptimization,
    test_jit_compiler, test_jit_compilation_context
)

from .test_mlir_compiler import (
    TestMLIRCompiler, TestMLIRCompilation, TestMLIROptimization,
    test_mlir_compiler, test_mlir_compilation_context
)

from .test_plugin_system import (
    TestPluginSystem, TestPluginManager, TestPluginRegistry,
    test_plugin_system, test_plugin_context
)

from .test_tf2tensorrt import (
    TestTF2TensorRT, TestTensorRTCompilation, TestTensorRTOptimization,
    test_tf2tensorrt, test_tensorrt_context
)

from .test_tf2xla import (
    TestTF2XLA, TestXLACompilation, TestXLAOptimization,
    test_tf2xla, test_xla_context
)

from .test_runner import (
    TestRunner, TestSuite, TestResult, TestConfig,
    create_test_runner, run_all_tests
)

__all__ = [
    'TestCompilerCore',
    'TestCompilationConfig',
    'TestCompilationResult',
    'test_compiler_core',
    'test_compilation_context',
    'TestAOTCompiler',
    'TestAOTCompilation',
    'TestAOTOptimization',
    'test_aot_compiler',
    'test_aot_compilation_context',
    'TestJITCompiler',
    'TestJITCompilation',
    'TestJITOptimization',
    'test_jit_compiler',
    'test_jit_compilation_context',
    'TestMLIRCompiler',
    'TestMLIRCompilation',
    'TestMLIROptimization',
    'test_mlir_compiler',
    'test_mlir_compilation_context',
    'TestPluginSystem',
    'TestPluginManager',
    'TestPluginRegistry',
    'test_plugin_system',
    'test_plugin_context',
    'TestTF2TensorRT',
    'TestTensorRTCompilation',
    'TestTensorRTOptimization',
    'test_tf2tensorrt',
    'test_tensorrt_context',
    'TestTF2XLA',
    'TestXLACompilation',
    'TestXLAOptimization',
    'test_tf2xla',
    'test_xla_context',
    'TestRunner',
    'TestSuite',
    'TestResult',
    'TestConfig',
    'create_test_runner',
    'run_all_tests'
]






