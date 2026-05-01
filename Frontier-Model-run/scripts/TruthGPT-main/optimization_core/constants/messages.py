"""
Message constants for TruthGPT Optimization Core
"""

# Error messages
ERROR_MESSAGES = {
    'optimization_failed': 'Optimization failed',
    'timeout': 'Optimization timed out',
    'memory_error': 'Insufficient memory for optimization',
    'compute_error': 'Compute error during optimization',
    'convergence_error': 'Optimization did not converge',
    'accuracy_error': 'Accuracy dropped below threshold',
    'invalid_config': 'Invalid optimization configuration',
    'unsupported_framework': 'Unsupported optimization framework',
    'unsupported_technique': 'Unsupported optimization technique',
    'unsupported_level': 'Unsupported optimization level',
    'model_incompatible': 'Model incompatible with optimization',
    'hardware_incompatible': 'Hardware incompatible with optimization',
    'software_incompatible': 'Software incompatible with optimization'
}

# Success messages
SUCCESS_MESSAGES = {
    'optimization_success': 'Optimization completed successfully',
    'speed_improved': 'Speed improved significantly',
    'memory_reduced': 'Memory usage reduced significantly',
    'energy_saved': 'Energy consumption reduced significantly',
    'accuracy_maintained': 'Accuracy maintained within acceptable range',
    'performance_enhanced': 'Overall performance enhanced',
    'optimization_optimal': 'Optimization reached optimal configuration',
    'benchmark_passed': 'All benchmarks passed successfully',
    'deployment_ready': 'Model ready for deployment',
    'production_ready': 'Model ready for production use'
}

# Warning messages
WARNING_MESSAGES = {
    'accuracy_degraded': 'Accuracy may have degraded slightly',
    'memory_increased': 'Memory usage may have increased',
    'energy_increased': 'Energy consumption may have increased',
    'speed_decreased': 'Speed may have decreased slightly',
    'convergence_slow': 'Optimization convergence is slow',
    'resource_usage_high': 'Resource usage is higher than expected',
    'compatibility_issues': 'Potential compatibility issues detected',
    'performance_variable': 'Performance may vary across different hardware',
    'optimization_partial': 'Optimization only partially successful',
    'benchmark_marginal': 'Benchmark results are marginal'
}

# Info messages
INFO_MESSAGES = {
    'optimization_started': 'Optimization started',
    'optimization_progress': 'Optimization in progress',
    'optimization_completed': 'Optimization completed',
    'benchmark_started': 'Benchmarking started',
    'benchmark_completed': 'Benchmarking completed',
    'deployment_started': 'Deployment started',
    'deployment_completed': 'Deployment completed',
    'testing_started': 'Testing started',
    'testing_completed': 'Testing completed',
    'validation_started': 'Validation started',
    'validation_completed': 'Validation completed'
}

# Debug messages
DEBUG_MESSAGES = {
    'optimization_debug': 'Optimization debug information',
    'performance_debug': 'Performance debug information',
    'memory_debug': 'Memory debug information',
    'energy_debug': 'Energy debug information',
    'accuracy_debug': 'Accuracy debug information',
    'speed_debug': 'Speed debug information',
    'technique_debug': 'Technique debug information',
    'framework_debug': 'Framework debug information',
    'level_debug': 'Level debug information',
    'config_debug': 'Configuration debug information'
}

__all__ = [
    'ERROR_MESSAGES',
    'SUCCESS_MESSAGES',
    'WARNING_MESSAGES',
    'INFO_MESSAGES',
    'DEBUG_MESSAGES',
]


