"""
Advanced examples for optimization_core.

Demonstrates advanced usage patterns including plugins, observability, and optimization.
"""
from utils import (
    get_tracer,
    get_metrics_exporter,
    HyperparameterOptimizer,
    register_plugin,
    PluginInfo,
    BasePlugin,
)

# Example 1: Using Tracer
def example_tracing():
    """Example of using distributed tracing."""
    tracer = get_tracer()
    
    with tracer.span("model_inference", tags={"model": "mistral-7b"}):
        with tracer.span("tokenization"):
            # Tokenize
            pass
        
        with tracer.span("generation"):
            # Generate
            pass
    
    trace = tracer.get_trace()
    print(f"Trace: {trace}")


# Example 2: Using Metrics Exporter
def example_metrics():
    """Example of using metrics exporter."""
    exporter = get_metrics_exporter()
    
    # Record metrics
    for i in range(100):
        exporter.record_metric(
            "inference_latency",
            value=0.05 + np.random.normal(0, 0.01),
            tags={"model": "mistral-7b"}
        )
    
    # Get summary
    summary = exporter.get_metric_summary("inference_latency")
    print(f"Average latency: {summary['avg']:.3f}s")


# Example 3: Hyperparameter Optimization
def example_hyperparameter_optimization():
    """Example of hyperparameter optimization."""
    def objective(params):
        # Your objective function
        return params["lr"] ** 2 + params["dropout"] ** 2
    
    param_space = {
        "lr": (0.0001, 0.01),
        "dropout": (0.0, 0.5),
    }
    
    optimizer = HyperparameterOptimizer(
        param_space=param_space,
        objective_func=objective,
        method="random"
    )
    
    result = optimizer.optimize(n_iterations=50)
    print(f"Best params: {result.best_params}")
    print(f"Best score: {result.best_score}")


# Example 4: Creating a Plugin
class MyCustomPlugin(BasePlugin):
    """Custom plugin example."""
    name = "my_custom_plugin"
    version = "1.0.0"
    description = "Custom plugin for optimization"
    
    def execute(self, data, **kwargs):
        """Execute plugin."""
        # Your custom logic
        return processed_data


def example_plugin():
    """Example of using plugins."""
    # Register plugin
    plugin_info = PluginInfo(
        name="my_plugin",
        version="1.0.0",
        description="My custom plugin"
    )
    register_plugin(plugin_info, MyCustomPlugin())
    
    # Use plugin
    from utils import get_plugin
    plugin = get_plugin("my_plugin")
    result = plugin.execute(data)













