"""
TruthGPT Utilities Examples
Comprehensive examples and usage patterns for TruthGPT utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import TruthGPT utilities
from .truthgpt_adapters import (
    TruthGPTIntegratedAdapter, TruthGPTConfig, create_truthgpt_adapter
)
from .truthgpt_optimization_utils import (
    TruthGPTIntegratedOptimizer, TruthGPTOptimizationConfig, create_truthgpt_optimizer
)
from .truthgpt_monitoring import (
    TruthGPTMonitor, TruthGPTAnalytics, TruthGPTDashboard, create_truthgpt_monitoring_suite
)
from .truthgpt_integration import (
    TruthGPTIntegrationManager, TruthGPTIntegrationConfig, TruthGPTQuickSetup,
    create_truthgpt_integration, quick_truthgpt_integration
)

logger = logging.getLogger(__name__)

class TruthGPTExampleModel(nn.Module):
    """Example TruthGPT model for demonstration."""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 50, output_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Model layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output

def example_basic_usage():
    """Example: Basic TruthGPT utilities usage."""
    print("🚀 TruthGPT Basic Usage Example")
    print("=" * 50)
    
    # Create a simple model
    model = TruthGPTExampleModel(input_size=100, hidden_size=50, output_size=10)
    print(f"✅ Created TruthGPT model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create configuration
    config = TruthGPTConfig(
        model_name="TruthGPT-Example",
        optimization_level="balanced",
        max_memory_gb=4.0,
        target_latency_ms=50.0
    )
    
    # Create adapter
    adapter = create_truthgpt_adapter(config)
    
    # Perform adaptation
    results = adapter.full_adaptation(model)
    print(f"✅ Adaptation completed: {len(results['adaptations'])} components adapted")
    
    # Print summary
    if 'summary' in results:
        summary = results['summary']
        print(f"📊 Summary: {summary['successful_adaptations']}/{summary['total_adaptations']} successful")
    
    return results

def example_optimization_usage():
    """Example: TruthGPT optimization usage."""
    print("\n🔧 TruthGPT Optimization Example")
    print("=" * 50)
    
    # Create model
    model = TruthGPTExampleModel()
    
    # Create optimization configuration
    config = TruthGPTOptimizationConfig(
        model_name="TruthGPT-Optimized",
        optimization_level="aggressive",
        target_latency_ms=25.0,
        target_memory_gb=2.0,
        enable_quantization=True,
        enable_pruning=True
    )
    
    # Create optimizer
    optimizer = create_truthgpt_optimizer(config)
    
    # Optimize model
    results = optimizer.optimize_model(model)
    print(f"✅ Optimization completed: {len(results['optimizations'])} optimizations applied")
    
    # Print optimization results
    for opt_name, opt_result in results['optimizations'].items():
        status = opt_result.get('status', 'unknown')
        print(f"  {opt_name}: {status}")
    
    return results

def example_monitoring_usage():
    """Example: TruthGPT monitoring usage."""
    print("\n📊 TruthGPT Monitoring Example")
    print("=" * 50)
    
    # Create monitoring suite
    monitor, analytics, dashboard = create_truthgpt_monitoring_suite("TruthGPT-Monitored")
    
    # Start monitoring
    monitor.start_monitoring()
    print("✅ Monitoring started")
    
    # Simulate some model operations
    model = TruthGPTExampleModel()
    model.eval()
    
    # Simulate inference
    for i in range(5):
        start_time = time.time()
        
        # Simulate inference
        with torch.no_grad():
            dummy_input = torch.randn(1, 100)
            _ = model(dummy_input)
        
        inference_time = time.time() - start_time
        monitor.record_inference(inference_time)
        
        # Add to analytics
        current_metrics = monitor.get_current_metrics()
        if current_metrics:
            analytics.add_performance_data(current_metrics)
        
        print(f"  Inference {i+1}: {inference_time*1000:.2f}ms")
        time.sleep(0.1)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"📈 Performance Summary:")
    print(f"  Average inference time: {summary.get('avg_inference_time', 0):.2f}ms")
    print(f"  Average throughput: {summary.get('avg_throughput', 0):.2f} samples/sec")
    
    # Generate analytics
    performance_analysis = analytics.analyze_performance_trends()
    print(f"📊 Performance Analysis:")
    if 'inference_time' in performance_analysis:
        inference_stats = performance_analysis['inference_time']
        print(f"  Mean inference time: {inference_stats['mean']:.2f}ms")
        print(f"  Std inference time: {inference_stats['std']:.2f}ms")
    
    # Generate insights
    insights = analytics.generate_insights()
    print(f"💡 Insights:")
    for insight in insights:
        print(f"  {insight}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("✅ Monitoring stopped")
    
    return {
        'monitor': monitor,
        'analytics': analytics,
        'dashboard': dashboard,
        'summary': summary,
        'insights': insights
    }

def example_integration_usage():
    """Example: Complete TruthGPT integration."""
    print("\n🔗 TruthGPT Integration Example")
    print("=" * 50)
    
    # Create model
    model = TruthGPTExampleModel()
    
    # Create integration configuration
    config = TruthGPTQuickSetup.create_balanced_config("TruthGPT-Integrated")
    
    # Create integration manager
    integration_manager = create_truthgpt_integration(config)
    
    # Perform full integration
    results = integration_manager.full_integration(model)
    print(f"✅ Integration completed: {len(results['integration_steps'])} steps")
    
    # Print integration summary
    if 'summary' in results:
        summary = results['summary']
        print(f"📊 Integration Summary:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Successful: {summary['successful_steps']}")
        print(f"  Failed: {summary['failed_steps']}")
        print(f"  Overall success: {summary['overall_success']}")
    
    # Export report
    report_path = "truthgpt_integration_report.json"
    integration_manager.export_integration_report(report_path)
    print(f"📄 Report exported to: {report_path}")
    
    # Create dashboard if available
    if integration_manager.dashboard:
        try:
            dashboard = integration_manager.dashboard
            dashboard.create_performance_plots("truthgpt_dashboard.png")
            dashboard.generate_report("truthgpt_dashboard_report.json")
            print("📊 Dashboard created with plots and report")
        except Exception as e:
            print(f"⚠️ Dashboard creation failed: {e}")
    
    # Stop monitoring
    integration_manager.stop_monitoring()
    
    return results

def example_advanced_usage():
    """Example: Advanced TruthGPT usage patterns."""
    print("\n⚡ TruthGPT Advanced Usage Example")
    print("=" * 50)
    
    # Create multiple models for comparison
    models = {
        'small': TruthGPTExampleModel(input_size=50, hidden_size=25, output_size=5),
        'medium': TruthGPTExampleModel(input_size=100, hidden_size=50, output_size=10),
        'large': TruthGPTExampleModel(input_size=200, hidden_size=100, output_size=20)
    }
    
    # Create different configurations
    configs = {
        'conservative': TruthGPTQuickSetup.create_conservative_config(),
        'balanced': TruthGPTQuickSetup.create_default_config(),
        'aggressive': TruthGPTQuickSetup.create_aggressive_config()
    }
    
    results = {}
    
    # Test each model with each configuration
    for model_name, model in models.items():
        print(f"\n🧪 Testing {model_name} model:")
        results[model_name] = {}
        
        for config_name, config in configs.items():
            print(f"  Configuration: {config_name}")
            
            # Create integration manager
            config.model_name = f"TruthGPT-{model_name}-{config_name}"
            integration_manager = create_truthgpt_integration(config)
            
            # Perform integration
            try:
                integration_results = integration_manager.full_integration(model)
                results[model_name][config_name] = {
                    'success': True,
                    'summary': integration_results.get('summary', {}),
                    'steps': len(integration_results.get('integration_steps', {}))
                }
                print(f"    ✅ Success: {integration_results.get('summary', {}).get('successful_steps', 0)} steps")
            except Exception as e:
                results[model_name][config_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"    ❌ Failed: {e}")
            
            # Cleanup
            integration_manager.stop_monitoring()
    
    # Print comparison summary
    print(f"\n📊 Comparison Summary:")
    for model_name, model_results in results.items():
        print(f"  {model_name} model:")
        for config_name, config_result in model_results.items():
            if config_result['success']:
                steps = config_result.get('steps', 0)
                print(f"    {config_name}: ✅ {steps} steps")
            else:
                print(f"    {config_name}: ❌ Failed")
    
    return results

def example_custom_adapters():
    """Example: Creating custom TruthGPT adapters."""
    print("\n🛠️ Custom TruthGPT Adapters Example")
    print("=" * 50)
    
    from .truthgpt_adapters import TruthGPTAdapter, TruthGPTConfig
    
    class CustomTruthGPTAdapter(TruthGPTAdapter):
        """Custom adapter for specific use case."""
        
        def __init__(self, config: TruthGPTConfig, custom_param: str = "default"):
            super().__init__(config)
            self.custom_param = custom_param
            self.logger.info(f"Custom adapter initialized with param: {custom_param}")
        
        def adapt(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
            """Custom adaptation logic."""
            self.logger.info("Running custom adaptation")
            
            # Custom analysis
            custom_analysis = {
                'model_type': type(model).__name__,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'custom_param': self.custom_param,
                'timestamp': time.time()
            }
            
            # Custom optimization
            if hasattr(model, 'custom_optimization'):
                model.custom_optimization()
                custom_analysis['custom_optimization_applied'] = True
            else:
                custom_analysis['custom_optimization_applied'] = False
            
            self.log_metrics("custom_adaptation", **custom_analysis)
            return custom_analysis
    
    # Use custom adapter
    config = TruthGPTConfig(model_name="Custom-TruthGPT")
    custom_adapter = CustomTruthGPTAdapter(config, custom_param="special_value")
    
    model = TruthGPTExampleModel()
    results = custom_adapter.adapt(model)
    
    print(f"✅ Custom adaptation completed:")
    print(f"  Model type: {results['model_type']}")
    print(f"  Parameter count: {results['parameter_count']}")
    print(f"  Custom param: {results['custom_param']}")
    print(f"  Custom optimization: {results['custom_optimization_applied']}")
    
    return results

def run_all_examples():
    """Run all TruthGPT examples."""
    print("🚀 TruthGPT Utilities - Complete Examples Suite")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Optimization", example_optimization_usage),
        ("Monitoring", example_monitoring_usage),
        ("Integration", example_integration_usage),
        ("Advanced Usage", example_advanced_usage),
        ("Custom Adapters", example_custom_adapters)
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        try:
            print(f"\n{'='*20} {example_name} {'='*20}")
            result = example_func()
            results[example_name] = {'success': True, 'result': result}
            print(f"✅ {example_name} completed successfully")
        except Exception as e:
            print(f"❌ {example_name} failed: {e}")
            results[example_name] = {'success': False, 'error': str(e)}
    
    # Print final summary
    print(f"\n{'='*60}")
    print("📊 Final Summary:")
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"  Successful examples: {successful}/{total}")
    
    for example_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {example_name}")
    
    return results

if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    # Save results
    with open("truthgpt_examples_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: truthgpt_examples_results.json")
    print("🎉 TruthGPT examples completed!")






