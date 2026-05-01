"""
Integration tests for advanced workflows
Tests complex optimization workflows and system integration
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.mock_components import MockModel, MockOptimizer, MockAttention, MockMLP, MockKVCache, MockDataset
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, MemoryTracker, TestAssertions

class TestAdvancedOptimizationWorkflows(unittest.TestCase):
    """Test suite for advanced optimization workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        self.memory_tracker = MemoryTracker()
        
    def test_multi_stage_optimization_workflow(self):
        """Test multi-stage optimization workflow"""
        class MultiStageOptimizer:
            def __init__(self):
                self.stages = []
                self.optimization_history = []
                
            def add_stage(self, name, optimizer, config):
                """Add optimization stage"""
                self.stages.append({
                    'name': name,
                    'optimizer': optimizer,
                    'config': config
                })
                
            def run_optimization(self, model, data, target):
                """Run multi-stage optimization"""
                current_model = model
                
                for stage in self.stages:
                    stage_name = stage['name']
                    optimizer = stage['optimizer']
                    config = stage['config']
                    
                    # Run optimization stage
                    for epoch in range(config.get('epochs', 1)):
                        output = current_model(data)
                        loss = nn.MSELoss()(output, target)
                        
                        result = optimizer.step(loss)
                        self.optimization_history.append({
                            'stage': stage_name,
                            'epoch': epoch,
                            'loss': loss.item(),
                            'result': result
                        })
                        
                return current_model
                
            def get_optimization_summary(self):
                """Get optimization summary"""
                return {
                    'total_stages': len(self.stages),
                    'total_optimizations': len(self.optimization_history),
                    'stages': [stage['name'] for stage in self.stages]
                }
        
        # Create multi-stage optimizer
        optimizer = MultiStageOptimizer()
        
        # Add stages
        optimizer.add_stage("initial", MockOptimizer(learning_rate=0.001), {'epochs': 3})
        optimizer.add_stage("refinement", MockOptimizer(learning_rate=0.0001), {'epochs': 2})
        optimizer.add_stage("fine_tuning", MockOptimizer(learning_rate=0.00001), {'epochs': 1})
        
        # Test optimization workflow
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Run optimization
        optimized_model = optimizer.run_optimization(model, data, target)
        
        # Verify results
        self.assertIsNotNone(optimized_model)
        summary = optimizer.get_optimization_summary()
        self.assertEqual(summary['total_stages'], 3)
        self.assertGreater(summary['total_optimizations'], 0)
        
    def test_adaptive_optimization_workflow(self):
        """Test adaptive optimization workflow"""
        class AdaptiveOptimizer:
            def __init__(self):
                self.optimizers = {}
                self.performance_history = []
                self.current_optimizer = None
                
            def add_optimizer(self, name, optimizer, performance_threshold=0.1):
                """Add optimizer with performance threshold"""
                self.optimizers[name] = {
                    'optimizer': optimizer,
                    'threshold': performance_threshold,
                    'performance': []
                }
                
            def select_optimizer(self, current_performance):
                """Select best optimizer based on performance"""
                best_optimizer = None
                best_performance = float('inf')
                
                for name, opt_info in self.optimizers.items():
                    if not opt_info['performance'] or current_performance < opt_info['threshold']:
                        if current_performance < best_performance:
                            best_performance = current_performance
                            best_optimizer = name
                            
                return best_optimizer
                
            def optimize(self, model, data, target):
                """Run adaptive optimization"""
                output = model(data)
                loss = nn.MSELoss()(output, target)
                current_performance = loss.item()
                
                # Select optimizer
                selected_optimizer = self.select_optimizer(current_performance)
                
                if selected_optimizer and selected_optimizer in self.optimizers:
                    optimizer = self.optimizers[selected_optimizer]['optimizer']
                    result = optimizer.step(loss)
                    
                    # Update performance history
                    self.optimizers[selected_optimizer]['performance'].append(current_performance)
                    self.performance_history.append({
                        'optimizer': selected_optimizer,
                        'performance': current_performance,
                        'result': result
                    })
                    
                    return result
                
                return None
                
            def get_optimization_stats(self):
                """Get optimization statistics"""
                return {
                    'total_optimizations': len(self.performance_history),
                    'optimizer_usage': {name: len(info['performance']) 
                                       for name, info in self.optimizers.items()},
                    'average_performance': sum(p['performance'] for p in self.performance_history) / 
                                         len(self.performance_history) if self.performance_history else 0
                }
        
        # Create adaptive optimizer
        adaptive_opt = AdaptiveOptimizer()
        
        # Add different optimizers
        adaptive_opt.add_optimizer("fast", MockOptimizer(learning_rate=0.01), 0.5)
        adaptive_opt.add_optimizer("balanced", MockOptimizer(learning_rate=0.001), 0.1)
        adaptive_opt.add_optimizer("precise", MockOptimizer(learning_rate=0.0001), 0.01)
        
        # Test adaptive optimization
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Run multiple optimization steps
        for i in range(5):
            result = adaptive_opt.optimize(model, data, target)
            self.assertIsNotNone(result)
            
        # Check stats
        stats = adaptive_opt.get_optimization_stats()
        self.assertGreater(stats['total_optimizations'], 0)
        self.assertGreater(stats['average_performance'], 0)
        
    def test_ensemble_optimization_workflow(self):
        """Test ensemble optimization workflow"""
        class EnsembleOptimizer:
            def __init__(self):
                self.optimizers = []
                self.weights = []
                self.performance_history = []
                
            def add_optimizer(self, optimizer, weight=1.0):
                """Add optimizer to ensemble"""
                self.optimizers.append(optimizer)
                self.weights.append(weight)
                
            def normalize_weights(self):
                """Normalize optimizer weights"""
                total_weight = sum(self.weights)
                self.weights = [w / total_weight for w in self.weights]
                
            def ensemble_optimize(self, model, data, target):
                """Run ensemble optimization"""
                output = model(data)
                loss = nn.MSELoss()(output, target)
                
                # Run each optimizer
                results = []
                for optimizer in self.optimizers:
                    result = optimizer.step(loss)
                    results.append(result)
                    
                # Weighted combination of results
                ensemble_result = {
                    'optimized': True,
                    'step': sum(r.get('step', 0) for r in results),
                    'ensemble_size': len(self.optimizers)
                }
                
                self.performance_history.append({
                    'loss': loss.item(),
                    'results': results,
                    'ensemble_result': ensemble_result
                })
                
                return ensemble_result
                
            def get_ensemble_stats(self):
                """Get ensemble statistics"""
                return {
                    'ensemble_size': len(self.optimizers),
                    'total_optimizations': len(self.performance_history),
                    'average_loss': sum(p['loss'] for p in self.performance_history) / 
                                 len(self.performance_history) if self.performance_history else 0
                }
        
        # Create ensemble optimizer
        ensemble_opt = EnsembleOptimizer()
        
        # Add multiple optimizers
        ensemble_opt.add_optimizer(MockOptimizer(learning_rate=0.001), 0.4)
        ensemble_opt.add_optimizer(MockOptimizer(learning_rate=0.0005), 0.3)
        ensemble_opt.add_optimizer(MockOptimizer(learning_rate=0.0001), 0.3)
        ensemble_opt.normalize_weights()
        
        # Test ensemble optimization
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Run ensemble optimization
        for i in range(3):
            result = ensemble_opt.ensemble_optimize(model, data, target)
            self.assertTrue(result['optimized'])
            self.assertEqual(result['ensemble_size'], 3)
            
        # Check stats
        stats = ensemble_opt.get_ensemble_stats()
        self.assertEqual(stats['ensemble_size'], 3)
        self.assertGreater(stats['total_optimizations'], 0)

class TestSystemIntegrationWorkflows(unittest.TestCase):
    """Test suite for system integration workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_complete_training_workflow(self):
        """Test complete training workflow"""
        class CompleteTrainingWorkflow:
            def __init__(self):
                self.model = MockModel(input_size=512, hidden_size=1024, output_size=512)
                self.optimizer = MockOptimizer(learning_rate=0.001)
                self.dataset = MockDataset(size=100, input_size=512, output_size=512)
                self.cache = MockKVCache(max_size=1000)
                self.training_history = []
                
            def train_epoch(self, epoch):
                """Train for one epoch"""
                epoch_losses = []
                
                for batch_idx in range(5):  # Simulate batches
                    batch = self.dataset.get_batch(8)
                    
                    # Forward pass
                    output = self.model(batch['input'])
                    loss = nn.MSELoss()(output, batch['target'])
                    
                    # Optimizer step
                    result = self.optimizer.step(loss)
                    
                    # Cache intermediate results
                    self.cache.put(f"epoch_{epoch}_batch_{batch_idx}", output)
                    
                    epoch_losses.append(loss.item())
                    
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                self.training_history.append({
                    'epoch': epoch,
                    'avg_loss': avg_loss,
                    'batches': len(epoch_losses)
                })
                
                return avg_loss
                
            def train(self, num_epochs=5):
                """Complete training workflow"""
                for epoch in range(num_epochs):
                    loss = self.train_epoch(epoch)
                    
                return self.training_history
                
            def get_training_stats(self):
                """Get training statistics"""
                return {
                    'total_epochs': len(self.training_history),
                    'final_loss': self.training_history[-1]['avg_loss'] if self.training_history else 0,
                    'model_stats': self.model.get_model_stats(),
                    'optimizer_stats': self.optimizer.get_optimization_stats(),
                    'cache_stats': self.cache.get_stats()
                }
        
        # Test complete training workflow
        workflow = CompleteTrainingWorkflow()
        training_history = workflow.train(num_epochs=3)
        
        # Verify training results
        self.assertEqual(len(training_history), 3)
        for epoch_data in training_history:
            self.assertIn('epoch', epoch_data)
            self.assertIn('avg_loss', epoch_data)
            self.assertIn('batches', epoch_data)
            
        # Check stats
        stats = workflow.get_training_stats()
        self.assertEqual(stats['total_epochs'], 3)
        self.assertGreater(stats['final_loss'], 0)
        
    def test_optimization_pipeline_workflow(self):
        """Test optimization pipeline workflow"""
        class OptimizationPipeline:
            def __init__(self):
                self.stages = []
                self.pipeline_history = []
                
            def add_stage(self, name, processor, config):
                """Add processing stage"""
                self.stages.append({
                    'name': name,
                    'processor': processor,
                    'config': config
                })
                
            def run_pipeline(self, data):
                """Run complete pipeline"""
                current_data = data
                
                for stage in self.stages:
                    stage_name = stage['name']
                    processor = stage['processor']
                    config = stage['config']
                    
                    # Process data through stage
                    processed_data = processor(current_data, config)
                    
                    # Record stage results
                    self.pipeline_history.append({
                        'stage': stage_name,
                        'input_shape': current_data.shape if hasattr(current_data, 'shape') else 'unknown',
                        'output_shape': processed_data.shape if hasattr(processed_data, 'shape') else 'unknown',
                        'config': config
                    })
                    
                    current_data = processed_data
                    
                return current_data
                
            def get_pipeline_stats(self):
                """Get pipeline statistics"""
                return {
                    'total_stages': len(self.stages),
                    'pipeline_runs': len(self.pipeline_history),
                    'stages': [stage['name'] for stage in self.stages]
                }
        
        # Create optimization pipeline
        pipeline = OptimizationPipeline()
        
        # Add processing stages
        def preprocess(data, config):
            return data * config.get('scale', 1.0)
            
        def optimize(data, config):
            return data + config.get('offset', 0.0)
            
        def postprocess(data, config):
            return data * config.get('final_scale', 1.0)
        
        pipeline.add_stage("preprocess", preprocess, {'scale': 0.5})
        pipeline.add_stage("optimize", optimize, {'offset': 1.0})
        pipeline.add_stage("postprocess", postprocess, {'final_scale': 2.0})
        
        # Test pipeline
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        result = pipeline.run_pipeline(data)
        
        # Verify pipeline execution
        self.assertIsNotNone(result)
        stats = pipeline.get_pipeline_stats()
        self.assertEqual(stats['total_stages'], 3)
        self.assertGreater(stats['pipeline_runs'], 0)
        
    def test_adaptive_system_workflow(self):
        """Test adaptive system workflow"""
        class AdaptiveSystem:
            def __init__(self):
                self.components = {}
                self.adaptation_history = []
                self.performance_metrics = []
                
            def add_component(self, name, component, adaptation_rules):
                """Add adaptive component"""
                self.components[name] = {
                    'component': component,
                    'rules': adaptation_rules,
                    'performance': []
                }
                
            def adapt_component(self, name, current_performance):
                """Adapt component based on performance"""
                if name not in self.components:
                    return False
                    
                component_info = self.components[name]
                rules = component_info['rules']
                
                # Check adaptation rules
                for rule in rules:
                    if rule['condition'](current_performance):
                        # Apply adaptation
                        rule['action'](component_info['component'])
                        
                        self.adaptation_history.append({
                            'component': name,
                            'rule': rule['name'],
                            'performance': current_performance,
                            'timestamp': len(self.adaptation_history)
                        })
                        
                        return True
                        
                return False
                
            def run_adaptive_workflow(self, data, target):
                """Run adaptive workflow"""
                # Simulate workflow execution
                for component_name, component_info in self.components.items():
                    component = component_info['component']
                    
                    # Process data
                    if hasattr(component, 'forward'):
                        output = component(data)
                        loss = nn.MSELoss()(output, target)
                        performance = loss.item()
                    else:
                        performance = 0.5  # Default performance
                        
                    # Record performance
                    component_info['performance'].append(performance)
                    self.performance_metrics.append({
                        'component': component_name,
                        'performance': performance
                    })
                    
                    # Try to adapt
                    self.adapt_component(component_name, performance)
                    
                return self.performance_metrics
                
            def get_adaptation_stats(self):
                """Get adaptation statistics"""
                return {
                    'total_components': len(self.components),
                    'total_adaptations': len(self.adaptation_history),
                    'average_performance': sum(p['performance'] for p in self.performance_metrics) / 
                                         len(self.performance_metrics) if self.performance_metrics else 0
                }
        
        # Create adaptive system
        adaptive_system = AdaptiveSystem()
        
        # Add adaptive components
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        optimizer = MockOptimizer(learning_rate=0.001)
        
        # Define adaptation rules
        def high_loss_condition(performance):
            return performance > 0.5
            
        def low_loss_condition(performance):
            return performance < 0.1
            
        def reduce_lr_action(component):
            component.learning_rate *= 0.9
            
        def increase_lr_action(component):
            component.learning_rate *= 1.1
        
        adaptive_system.add_component("model", model, [
            {'name': 'reduce_lr', 'condition': high_loss_condition, 'action': reduce_lr_action},
            {'name': 'increase_lr', 'condition': low_loss_condition, 'action': increase_lr_action}
        ])
        
        adaptive_system.add_component("optimizer", optimizer, [
            {'name': 'adapt_optimizer', 'condition': high_loss_condition, 'action': reduce_lr_action}
        ])
        
        # Test adaptive workflow
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Run adaptive workflow
        performance_metrics = adaptive_system.run_adaptive_workflow(data, target)
        
        # Verify results
        self.assertGreater(len(performance_metrics), 0)
        stats = adaptive_system.get_adaptation_stats()
        self.assertEqual(stats['total_components'], 2)
        self.assertGreaterEqual(stats['total_adaptations'], 0)

class TestPerformanceOptimizationWorkflows(unittest.TestCase):
    """Test suite for performance optimization workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_performance_optimization_workflow(self):
        """Test performance optimization workflow"""
        class PerformanceOptimizer:
            def __init__(self):
                self.optimization_techniques = []
                self.performance_history = []
                
            def add_technique(self, name, technique, performance_threshold=0.1):
                """Add optimization technique"""
                self.optimization_techniques.append({
                    'name': name,
                    'technique': technique,
                    'threshold': performance_threshold,
                    'applied': False
                })
                
            def optimize_performance(self, model, data, target):
                """Apply performance optimizations"""
                # Measure baseline performance
                self.profiler.start_profile("baseline")
                output = model(data)
                baseline_metrics = self.profiler.end_profile()
                
                # Apply optimization techniques
                optimized_model = model
                applied_techniques = []
                
                for technique_info in self.optimization_techniques:
                    technique = technique_info['technique']
                    threshold = technique_info['threshold']
                    
                    # Apply technique
                    optimized_model = technique(optimized_model)
                    technique_info['applied'] = True
                    applied_techniques.append(technique_info['name'])
                    
                    # Measure performance improvement
                    self.profiler.start_profile(f"optimized_{technique_info['name']}")
                    optimized_output = optimized_model(data)
                    optimized_metrics = self.profiler.end_profile()
                    
                    # Record performance
                    improvement = baseline_metrics['execution_time'] / optimized_metrics['execution_time']
                    self.performance_history.append({
                        'technique': technique_info['name'],
                        'improvement': improvement,
                        'baseline_time': baseline_metrics['execution_time'],
                        'optimized_time': optimized_metrics['execution_time']
                    })
                    
                    # Check if improvement meets threshold
                    if improvement >= (1.0 + threshold):
                        break
                        
                return optimized_model, applied_techniques
                
            def get_optimization_stats(self):
                """Get optimization statistics"""
                if not self.performance_history:
                    return {}
                    
                total_improvement = sum(p['improvement'] for p in self.performance_history)
                return {
                    'total_techniques': len(self.optimization_techniques),
                    'applied_techniques': len([t for t in self.optimization_techniques if t['applied']]),
                    'average_improvement': total_improvement / len(self.performance_history),
                    'best_technique': max(self.performance_history, key=lambda x: x['improvement'])['technique']
                }
        
        # Create performance optimizer
        perf_optimizer = PerformanceOptimizer()
        
        # Add optimization techniques
        def technique1(model):
            # Simulate optimization technique 1
            return model
            
        def technique2(model):
            # Simulate optimization technique 2
            return model
            
        def technique3(model):
            # Simulate optimization technique 3
            return model
        
        perf_optimizer.add_technique("technique1", technique1, 0.1)
        perf_optimizer.add_technique("technique2", technique2, 0.2)
        perf_optimizer.add_technique("technique3", technique3, 0.3)
        
        # Test performance optimization
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        optimized_model, applied_techniques = perf_optimizer.optimize_performance(model, data, target)
        
        # Verify optimization results
        self.assertIsNotNone(optimized_model)
        self.assertGreater(len(applied_techniques), 0)
        
        # Check stats
        stats = perf_optimizer.get_optimization_stats()
        self.assertGreater(stats['total_techniques'], 0)
        self.assertGreaterEqual(stats['applied_techniques'], 0)

if __name__ == '__main__':
    unittest.main()





