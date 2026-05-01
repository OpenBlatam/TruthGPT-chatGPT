"""
TruthGPT Testing Framework
Comprehensive testing utilities for all TruthGPT modules
"""

import unittest
import pytest
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from contextlib import contextmanager
import tempfile
import os
import json
from pathlib import Path

# Import all TruthGPT modules
from .training import TruthGPTTrainer, TruthGPTTrainingConfig, TruthGPTTrainingMetrics
from .data import TruthGPTDataLoader, TruthGPTDataset, TruthGPTDataConfig
from .models import TruthGPTModel, TruthGPTModelConfig
from .optimizers import TruthGPTOptimizer, TruthGPTScheduler, TruthGPTOptimizerConfig
from .evaluation import TruthGPTEvaluator, TruthGPTEvaluationConfig
from .inference import TruthGPTInference, TruthGPTInferenceConfig
from .monitoring import TruthGPTMonitor, TruthGPTProfiler, TruthGPTLogger
from .config import TruthGPTConfigManager, TruthGPTConfigValidator
from .distributed import TruthGPTDistributedManager, TruthGPTDistributedTrainer
from .compression import TruthGPTCompressionManager
from .attention import TruthGPTAttentionFactory, TruthGPTRotaryEmbedding
from .augmentation import TruthGPTAugmentationManager
from .analytics import TruthGPTAnalyticsManager
from .deployment import TruthGPTDeploymentManager
from .integration import TruthGPTIntegrationManager
from .security import TruthGPTSecurityManager


class TestLevel(Enum):
    """Test complexity levels"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    STRESS = "stress"


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestConfig:
    """Configuration for TruthGPT testing"""
    test_level: TestLevel = TestLevel.UNIT
    device: str = "cpu"
    batch_size: int = 2
    sequence_length: int = 128
    vocab_size: int = 1000
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_iterations: int = 10
    timeout_seconds: int = 300
    memory_limit_mb: int = 1024
    enable_profiling: bool = True
    enable_logging: bool = True
    temp_dir: Optional[str] = None
    cleanup_after_test: bool = True
    parallel_tests: bool = False
    random_seed: int = 42


@dataclass
class TestMetrics:
    """Test execution metrics"""
    test_name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    result: TestResult = TestResult.PASSED
    error_message: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class TruthGPTTestSuite:
    """Comprehensive test suite for TruthGPT modules"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestMetrics] = []
        self.temp_dir = config.temp_dir or tempfile.mkdtemp()
        self.logger = self._setup_logger()
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup test logger"""
        logger = logging.getLogger(f"TruthGPTTestSuite_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @contextmanager
    def test_context(self, test_name: str):
        """Context manager for test execution"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        self.logger.info(f"Starting test: {test_name}")
        
        try:
            yield
            duration = time.time() - start_time
            end_memory = self._get_memory_usage()
            
            metrics = TestMetrics(
                test_name=test_name,
                duration=duration,
                memory_usage=end_memory - start_memory,
                cpu_usage=self._get_cpu_usage(),
                result=TestResult.PASSED
            )
            
            self.results.append(metrics)
            self.logger.info(f"Test passed: {test_name} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            
            metrics = TestMetrics(
                test_name=test_name,
                duration=duration,
                memory_usage=self._get_memory_usage() - start_memory,
                cpu_usage=self._get_cpu_usage(),
                result=TestResult.FAILED,
                error_message=str(e)
            )
            
            self.results.append(metrics)
            self.logger.error(f"Test failed: {test_name} - {str(e)}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def test_model_creation(self):
        """Test TruthGPT model creation and basic operations"""
        with self.test_context("test_model_creation"):
            config = TruthGPTModelConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                max_length=self.config.sequence_length
            )
            
            model = TruthGPTModel(config)
            model = model.to(self.device)
            
            # Test forward pass
            batch_size = self.config.batch_size
            input_ids = torch.randint(0, self.config.vocab_size, 
                                    (batch_size, self.config.sequence_length),
                                    device=self.device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            assert outputs.logits.shape == (batch_size, self.config.sequence_length, self.config.vocab_size)
            assert outputs.last_hidden_state.shape == (batch_size, self.config.sequence_length, self.config.hidden_size)
    
    def test_data_loading(self):
        """Test data loading and preprocessing"""
        with self.test_context("test_data_loading"):
            # Create dummy data
            data_path = os.path.join(self.temp_dir, "test_data.jsonl")
            with open(data_path, 'w') as f:
                for i in range(100):
                    f.write(json.dumps({"text": f"Sample text {i}"}) + "\n")
            
            config = TruthGPTDataConfig(
                data_path=data_path,
                batch_size=self.config.batch_size,
                max_length=self.config.sequence_length,
                tokenizer_name="gpt2"
            )
            
            dataset = TruthGPTDataset(config)
            dataloader = TruthGPTDataLoader(dataset, config)
            
            # Test data loading
            batch = next(iter(dataloader))
            assert batch.input_ids.shape[0] == self.config.batch_size
            assert batch.input_ids.shape[1] <= self.config.sequence_length
    
    def test_training_loop(self):
        """Test training loop functionality"""
        with self.test_context("test_training_loop"):
            # Create model and data
            model_config = TruthGPTModelConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads
            )
            
            training_config = TruthGPTTrainingConfig(
                learning_rate=1e-4,
                batch_size=self.config.batch_size,
                max_epochs=1,
                max_steps=self.config.max_iterations,
                device=self.config.device
            )
            
            trainer = TruthGPTTrainer(model_config, training_config)
            
            # Create dummy data
            data_path = os.path.join(self.temp_dir, "training_data.jsonl")
            with open(data_path, 'w') as f:
                for i in range(50):
                    f.write(json.dumps({"text": f"Training sample {i}"}) + "\n")
            
            # Test training
            metrics = trainer.train(data_path)
            assert isinstance(metrics, TruthGPTTrainingMetrics)
            assert metrics.total_steps > 0
    
    def test_optimization(self):
        """Test optimization utilities"""
        with self.test_context("test_optimization"):
            config = TruthGPTModelConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads
            )
            
            model = TruthGPTModel(config).to(self.device)
            
            # Test optimizer creation
            opt_config = TruthGPTOptimizerConfig(
                learning_rate=1e-4,
                weight_decay=0.01,
                optimizer_type="adamw"
            )
            
            optimizer = TruthGPTOptimizer(model.parameters(), opt_config)
            scheduler = TruthGPTScheduler(optimizer, opt_config)
            
            assert optimizer is not None
            assert scheduler is not None
    
    def test_evaluation(self):
        """Test evaluation functionality"""
        with self.test_context("test_evaluation"):
            config = TruthGPTModelConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads
            )
            
            model = TruthGPTModel(config).to(self.device)
            
            eval_config = TruthGPTEvaluationConfig(
                batch_size=self.config.batch_size,
                device=self.config.device,
                metrics=["perplexity", "accuracy"]
            )
            
            evaluator = TruthGPTEvaluator(model, eval_config)
            
            # Create dummy test data
            test_data = [
                {"text": "Test sample 1"},
                {"text": "Test sample 2"},
                {"text": "Test sample 3"}
            ]
            
            metrics = evaluator.evaluate(test_data)
            assert isinstance(metrics, dict)
            assert "perplexity" in metrics or "accuracy" in metrics
    
    def test_inference(self):
        """Test inference functionality"""
        with self.test_context("test_inference"):
            config = TruthGPTModelConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads
            )
            
            model = TruthGPTModel(config).to(self.device)
            
            inference_config = TruthGPTInferenceConfig(
                max_length=self.config.sequence_length,
                temperature=0.8,
                top_p=0.9,
                device=self.config.device
            )
            
            inference = TruthGPTInference(model, inference_config)
            
            # Test text generation
            prompt = "Hello world"
            generated = inference.generate(prompt, max_length=50)
            
            assert isinstance(generated, str)
            assert len(generated) > len(prompt)
    
    def test_monitoring(self):
        """Test monitoring and profiling"""
        with self.test_context("test_monitoring"):
            monitor = TruthGPTMonitor()
            profiler = TruthGPTProfiler()
            logger = TruthGPTLogger()
            
            # Test monitoring
            monitor.start_monitoring()
            time.sleep(0.1)
            metrics = monitor.get_metrics()
            monitor.stop_monitoring()
            
            assert isinstance(metrics, dict)
            assert "cpu_usage" in metrics
            assert "memory_usage" in metrics
    
    def test_config_management(self):
        """Test configuration management"""
        with self.test_context("test_config_management"):
            config_manager = TruthGPTConfigManager()
            validator = TruthGPTConfigValidator()
            
            # Test config validation
            config = {
                "model": {
                    "vocab_size": 1000,
                    "hidden_size": 256,
                    "num_layers": 2
                },
                "training": {
                    "learning_rate": 1e-4,
                    "batch_size": 32
                }
            }
            
            is_valid = validator.validate(config)
            assert isinstance(is_valid, bool)
    
    def test_compression(self):
        """Test model compression"""
        with self.test_context("test_compression"):
            config = TruthGPTModelConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads
            )
            
            model = TruthGPTModel(config).to(self.device)
            
            compression_manager = TruthGPTCompressionManager()
            
            # Test quantization
            compressed_model = compression_manager.quantize_model(model)
            assert compressed_model is not None
    
    def test_attention_mechanisms(self):
        """Test advanced attention mechanisms"""
        with self.test_context("test_attention_mechanisms"):
            attention_factory = TruthGPTAttentionFactory()
            
            # Test rotary embedding
            rotary_emb = TruthGPTRotaryEmbedding(
                dim=self.config.hidden_size,
                max_seq_len=self.config.sequence_length
            )
            
            assert rotary_emb is not None
    
    def test_augmentation(self):
        """Test data augmentation"""
        with self.test_context("test_augmentation"):
            augmentation_manager = TruthGPTAugmentationManager()
            
            # Test text augmentation
            original_text = "This is a test sentence."
            augmented = augmentation_manager.augment_text(original_text)
            
            assert isinstance(augmented, str)
            assert len(augmented) > 0
    
    def test_analytics(self):
        """Test analytics and reporting"""
        with self.test_context("test_analytics"):
            analytics_manager = TruthGPTAnalyticsManager()
            
            # Test analytics
            data = {"metric1": 0.5, "metric2": 0.8, "metric3": 0.3}
            report = analytics_manager.generate_report(data)
            
            assert isinstance(report, dict)
    
    def test_deployment(self):
        """Test deployment functionality"""
        with self.test_context("test_deployment"):
            deployment_manager = TruthGPTDeploymentManager()
            
            # Test deployment preparation
            config = TruthGPTModelConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads
            )
            
            model = TruthGPTModel(config)
            
            # Test model export
            export_path = os.path.join(self.temp_dir, "exported_model")
            deployment_manager.export_model(model, export_path)
            
            assert os.path.exists(export_path)
    
    def test_integration(self):
        """Test integration capabilities"""
        with self.test_context("test_integration"):
            integration_manager = TruthGPTIntegrationManager()
            
            # Test integration setup
            config = {
                "api_endpoints": ["http://localhost:8000"],
                "timeout": 30,
                "retry_attempts": 3
            }
            
            integration_manager.setup_integration(config)
            assert integration_manager.is_configured()
    
    def test_security(self):
        """Test security features"""
        with self.test_context("test_security"):
            security_manager = TruthGPTSecurityManager()
            
            # Test encryption
            data = "sensitive data"
            encrypted = security_manager.encrypt_data(data)
            decrypted = security_manager.decrypt_data(encrypted)
            
            assert decrypted == data
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        self.logger.info("Starting comprehensive TruthGPT test suite")
        
        test_methods = [
            self.test_model_creation,
            self.test_data_loading,
            self.test_training_loop,
            self.test_optimization,
            self.test_evaluation,
            self.test_inference,
            self.test_monitoring,
            self.test_config_management,
            self.test_compression,
            self.test_attention_mechanisms,
            self.test_augmentation,
            self.test_analytics,
            self.test_deployment,
            self.test_integration,
            self.test_security
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
            except Exception as e:
                failed += 1
                self.logger.error(f"Test {test_method.__name__} failed: {str(e)}")
        
        # Cleanup
        if self.config.cleanup_after_test:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        summary = {
            "total_tests": len(test_methods),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_methods),
            "results": self.results,
            "config": self.config.__dict__
        }
        
        self.logger.info(f"Test suite completed: {passed}/{len(test_methods)} tests passed")
        return summary


def create_truthgpt_test_suite(config: Optional[TestConfig] = None) -> TruthGPTTestSuite:
    """Create a TruthGPT test suite with default configuration"""
    if config is None:
        config = TestConfig()
    return TruthGPTTestSuite(config)


def quick_truthgpt_testing(
    test_level: TestLevel = TestLevel.UNIT,
    device: str = "cpu",
    timeout: int = 300
) -> Dict[str, Any]:
    """Quick testing function for TruthGPT modules"""
    config = TestConfig(
        test_level=test_level,
        device=device,
        timeout_seconds=timeout,
        batch_size=2,
        max_iterations=5
    )
    
    test_suite = TruthGPTTestSuite(config)
    return test_suite.run_all_tests()


# Example usage
if __name__ == "__main__":
    # Run comprehensive test suite
    results = quick_truthgpt_testing(TestLevel.INTEGRATION)
    print(f"Test Results: {results['passed']}/{results['total_tests']} passed")
    print(f"Success Rate: {results['success_rate']:.2%}")
