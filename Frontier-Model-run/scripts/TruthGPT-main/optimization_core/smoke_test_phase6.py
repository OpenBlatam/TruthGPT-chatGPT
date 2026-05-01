import os
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
mock_modules = [
    "omegaconf", "torch", "torch.nn", "torch.nn.functional", "torch.optim", 
    "torch.cuda", "torch.cuda.amp", "torch.utils", "torch.utils.data",
    "torch.distributed", "numpy", "psutil", "yaml", "tqdm", "transformers", "diffusers"
]
for module in mock_modules:
    m = MagicMock()
    m.__path__ = []
    sys.modules[module] = m

import torch
import torch.nn

# Add paths to sys.path
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from optimization_core.modules.optimizers.core.pytorch_optimizer_base import OptimizationConfig
from optimization_core.modules.optimizers.gpu_optimizer import GPUOptimizationConfig
from optimization_core.modules.optimizers.cosmic_optimization_system import CosmicOptimizationConfig, CosmicOptimizationResult
from optimization_core.modules.optimizers.hyperparameter_optimization import SearchSpace, HyperparameterConfig, HyperparameterTrial
from optimization_core.modules.optimizers.memory_optimizer import MemoryOptimizationConfig
from optimization_core.modules.acceleration.gpu.cuda_kernels import CudaKernelConfig
from optimization_core.utils.truthgpt_enhanced_utils import TruthGPTEnhancedConfig

def test_pydantic_models():
    print("Testing Pydantic Models for Phase 6...")
    
    # 1. OptimizationConfig
    print("  Testing OptimizationConfig...")
    opt_config = OptimizationConfig(learning_rate=0.01)
    assert opt_config.learning_rate == 0.01
    
    # 2. GPUOptimizationConfig
    print("  Testing GPUOptimizationConfig...")
    gpu_config = GPUOptimizationConfig(device_id=0)
    assert gpu_config.device_id == 0
    
    # 3. CosmicOptimizationConfig
    print("  Testing CosmicOptimizationConfig...")
    cosmic_config = CosmicOptimizationConfig(cosmic_frequency=500.0)
    assert cosmic_config.cosmic_frequency == 500.0
    
    # 4. SearchSpace & HyperparameterConfig
    print("  Testing Hyperparameter Optimization Configs...")
    ss = SearchSpace()
    hp_config = HyperparameterConfig(search_space=ss)
    assert hp_config.search_space.learning_rate == (1e-5, 1e-1)
    
    # 5. HyperparameterTrial
    print("  Testing HyperparameterTrial...")
    trial = HyperparameterTrial(trial_id="test", hyperparameters={"lr": 0.01})
    trial.performance_metrics["task_name"] = "smoke_test"
    assert trial.performance_metrics["task_name"] == "smoke_test"
    
    # 6. MemoryOptimizationConfig
    print("  Testing MemoryOptimizationConfig...")
    mem_config = MemoryOptimizationConfig(memory_fraction=0.8)
    assert mem_config.memory_fraction == 0.8
    
    # 7. CudaKernelConfig
    print("  Testing CudaKernelConfig...")
    cuda_config = CudaKernelConfig(threads_per_block=128)
    assert cuda_config.threads_per_block == 128
    
    # 8. TruthGPTEnhancedConfig
    print("  Testing TruthGPTEnhancedConfig...")
    enhanced_config = TruthGPTEnhancedConfig(model_name="test_model")
    assert enhanced_config.model_name == "test_model"
    
    print("\n✅ All Phase 6 Pydantic models verified successfully!")

if __name__ == "__main__":
    try:
        test_pydantic_models()
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        sys.exit(1)
