
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add parent of CWD to path so we can import optimization_core
sys.path.append(str(Path(os.getcwd()).parent.absolute()))

# --- MOCKS ---
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

def _make_mock_module(name):
    mock = MagicMock()
    mock.__name__ = name
    mock.__path__ = []
    # If it's pydantic, supply BaseModel
    if name == "pydantic":
        mock.BaseModel = MockBaseModel
        mock.ConfigDict = MagicMock()
        mock.Field = MagicMock()
    return mock

_MOCK_MODULES = ["torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.cuda.amp", "numpy", "omegaconf", "pydantic"]

for mod in _MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = _make_mock_module(mod)

# --- IMPORTS ---
from optimization_core.modules.truthgpt.integration import TruthGPTIntegrationConfig
from optimization_core.modules.truthgpt.optimization_utils import TruthGPTOptimizationConfig
from optimization_core.modules.feed_forward.optimization.ultra_optimization.zero_copy_optimizer import ZeroCopyConfig
from optimization_core.modules.feed_forward.optimization.ultra_optimization.model_compiler import CompilationConfig
from optimization_core.modules.feed_forward.optimization.ultra_optimization.dynamic_batcher import BatchingConfig

def test_pydantic_configs():
    print("🧪 Testing Pydantic Configs (Mocked Edition)...")
    
    configs = [
        ("TruthGPTIntegrationConfig", TruthGPTIntegrationConfig()),
        ("TruthGPTOptimizationConfig", TruthGPTOptimizationConfig()),
        ("ZeroCopyConfig", ZeroCopyConfig()),
        ("CompilationConfig", CompilationConfig()),
        ("BatchingConfig", BatchingConfig())
    ]
    
    for name, config in configs:
        print(f"  Checking {name}...")
        # Test instantiation
        assert config is not None
        # Test model_dump (mocked)
        data = config.model_dump()
        assert isinstance(data, dict)
        print(f"    ✅ {name} instantiated and dumped successfully.")
        
    print("\n🚀 All Pydantic config tests passed!")

if __name__ == "__main__":
    try:
        test_pydantic_configs()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
