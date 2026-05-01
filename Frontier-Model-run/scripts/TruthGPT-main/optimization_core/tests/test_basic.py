"""
Basic tests for optimization_core components.
Run with: pytest tests/test_basic.py -v
"""

import pytest
import torch
import yaml
from pathlib import Path


def test_registry_base():
    """Test registry base functionality."""
    from factories.registry import Registry
    
    reg = Registry()
    
    @reg.register("test_item")
    def test_fn():
        return "test"
    
    assert reg.get("test_item") == test_fn
    assert reg.build("test_item")() == "test"
    
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_attention_backends():
    """Test attention backend registry."""
    from factories.attention import ATTENTION_BACKENDS
    
    assert "sdpa" in ATTENTION_BACKENDS._items
    assert "flash" in ATTENTION_BACKENDS._items
    assert "triton" in ATTENTION_BACKENDS._items


def test_optimizer_registry():
    """Test optimizer registry."""
    from factories.optimizer import OPTIMIZERS
    
    assert "adamw" in OPTIMIZERS._items
    assert "lion" in OPTIMIZERS._items
    assert "adafactor" in OPTIMIZERS._items


def test_config_yaml():
    """Test that default config is valid YAML."""
    config_path = Path("configs/llm_default.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    assert "model" in cfg
    assert "training" in cfg
    assert "data" in cfg


def test_trainer_config():
    """Test TrainerConfig dataclass."""
    from trainers.trainer import TrainerConfig
    
    cfg = TrainerConfig()
    assert cfg.seed == 42
    assert cfg.model_name == "gpt2"
    assert cfg.epochs == 3
    assert cfg.mixed_precision == "bf16"
    assert cfg.ema_enabled is True


def test_memory_manager():
    """Test advanced memory manager."""
    from modules.memory.advanced_memory_manager import create_advanced_memory_manager
    
    mm = create_advanced_memory_manager()
    assert mm is not None
    
    dtype = mm.select_dtype_adaptive()
    assert dtype in [torch.float32, torch.float16, torch.bfloat16]
    
    caps = mm.detect_gpu_capabilities()
    assert "cuda" in caps
    assert "bf16_ok" in caps


def test_kv_cache():
    """Test KV cache creation."""
    from modules.attention.ultra_efficient_kv_cache import PagedKVCache
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    cache = PagedKVCache(num_heads=8, head_dim=64, max_tokens=1024, block_size=128)
    assert cache.length == 0
    
    # Test append
    k = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device="cuda")
    cache.append(k, v)
    assert cache.length == 32


def test_build_components():
    """Test build_components function."""
    from optimization_core.build import build_components
    
    config = {
        "model": {
            "attention": {"backend": "sdpa"},
            "kv_cache": {"type": "paged", "block_size": 128},
            "memory": {"policy": "adaptive"},
        }
    }
    
    comps = build_components(config)
    assert "attention" in comps
    assert "kv_cache_builder" in comps
    assert "memory_manager" in comps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



