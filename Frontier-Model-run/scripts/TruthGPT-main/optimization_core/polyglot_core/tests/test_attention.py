"""
Tests for unified Attention module.
"""

import pytest
import numpy as np
from optimization_core.polyglot_core.attention import (
    Attention,
    AttentionConfig,
    AttentionPattern,
    PositionEncoding,
    FlashAttention,
    SparseAttention,
)
from optimization_core.polyglot_core.backend import Backend


def test_attention_config():
    """Test AttentionConfig."""
    config = AttentionConfig(d_model=768, n_heads=12)
    assert config.d_model == 768
    assert config.n_heads == 12
    assert config.head_dim == 64  # 768 / 12
    assert config.softmax_scale == pytest.approx(1.0 / np.sqrt(64))
    
    # Preset configs
    llama_config = AttentionConfig.llama_7b()
    assert llama_config.d_model == 4096
    assert llama_config.n_heads == 32
    
    mistral_config = AttentionConfig.mistral_7b()
    assert mistral_config.pattern == AttentionPattern.SLIDING_WINDOW


def test_attention_forward():
    """Test attention forward pass."""
    config = AttentionConfig(d_model=256, n_heads=4)
    attention = Attention(config)
    
    batch_size = 2
    seq_len = 8
    d_model = 256
    
    q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    k = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    v = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    
    output = attention.forward(q, k, v, batch_size, seq_len)
    
    assert output.output.shape == (batch_size * seq_len, d_model)
    assert not np.isnan(output.output).any()
    assert not np.isinf(output.output).any()


def test_attention_causal():
    """Test causal attention."""
    config = AttentionConfig(
        d_model=128,
        n_heads=2,
        use_causal_mask=True
    )
    attention = Attention(config)
    
    batch_size = 1
    seq_len = 4
    d_model = 128
    
    q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    output = attention.forward(q, q, q, batch_size, seq_len)
    
    assert output.output.shape == (batch_size * seq_len, d_model)


def test_flash_attention():
    """Test Flash Attention."""
    config = AttentionConfig(d_model=256, n_heads=4, block_size=32)
    flash_attn = FlashAttention(config)
    
    batch_size = 2
    seq_len = 16
    d_model = 256
    
    q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    output = flash_attn.forward(q, q, q, batch_size, seq_len)
    
    assert output.output.shape == (batch_size * seq_len, d_model)


def test_sparse_attention():
    """Test Sparse Attention."""
    config = AttentionConfig(d_model=256, n_heads=4)
    sparse_attn = SparseAttention(config, window_size=4, global_tokens=2)
    
    batch_size = 1
    seq_len = 16
    d_model = 256
    
    q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    output = sparse_attn.forward(q, q, q, batch_size, seq_len)
    
    assert output.output.shape == (batch_size * seq_len, d_model)


def test_attention_gqa():
    """Test Grouped-Query Attention."""
    config = AttentionConfig(
        d_model=512,
        n_heads=8,
        n_kv_heads=2  # GQA
    )
    assert config.is_gqa is True
    
    attention = Attention(config)
    
    batch_size = 1
    seq_len = 8
    d_model = 512
    
    q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    k = np.random.randn(batch_size * seq_len, config.n_kv_heads * config.head_dim).astype(np.float32)
    v = np.random.randn(batch_size * seq_len, config.n_kv_heads * config.head_dim).astype(np.float32)
    
    output = attention.forward(q, k, v, batch_size, seq_len)
    assert output.output.shape == (batch_size * seq_len, d_model)


def test_attention_backend():
    """Test backend selection."""
    attention = Attention(AttentionConfig(d_model=256, n_heads=4))
    assert attention.backend in [Backend.PYTHON, Backend.CPP, Backend.RUST]


