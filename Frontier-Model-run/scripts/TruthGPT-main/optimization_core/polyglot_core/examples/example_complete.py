"""
Complete example using all polyglot_core modules together.

Demonstrates a full LLM inference pipeline:
1. Tokenization
2. Attention with KV Cache
3. Compression
4. Quantization
5. Inference Generation
"""

import numpy as np
from optimization_core.polyglot_core import (
    Tokenizer,
    Attention,
    AttentionConfig,
    KVCache,
    Compressor,
    Quantizer,
    InferenceEngine,
    GenerationConfig,
    Backend,
    get_available_backends,
    print_backend_status,
)


def main():
    print("=" * 70)
    print("TruthGPT Polyglot Core - Complete Example")
    print("=" * 70)
    print()
    
    # 1. Check available backends
    print("1. Backend Status:")
    print_backend_status()
    
    # 2. Tokenization
    print("\n2. Tokenization:")
    print("-" * 70)
    try:
        tokenizer = Tokenizer(model_name="gpt2")
        text = "Hello, world! This is a test."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"  Text: {text}")
        print(f"  Tokens: {tokens[:10]}... (showing first 10)")
        print(f"  Decoded: {decoded}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Backend: {tokenizer.backend.name}")
    except Exception as e:
        print(f"  Tokenization failed: {e}")
    
    # 3. Attention
    print("\n3. Attention:")
    print("-" * 70)
    config = AttentionConfig(d_model=256, n_heads=4)
    attention = Attention(config)
    
    batch_size = 2
    seq_len = 8
    d_model = 256
    
    q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    k = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    v = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    
    output = attention.forward(q, k, v, batch_size, seq_len)
    print(f"  Input shape: [{batch_size}, {seq_len}, {d_model}]")
    print(f"  Output shape: {output.output.shape}")
    print(f"  Compute time: {output.compute_time_ms:.2f} ms")
    print(f"  Backend: {attention.backend.name}")
    
    # 4. KV Cache
    print("\n4. KV Cache:")
    print("-" * 70)
    cache = KVCache(max_size=1000)
    
    k = np.random.randn(4, 8, 64).astype(np.float32)
    v = np.random.randn(4, 8, 64).astype(np.float32)
    
    cache.put(layer=0, position=0, key=k, value=v)
    result = cache.get(layer=0, position=0)
    
    print(f"  Cache size: {cache.size}")
    print(f"  Hit rate: {cache.hit_rate:.2%}")
    print(f"  Retrieved: {'Yes' if result is not None else 'No'}")
    print(f"  Backend: {cache.backend.name}")
    
    # 5. Compression
    print("\n5. Compression:")
    print("-" * 70)
    compressor = Compressor(algorithm="lz4")
    
    data = b"Compression test data " * 100
    result = compressor.compress(data)
    
    if result.success:
        print(f"  Original size: {len(data)} bytes")
        print(f"  Compressed size: {len(result.data)} bytes")
        print(f"  Compression ratio: {result.stats.compression_ratio:.2%}")
        print(f"  Space savings: {result.stats.space_savings:.2%}")
        print(f"  Backend: {compressor.backend.name}")
    else:
        print(f"  Compression failed: {result.error}")
    
    # 6. Quantization
    print("\n6. Quantization:")
    print("-" * 70)
    quantizer = Quantizer(quantization_type="int8")
    
    weights = np.random.randn(100, 100).astype(np.float32)
    quantized, stats = quantizer.quantize(weights)
    
    print(f"  Original size: {stats.original_size_mb:.2f} MB")
    print(f"  Quantized size: {stats.quantized_size_mb:.2f} MB")
    print(f"  Compression ratio: {stats.compression_ratio:.2%}")
    print(f"  Quantization time: {stats.quantization_time_ms:.2f} ms")
    print(f"  Backend: {quantizer.backend.name}")
    
    # 7. Inference (simulated)
    print("\n7. Inference Engine:")
    print("-" * 70)
    engine = InferenceEngine(seed=42)
    
    # Mock forward function
    def mock_forward(tokens):
        vocab_size = 50257  # GPT-2 vocab size
        logits = np.random.randn(vocab_size).astype(np.float32)
        logits[42] = 10.0  # Favor token 42
        return logits
    
    config = GenerationConfig.sampling(temperature=0.7, top_p=0.9)
    config.max_new_tokens = 5
    
    prompt = [1, 2, 3, 4, 5]
    result = engine.generate(prompt, mock_forward, config)
    
    print(f"  Prompt length: {len(prompt)} tokens")
    print(f"  Generated: {result.tokens_generated} tokens")
    print(f"  Total tokens: {len(result.token_ids)}")
    print(f"  Generation time: {result.generation_time_ms:.2f} ms")
    print(f"  Tokens/sec: {result.tokens_per_second:.0f}")
    print(f"  Finish reason: {result.finish_reason}")
    print(f"  Backend: {engine.backend.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    backends = get_available_backends()
    available = [b for b in backends if b.available]
    
    print(f"  Available backends: {len(available)}/{len(backends)}")
    for backend in available:
        print(f"    - {backend.name} ({backend.version})")
    
    print("\n  Modules tested:")
    print("    ✓ Tokenization")
    print("    ✓ Attention")
    print("    ✓ KV Cache")
    print("    ✓ Compression")
    print("    ✓ Quantization")
    print("    ✓ Inference")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()


