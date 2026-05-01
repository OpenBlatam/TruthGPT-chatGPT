"""
Example: Batch processing and streaming.

Demonstrates batching, streaming, and parallel processing.
"""

import numpy as np
from optimization_core.polyglot_core import (
    KVCache,
    Attention,
    AttentionConfig,
    batch,
    process_batches,
    pad_batch,
    stream_process,
    TokenStreamer,
    InferenceEngine,
    GenerationConfig,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Batch Processing & Streaming Example")
    print("=" * 80)
    print()
    
    # 1. Batch Processing
    print("1. Batch Processing:")
    print("-" * 80)
    
    # Create test data
    items = [np.random.randn(64).astype(np.float32) for _ in range(100)]
    
    # Process in batches
    def process_batch(batch_items):
        # Simulate processing
        return [item * 2 for item in batch_items]
    
    batches = list(batch(items, batch_size=10))
    print(f"Created {len(batches)} batches from {len(items)} items")
    
    results = process_batches(items, process_batch, batch_size=10, parallel=False)
    print(f"Processed {len(results)} batches")
    
    # 2. Padding
    print("\n2. Batch Padding:")
    print("-" * 80)
    
    sequences = [
        np.random.randn(5, 10).astype(np.float32),
        np.random.randn(8, 10).astype(np.float32),
        np.random.randn(3, 10).astype(np.float32),
    ]
    
    padded = pad_batch(sequences)
    print(f"Original shapes: {[s.shape for s in sequences]}")
    print(f"Padded shape: {padded.shape}")
    
    # 3. Streaming
    print("\n3. Streaming:")
    print("-" * 80)
    
    def process_item(item):
        return item * 2
    
    # Create stream
    def item_generator():
        for i in range(10):
            yield np.random.randn(32).astype(np.float32)
    
    stream = stream_process(item_generator(), process_item, batch_size=3)
    
    count = 0
    for result in stream:
        count += 1
        if count <= 3:
            print(f"  Processed item {count}: shape {result.shape}")
    
    print(f"Total items processed: {count}")
    
    # 4. Attention Batching
    print("\n4. Attention Batching:")
    print("-" * 80)
    
    attention = Attention(AttentionConfig(d_model=256, n_heads=4))
    
    # Create batch of queries
    queries = [np.random.randn(8, 256).astype(np.float32) for _ in range(4)]
    
    # Process in batches
    def process_attention_batch(batch_queries):
        results = []
        for q in batch_queries:
            output = attention.forward(q, q, q, batch_size=1, seq_len=8)
            results.append(output.output)
        return results
    
    attention_results = process_batches(queries, process_attention_batch, batch_size=2)
    print(f"Processed {len(attention_results)} attention batches")
    
    # 5. Cache Batching
    print("\n5. Cache Batching:")
    print("-" * 80)
    
    cache = KVCache(max_size=1000)
    
    # Batch cache operations
    cache_ops = [
        (0, i, np.random.randn(32).astype(np.float32), np.random.randn(32).astype(np.float32))
        for i in range(20)
    ]
    
    def process_cache_batch(batch_ops):
        for layer, position, key, value in batch_ops:
            cache.put(layer, position, key, value)
        return [cache.get(layer, position) for layer, position, _, _ in batch_ops]
    
    cache_results = process_batches(cache_ops, process_cache_batch, batch_size=5)
    print(f"Processed {len(cache_results)} cache batches")
    print(f"Cache size: {cache.size}")
    
    print("\n" + "=" * 80)
    print("Batch Processing & Streaming Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()













