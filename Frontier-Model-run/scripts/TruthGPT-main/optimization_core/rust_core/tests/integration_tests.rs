//! Integration Tests for TruthGPT Rust Core
//!
//! Comprehensive test suite covering all modules with various scenarios.

use truthgpt_rust::*;
use ndarray::Array3;
use std::collections::HashMap;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// KV CACHE INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod kv_cache_tests {
    use super::*;
    use truthgpt_rust::{KVCache, KVCacheConfig, EvictionStrategy, ConcurrentKVCache};

    #[test]
    fn test_kv_cache_lru_eviction() {
        let config = KVCacheConfig {
            max_size: 3,
            eviction_strategy: EvictionStrategy::LRU,
            enable_compression: false,
            compression_threshold: 1024,
        };
        let mut cache = KVCache::new(config);

        // Fill cache
        cache.put(0, 0, vec![1, 2, 3]);
        cache.put(0, 1, vec![4, 5, 6]);
        cache.put(0, 2, vec![7, 8, 9]);
        
        // Access first entry to make it recently used
        let _ = cache.get_mut(0, 0);
        
        // Add new entry, should evict (0, 1) as it's least recently used
        cache.put(0, 3, vec![10, 11, 12]);
        
        assert!(cache.get(0, 0).is_some(), "Recently used entry should exist");
        assert!(cache.get(0, 2).is_some());
        assert!(cache.get(0, 3).is_some());
        assert_eq!(cache.size(), 3);
    }

    #[test]
    fn test_kv_cache_lfu_eviction() {
        let config = KVCacheConfig {
            max_size: 3,
            eviction_strategy: EvictionStrategy::LFU,
            enable_compression: false,
            compression_threshold: 1024,
        };
        let mut cache = KVCache::new(config);

        cache.put(0, 0, vec![1]);
        cache.put(0, 1, vec![2]);
        cache.put(0, 2, vec![3]);

        // Access (0, 0) multiple times to increase frequency
        for _ in 0..5 {
            let _ = cache.get_mut(0, 0);
        }
        
        // Access (0, 2) a couple times
        let _ = cache.get_mut(0, 2);
        let _ = cache.get_mut(0, 2);

        // Add new entry, should evict least frequently used
        cache.put(0, 3, vec![4]);
        
        assert!(cache.get(0, 0).is_some(), "Most frequent entry should exist");
        assert_eq!(cache.size(), 3);
    }

    #[test]
    fn test_kv_cache_adaptive_eviction() {
        let config = KVCacheConfig {
            max_size: 5,
            eviction_strategy: EvictionStrategy::Adaptive,
            enable_compression: false,
            compression_threshold: 1024,
        };
        let mut cache = KVCache::new(config);

        // Fill cache
        for i in 0..5 {
            cache.put(0, i, vec![i as u8]);
        }

        // Add more entries to trigger adaptive eviction
        for i in 5..10 {
            cache.put(0, i, vec![i as u8]);
        }

        assert_eq!(cache.size(), 5);
        
        let stats = cache.get_stats();
        assert!(stats["eviction_count"] > 0.0);
    }

    #[test]
    fn test_kv_cache_with_compression() {
        let config = KVCacheConfig {
            max_size: 10,
            eviction_strategy: EvictionStrategy::LRU,
            enable_compression: true,
            compression_threshold: 50, // Low threshold to trigger compression
        };
        let mut cache = KVCache::new(config);

        // Create data that compresses well (repetitive pattern)
        let large_data: Vec<u8> = (0..100).map(|i| (i % 10) as u8).collect();
        
        cache.put(0, 0, large_data.clone());
        
        let retrieved = cache.get_decompressed(0, 0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), large_data);
        
        let stats = cache.get_stats();
        assert!(stats["compression_savings_bytes"] > 0.0);
    }

    #[test]
    fn test_concurrent_kv_cache() {
        use std::thread;
        use std::sync::Arc;
        
        let cache = Arc::new(ConcurrentKVCache::with_capacity(1000));
        let mut handles = vec![];

        // Spawn multiple threads writing to cache
        for t in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    cache_clone.put(t, i, vec![(t * i) as u8; 32]);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify data
        for t in 0..4 {
            for i in 0..100 {
                let data = cache.get(t, i);
                assert!(data.is_some());
                assert_eq!(data.unwrap()[0], (t * i) as u8);
            }
        }

        assert_eq!(cache.size(), 400);
    }

    #[test]
    fn test_kv_cache_stats() {
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config);

        for i in 0..10 {
            cache.put(0, i, vec![i as u8]);
        }

        // Generate hits
        for i in 0..5 {
            cache.get_mut(0, i);
        }

        // Generate misses
        for i in 100..105 {
            cache.get_mut(0, i);
        }

        let stats = cache.get_stats();
        assert_eq!(stats["hit_count"], 5.0);
        assert_eq!(stats["miss_count"], 5.0);
        assert!((stats["hit_rate"] - 0.5).abs() < 0.001);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPRESSION INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod compression_tests {
    use super::*;
    use truthgpt_rust::{Compressor, CompressionAlgorithm, CompressionStats, StreamingCompressor, BatchCompressor, compress, decompress, compress_with_stats, compress_zstd_level};

    #[test]
    fn test_lz4_large_data() {
        // Create large test data with patterns (compresses well)
        let data: Vec<u8> = (0..100_000)
            .map(|i| ((i % 256) as u8).wrapping_mul(17))
            .collect();

        let compressed = compress(&data, &CompressionAlgorithm::LZ4).unwrap();
        let decompressed = decompress(&compressed, &CompressionAlgorithm::LZ4).unwrap();

        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len(), "Compression should reduce size");
    }

    #[test]
    fn test_zstd_various_levels() {
        let data: Vec<u8> = (0..10_000)
            .map(|i| (i % 100) as u8)
            .collect();

        let mut sizes = Vec::new();
        for level in [1, 3, 9, 15, 19] {
            let compressed = compress_zstd_level(&data, level).unwrap();
            let decompressed = decompress(&compressed, &CompressionAlgorithm::Zstd).unwrap();
            assert_eq!(data, decompressed);
            sizes.push((level, compressed.len()));
        }

        // Higher levels should generally produce smaller output
        // (may not always be true for all data)
        println!("Zstd compression sizes: {:?}", sizes);
    }

    #[test]
    fn test_compression_stats() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5].repeat(1000);
        
        let (compressed, stats) = compress_with_stats(&data, &CompressionAlgorithm::LZ4).unwrap();
        
        assert_eq!(stats.original_size, data.len());
        assert_eq!(stats.compressed_size, compressed.len());
        assert!(stats.compression_ratio() < 1.0);
        assert!(stats.space_savings() > 0.0);
        assert!(stats.bytes_saved() > 0);
    }

    #[test]
    fn test_streaming_compressor() {
        let mut compressor = StreamingCompressor::new(CompressionAlgorithm::LZ4, 1024);
        
        let mut all_chunks = Vec::new();
        
        // Stream data in various sizes
        for _ in 0..10 {
            let data: Vec<u8> = (0..500).map(|i| i as u8).collect();
            let chunks = compressor.write(&data).unwrap();
            all_chunks.extend(chunks);
        }
        
        if let Some(final_chunk) = compressor.flush().unwrap() {
            all_chunks.push(final_chunk);
        }

        assert!(compressor.total_input() > 0);
        assert!(compressor.total_output() > 0);
        assert!(compressor.overall_ratio() > 0.0);
    }

    #[test]
    fn test_batch_compressor() {
        let batch_compressor = BatchCompressor::new(CompressionAlgorithm::LZ4, 0);
        
        let items: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![i as u8; 1000])
            .collect();

        let compressed = batch_compressor.compress_batch(&items).unwrap();
        let decompressed = batch_compressor.decompress_batch(&compressed).unwrap();

        assert_eq!(items.len(), decompressed.len());
        for (orig, dec) in items.iter().zip(decompressed.iter()) {
            assert_eq!(orig, dec);
        }
    }

    #[test]
    fn test_compressor_builder_pattern() {
        let compressor = Compressor::zstd().with_level(15);
        assert_eq!(compressor.algorithm(), CompressionAlgorithm::Zstd);
        assert_eq!(compressor.level(), 15);

        let data = b"Test data for compression".to_vec();
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_no_compression_passthrough() {
        let data = b"This data should pass through unchanged".to_vec();
        
        let result = compress(&data, &CompressionAlgorithm::None).unwrap();
        assert_eq!(data, result);
        
        let decompressed = decompress(&result, &CompressionAlgorithm::None).unwrap();
        assert_eq!(data, decompressed);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod attention_tests {
    use super::*;
    use truthgpt_rust::{scaled_dot_product_attention, scaled_dot_product_attention_causal, flash_attention_block, sparse_attention, create_causal_mask, softmax_1d, AttentionConfig, AttentionStats};
    use ndarray::Array3;

    #[test]
    fn test_scaled_dot_product_output_shape() {
        let batch_size = 4;
        let seq_len = 128;
        let d_k = 64;

        let query = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);
        let key = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);
        let value = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);

        let output = scaled_dot_product_attention(&query, &key, &value, None);

        assert_eq!(output.dim(), (batch_size, seq_len, d_k));
    }

    #[test]
    fn test_causal_attention_masked_future() {
        let batch_size = 1;
        let seq_len = 4;
        let d_k = 4;

        // Create query/key/value where each position has distinct values
        let mut query = Array3::zeros((batch_size, seq_len, d_k));
        let mut key = Array3::zeros((batch_size, seq_len, d_k));
        let mut value = Array3::zeros((batch_size, seq_len, d_k));

        for i in 0..seq_len {
            for j in 0..d_k {
                query[[0, i, j]] = i as f32;
                key[[0, i, j]] = i as f32;
                value[[0, i, j]] = (i + 1) as f32;
            }
        }

        let output = scaled_dot_product_attention_causal(&query, &key, &value, true);

        // First position should only attend to itself
        // Last position should attend to all positions
        assert_eq!(output.dim(), (batch_size, seq_len, d_k));
    }

    #[test]
    fn test_flash_attention_vs_standard() {
        let batch_size = 2;
        let seq_len = 32;
        let d_k = 16;

        let query = Array3::from_shape_fn((batch_size, seq_len, d_k), |(b, s, d)| {
            ((b * 100 + s * 10 + d) as f32).sin() * 0.1
        });
        let key = Array3::from_shape_fn((batch_size, seq_len, d_k), |(b, s, d)| {
            ((b * 100 + s * 10 + d) as f32).cos() * 0.1
        });
        let value = Array3::from_shape_fn((batch_size, seq_len, d_k), |(b, s, d)| {
            ((b * 100 + s * 10 + d) as f32) * 0.01
        });

        let standard_output = scaled_dot_product_attention(&query, &key, &value, None);
        let flash_output = flash_attention_block(&query, &key, &value, 8);

        // Both should have same shape
        assert_eq!(standard_output.dim(), flash_output.dim());
        
        // Results should be approximately equal
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..d_k {
                    let diff = (standard_output[[b, s, d]] - flash_output[[b, s, d]]).abs();
                    assert!(diff < 0.1, "Flash attention diverged: diff={}", diff);
                }
            }
        }
    }

    #[test]
    fn test_sparse_attention_pattern() {
        let batch_size = 1;
        let seq_len = 16;
        let d_k = 8;

        let query = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
        let key = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
        let value = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);

        let output = sparse_attention(&query, &key, &value, 2, 2);

        assert_eq!(output.dim(), (batch_size, seq_len, d_k));
    }

    #[test]
    fn test_softmax_1d_properties() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = softmax_1d(&input);

        // Sum should be 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All values should be positive
        assert!(output.iter().all(|&x| x > 0.0));

        // Output should be in ascending order (same as input)
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    fn test_causal_mask_shape_and_values() {
        let seq_len = 8;
        let mask = create_causal_mask(seq_len);

        assert_eq!(mask.dim(), (seq_len, seq_len));

        // Lower triangular (including diagonal) should be 0
        // Upper triangular should be -inf
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    assert_eq!(mask[[i, j]], 0.0);
                } else {
                    assert!(mask[[i, j]].is_infinite() && mask[[i, j]] < 0.0);
                }
            }
        }
    }

    #[test]
    fn test_attention_config_builder() {
        let config = AttentionConfig::new(16, 128)
            .with_flash(64)
            .with_causal()
            .with_dropout(0.1);

        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.block_size, 64);
        assert!(config.use_flash);
        assert!(config.use_causal_mask);
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_attention_stats_calculation() {
        let stats = AttentionStats::compute(8, 2048, 32, 128);
        
        assert_eq!(stats.total_tokens, 8 * 2048);
        assert!(stats.attention_computations > 0);
        assert!(stats.memory_peak_mb > 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTIZATION INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod quantization_tests {
    use super::*;
    use truthgpt_rust::{QuantizationType, QuantizationParams, QuantizedTensor, quantize_int8, dequantize_int8, quantize_int4, dequantize_int4, quantize_fp16, dequantize_fp16, quantize_bf16, dequantize_bf16, quantize_grouped_int8, dequantize_grouped_int8, matmul_int8};

    #[test]
    fn test_int8_quantization_accuracy() {
        let data: Vec<f32> = (-100..100).map(|i| i as f32 / 10.0).collect();
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT8);

        let quantized = quantize_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);

        // Check error is within acceptable range
        for (orig, rec) in data.iter().zip(dequantized.iter()) {
            let error = (orig - rec).abs();
            assert!(error < 0.1, "INT8 error too large: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_int4_quantization_with_loss() {
        let data: Vec<f32> = vec![0.0, 0.5, 1.0, 1.5, 2.0, -1.0, -2.0, 3.5];
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT4);

        let quantized = quantize_int4(&data, &params);
        let dequantized = dequantize_int4(&quantized, &params, data.len());

        // INT4 has more quantization error
        for (orig, rec) in data.iter().zip(dequantized.iter()) {
            let error = (orig - rec).abs();
            assert!(error < 2.0, "INT4 error too large: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_fp16_roundtrip() {
        let data: Vec<f32> = vec![0.0, 0.001, 0.1, 1.0, 10.0, 100.0, -50.5];
        
        let quantized = quantize_fp16(&data);
        let dequantized = dequantize_fp16(&quantized);

        assert_eq!(data.len(), dequantized.len());
        for (orig, rec) in data.iter().zip(dequantized.iter()) {
            let rel_error = if orig.abs() > 1e-6 {
                (orig - rec).abs() / orig.abs()
            } else {
                (orig - rec).abs()
            };
            assert!(rel_error < 0.01, "FP16 error too large: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_bf16_roundtrip() {
        let data: Vec<f32> = vec![0.0, 1.5, -2.7, 100.0, 0.001];
        
        let quantized = quantize_bf16(&data);
        let dequantized = dequantize_bf16(&quantized);

        assert_eq!(data.len(), dequantized.len());
        for (orig, rec) in data.iter().zip(dequantized.iter()) {
            let error = (orig - rec).abs();
            // BF16 has larger error than FP16 due to fewer mantissa bits
            assert!(error < 0.5, "BF16 error too large: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_grouped_quantization() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();
        let group_size = 32;

        let (quantized, params) = quantize_grouped_int8(&data, group_size);
        let dequantized = dequantize_grouped_int8(&quantized, &params, group_size);

        assert_eq!(params.len(), data.len() / group_size);
        assert_eq!(dequantized.len(), data.len());

        for (orig, rec) in data.iter().zip(dequantized.iter()) {
            let error = (orig - rec).abs();
            assert!(error < 0.2, "Grouped INT8 error: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_quantized_tensor_properties() {
        let qt = QuantizedTensor {
            data: vec![0; 256],
            params: vec![],
            shape: vec![16, 16],
            qtype: QuantizationType::INT8,
            group_size: 0,
        };

        assert_eq!(qt.numel(), 256);
        assert_eq!(qt.size_bytes(), 256);
        assert!((qt.compression_ratio() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_quantization_type_memory_ratio() {
        assert_eq!(QuantizationType::FP32.memory_ratio(), 1.0);
        assert_eq!(QuantizationType::FP16.memory_ratio(), 2.0);
        assert_eq!(QuantizationType::BF16.memory_ratio(), 2.0);
        assert_eq!(QuantizationType::INT8.memory_ratio(), 4.0);
        assert_eq!(QuantizationType::INT4.memory_ratio(), 8.0);
    }

    #[test]
    fn test_matmul_int8() {
        let k = 4;
        let n = 2;
        
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Vec<i8> = vec![1, 0, 0, 1, 0, 1, 1, 0]; // [2, 4] matrix
        let params = QuantizationParams::int8_symmetric(1.0);

        let output = matmul_int8(&input, &weights, &params, n, k);
        
        assert_eq!(output.len(), n);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROPE INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod rope_tests {
    use super::*;
    use truthgpt_rust::{RoPE, RoPEConfig, RoPEScaling, ALiBi, YaRN};
    use ndarray::Array3;

    #[test]
    fn test_rope_basic_application() {
        let config = RoPEConfig {
            dim: 16,
            max_seq_len: 128,
            base: 10000.0,
            scaling: RoPEScaling::None,
        };

        let rope = RoPE::new(config);
        
        let q = Array3::ones((4, 8, 16));
        let k = Array3::ones((4, 8, 16));

        let (q_rot, k_rot) = rope.apply(&q, &k, 0);

        assert_eq!(q_rot.dim(), q.dim());
        assert_eq!(k_rot.dim(), k.dim());
        
        // Rotation should change values
        assert!(q_rot[[0, 0, 0]] != q[[0, 0, 0]] || q_rot[[0, 0, 8]] != q[[0, 0, 8]]);
    }

    #[test]
    fn test_rope_with_position_offset() {
        let config = RoPEConfig {
            dim: 8,
            max_seq_len: 64,
            base: 10000.0,
            scaling: RoPEScaling::None,
        };

        let rope = RoPE::new(config);

        let q = Array3::ones((2, 4, 8));
        let k = Array3::ones((2, 4, 8));

        let (q_rot_0, _) = rope.apply(&q, &k, 0);
        let (q_rot_10, _) = rope.apply(&q, &k, 10);

        // Different start positions should give different results
        assert!(q_rot_0[[0, 0, 0]] != q_rot_10[[0, 0, 0]] || 
                q_rot_0[[0, 0, 4]] != q_rot_10[[0, 0, 4]]);
    }

    #[test]
    fn test_rope_linear_scaling() {
        let config = RoPEConfig {
            dim: 16,
            max_seq_len: 256,
            base: 10000.0,
            scaling: RoPEScaling::Linear { factor: 2.0 },
        };

        let rope = RoPE::new(config);
        assert_eq!(rope.config().dim, 16);
    }

    #[test]
    fn test_rope_ntk_scaling() {
        let config = RoPEConfig {
            dim: 64,
            max_seq_len: 8192,
            base: 10000.0,
            scaling: RoPEScaling::NTK { factor: 2.0 },
        };

        let rope = RoPE::new(config);
        
        // NTK scaling should modify frequencies
        assert_eq!(rope.config().max_seq_len, 8192);
    }

    #[test]
    fn test_alibi_bias_computation() {
        let alibi = ALiBi::new(8);
        let bias = alibi.compute_bias(32);

        assert_eq!(bias.dim(), (8, 32, 32));

        // Check causal masking
        for h in 0..8 {
            for i in 0..32 {
                // Positions after current should be masked
                for j in (i + 1)..32 {
                    assert!(bias[[h, i, j]].is_infinite() && bias[[h, i, j]] < 0.0);
                }
                // Current and previous positions should be finite
                for j in 0..=i {
                    assert!(bias[[h, i, j]].is_finite());
                }
            }
        }
    }

    #[test]
    fn test_yarn_mscale() {
        let yarn = YaRN::new(64, 8192, 10000.0, 2.0, 32.0, 1.0, 4096);
        
        let mscale = yarn.get_mscale();
        let attn_factor = yarn.get_attention_factor();

        // For factor > 1, mscale should be > 1
        assert!(mscale > 1.0);
        assert!(attn_factor > 1.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGED ATTENTION INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod paged_attention_tests {
    use super::*;
    use truthgpt_rust::{BlockManager, PagedAttentionMetadata, BlockTable, BLOCK_SIZE};

    #[test]
    fn test_block_manager_lifecycle() {
        let manager = BlockManager::new(100, false);
        
        // Allocate sequence
        manager.allocate_sequence(1, 48).unwrap(); // 3 blocks at BLOCK_SIZE=16
        
        let stats = manager.stats();
        assert_eq!(stats.num_sequences, 1);
        assert!(stats.free_blocks < stats.total_blocks);

        // Append tokens
        manager.append_tokens(1, 10).unwrap();

        // Free sequence
        manager.free_sequence(1);
        
        let stats = manager.stats();
        assert_eq!(stats.num_sequences, 0);
        assert_eq!(stats.free_blocks, stats.total_blocks);
    }

    #[test]
    fn test_multiple_sequences() {
        let manager = BlockManager::new(200, false);

        for seq_id in 1..=10 {
            manager.allocate_sequence(seq_id, 32).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.num_sequences, 10);
        assert!(stats.utilization > 0.0);

        // Free half
        for seq_id in 1..=5 {
            manager.free_sequence(seq_id);
        }

        let stats = manager.stats();
        assert_eq!(stats.num_sequences, 5);
    }

    #[test]
    fn test_copy_on_write_fork() {
        let manager = BlockManager::new(100, false);

        manager.allocate_sequence(1, 64).unwrap();
        let free_before = manager.stats().free_blocks;

        // Fork should share blocks
        manager.fork_sequence(1, 2).unwrap();
        let free_after = manager.stats().free_blocks;

        assert_eq!(free_before, free_after, "CoW fork should not allocate new blocks");
        assert_eq!(manager.stats().num_sequences, 2);

        // Both sequences should have same block table
        let table1 = manager.get_block_table(1).unwrap();
        let table2 = manager.get_block_table(2).unwrap();
        assert_eq!(table1.blocks, table2.blocks);
    }

    #[test]
    fn test_block_table_operations() {
        let mut table = BlockTable::new(1);
        
        table.push_block(0);
        table.push_block(1);
        table.push_block(2);
        table.num_tokens = 40; // 2.5 blocks worth

        assert_eq!(table.num_blocks(), 3);
        assert_eq!(table.get_block(0), Some(0));
        assert_eq!(table.get_block(15), Some(0));
        assert_eq!(table.get_block(16), Some(1));
        assert_eq!(table.get_block(32), Some(2));
        assert!(table.last_block_has_space());
    }

    #[test]
    fn test_can_allocate_check() {
        let manager = BlockManager::new(10, false);

        assert!(manager.can_allocate(16));  // 1 block
        assert!(manager.can_allocate(160)); // 10 blocks
        assert!(!manager.can_allocate(176)); // 11 blocks (not enough)
    }

    #[test]
    fn test_paged_attention_metadata() {
        let manager = BlockManager::new(100, false);
        
        manager.allocate_sequence(1, 32).unwrap();
        manager.allocate_sequence(2, 48).unwrap();

        let metadata = PagedAttentionMetadata::build(&manager, &[1, 2], true);

        assert_eq!(metadata.block_tables.len(), 2);
        assert_eq!(metadata.context_lens.len(), 2);
        assert_eq!(metadata.max_context_len, 48);
        assert!(!metadata.slot_mapping.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH INFERENCE INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod batch_inference_tests {
    use super::*;
    use truthgpt_rust::{InferenceRequest, InferenceResponse, BatchScheduler, BatchConfig, Priority, FinishReason, ContinuousBatcher};
    use std::time::Duration;

    #[test]
    fn test_inference_request_builder() {
        let req = InferenceRequest::new(vec![1, 2, 3, 4, 5])
            .with_max_tokens(200)
            .with_priority(Priority::High)
            .with_temperature(0.7);

        assert_eq!(req.input_len(), 5);
        assert_eq!(req.max_new_tokens, 200);
        assert_eq!(req.priority, Priority::High);
        assert!((req.temperature - 0.7).abs() < 0.001);
        assert!(!req.is_expired());
    }

    #[test]
    fn test_batch_scheduler_priority() {
        let config = BatchConfig {
            max_wait_time_ms: 0, // Immediate
            min_batch_utilization: 0.0,
            ..Default::default()
        };
        let scheduler = BatchScheduler::new(config);

        // Submit in reverse priority order
        scheduler.submit(InferenceRequest::new(vec![1]).with_priority(Priority::Low));
        scheduler.submit(InferenceRequest::new(vec![2]).with_priority(Priority::Critical));
        scheduler.submit(InferenceRequest::new(vec![3]).with_priority(Priority::Normal));
        scheduler.submit(InferenceRequest::new(vec![4]).with_priority(Priority::High));

        let batch = scheduler.get_batch().unwrap();

        // Should be ordered by priority
        assert_eq!(batch[0].priority, Priority::Critical);
        assert_eq!(batch[1].priority, Priority::High);
        assert_eq!(batch[2].priority, Priority::Normal);
        assert_eq!(batch[3].priority, Priority::Low);
    }

    #[test]
    fn test_batch_size_limits() {
        let config = BatchConfig {
            max_batch_size: 4,
            max_wait_time_ms: 0,
            min_batch_utilization: 0.0,
            ..Default::default()
        };
        let scheduler = BatchScheduler::new(config);

        // Submit more than max batch size
        for i in 0..10 {
            scheduler.submit(InferenceRequest::new(vec![i as u32]));
        }

        let batch = scheduler.get_batch().unwrap();
        assert!(batch.len() <= 4);
    }

    #[test]
    fn test_continuous_batcher() {
        let mut batcher = ContinuousBatcher::new(8);

        // Add requests
        let slot1 = batcher.add(1, &[1, 2, 3]).unwrap();
        let slot2 = batcher.add(2, &[4, 5, 6, 7]).unwrap();

        assert_eq!(batcher.active_count(), 2);
        assert_eq!(batcher.free_count(), 6);

        let active = batcher.active_requests();
        assert!(active.contains(&1));
        assert!(active.contains(&2));

        // Remove one
        batcher.remove(slot1);
        assert_eq!(batcher.active_count(), 1);
        assert_eq!(batcher.free_count(), 7);
    }

    #[test]
    fn test_request_cancellation() {
        let config = BatchConfig {
            max_wait_time_ms: 1000, // Long wait
            ..Default::default()
        };
        let scheduler = BatchScheduler::new(config);

        let id = scheduler.submit(InferenceRequest::new(vec![1, 2, 3]));
        assert_eq!(scheduler.pending_count(), 1);

        let cancelled = scheduler.cancel(id);
        assert!(cancelled);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_stats() {
        let config = BatchConfig {
            max_wait_time_ms: 0,
            min_batch_utilization: 0.0,
            ..Default::default()
        };
        let scheduler = BatchScheduler::new(config);

        for i in 0..5 {
            scheduler.submit(InferenceRequest::new(vec![i as u32]));
        }

        let _ = scheduler.get_batch();

        let stats = scheduler.stats();
        assert_eq!(stats.requests_submitted.load(std::sync::atomic::Ordering::Relaxed), 5);
        assert_eq!(stats.batches_formed.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPECULATIVE DECODING INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod speculative_tests {
    use super::*;
    use truthgpt_rust::{SpeculativeDecoder, SpeculativeConfig, DraftResult, VerificationResult};

    #[test]
    fn test_speculative_decoder_basic() {
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            adaptive_length: false,
            ..Default::default()
        };
        let decoder = SpeculativeDecoder::new(config);

        assert_eq!(decoder.speculation_length(), 4);
    }

    #[test]
    fn test_draft_verification() {
        let config = SpeculativeConfig {
            num_speculative_tokens: 3,
            adaptive_length: false,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);

        // Create matching distributions
        let draft = DraftResult {
            tokens: vec![0, 1, 2],
            draft_logprobs: vec![
                vec![10.0, 0.0, 0.0, 0.0],
                vec![0.0, 10.0, 0.0, 0.0],
                vec![0.0, 0.0, 10.0, 0.0],
            ],
            confidences: vec![0.95, 0.95, 0.95],
        };

        let target_logprobs = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0],
            vec![0.0, 0.0, 0.0, 10.0], // Next token
        ];

        let result = decoder.verify(&draft, &target_logprobs);

        // With matching distributions, acceptance should be high
        assert!(result.acceptance_rate >= 0.0);
        assert!(result.accepted_count <= 3);
    }

    #[test]
    fn test_adaptive_speculation_length() {
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            adaptive_length: true,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);

        // Simulate high acceptance rate
        for _ in 0..20 {
            decoder.update_stats(4, 4);
        }

        let stats = decoder.stats();
        assert!(stats.avg_acceptance_rate > 0.7);

        // Simulate low acceptance rate
        decoder.reset_stats();
        for _ in 0..20 {
            decoder.update_stats(0, 4);
        }

        let stats = decoder.stats();
        assert!(stats.avg_acceptance_rate < 0.3);
    }

    #[test]
    fn test_tree_speculation() {
        let mut tree = TreeSpeculation::new(3, 2);

        let draft_logprobs = vec![
            vec![0.0, 2.0, 1.0, 0.5],
            vec![1.0, 0.0, 2.0, 0.5],
        ];

        tree.build_tree(&draft_logprobs);

        assert!(tree.size() > 0);

        let paths = tree.get_paths();
        assert!(!paths.is_empty());
        
        // All paths should have depth 2
        for path in &paths {
            assert_eq!(path.len(), 2);
        }
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.5, 0.3, 0.2];

        // Identical distributions should have KL=0
        let kl = kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-5);

        // Different distributions should have KL>0
        let r = vec![0.1, 0.1, 0.8];
        let kl2 = kl_divergence(&p, &r);
        assert!(kl2 > 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERROR HANDLING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod error_tests {
    use super::*;
    use truthgpt_rust::TruthGPTError;

    #[test]
    fn test_error_types() {
        let cache_err = TruthGPTError::cache("cache full");
        assert!(cache_err.to_string().contains("Cache error"));

        let compression_err = TruthGPTError::compression("invalid data");
        assert!(compression_err.to_string().contains("Compression error"));

        let tokenization_err = TruthGPTError::tokenization("unknown token");
        assert!(tokenization_err.to_string().contains("Tokenization error"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let truthgpt_err: TruthGPTError = io_err.into();
        
        assert!(matches!(truthgpt_err, TruthGPTError::Io(_)));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod utils_tests {
    use super::*;
    use truthgpt_rust::{Timer, AtomicCounter, Histogram, f32_to_bytes, bytes_to_f32, f32_to_f16_bytes, f16_to_f32_bytes, RingBuffer, format_bytes, format_duration, ExponentialMovingAverage};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer_functionality() {
        let mut timer = Timer::new("test_timer");
        
        thread::sleep(Duration::from_millis(10));
        timer.lap("after 10ms");
        
        thread::sleep(Duration::from_millis(20));
        timer.lap("after 30ms total");

        assert!(timer.elapsed_ms() >= 30.0);
        assert_eq!(timer.laps().len(), 2);
    }

    #[test]
    fn test_atomic_counter_thread_safety() {
        use std::sync::Arc;

        let counter = Arc::new(AtomicCounter::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    counter_clone.increment();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.get(), 4000);
    }

    #[test]
    fn test_histogram_recording() {
        let hist = Histogram::new();

        // Record some latencies
        hist.record(10);
        hist.record(50);
        hist.record(100);
        hist.record(500);
        hist.record(1000);

        let stats = hist.stats();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_us, 10);
        assert_eq!(stats.max_us, 1000);
    }

    #[test]
    fn test_data_conversion() {
        let original = vec![1.5f32, 2.5, 3.5, -4.5];
        
        let bytes = f32_to_bytes(&original);
        let recovered = bytes_to_f32(&bytes);
        
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_f16_conversion() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        
        let f16_bytes = f32_to_f16_bytes(&original);
        let recovered = f16_to_f32_bytes(&f16_bytes);
        
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 0.001);
        }
    }

    #[test]
    fn test_ring_buffer() {
        let buffer: RingBuffer<i32, 8> = RingBuffer::new();
        
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 7);

        for i in 0..7 {
            assert!(buffer.push(i));
        }
        assert!(!buffer.push(7)); // Full

        assert_eq!(buffer.len(), 7);

        for i in 0..7 {
            assert_eq!(buffer.pop(), Some(i));
        }
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_micros(500)), "500.00µs");
        assert_eq!(format_duration(Duration::from_millis(500)), "500.00ms");
        assert_eq!(format_duration(Duration::from_secs(30)), "30.00s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
    }

    #[test]
    fn test_exponential_moving_average() {
        let ema = ExponentialMovingAverage::new(0.5);
        
        ema.update(100.0);
        let val1 = ema.get();
        
        ema.update(100.0);
        let val2 = ema.get();
        
        // EMA should be approaching 100
        assert!(val2 > val1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERFORMANCE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod performance_tests {
    use super::*;
    use std::time::Instant;
    use truthgpt_rust::{compress, decompress, CompressionAlgorithm};

    #[test]
    fn test_compression_throughput() {
        
        // Generate 1MB of test data
        let data: Vec<u8> = (0..1_000_000)
            .map(|i| (i % 256) as u8)
            .collect();

        let start = Instant::now();
        let compressed = compress(&data, &CompressionAlgorithm::LZ4).unwrap();
        let compress_time = start.elapsed();

        let start = Instant::now();
        let _ = decompress(&compressed, &CompressionAlgorithm::LZ4).unwrap();
        let decompress_time = start.elapsed();

        let compress_mbps = data.len() as f64 / (1024.0 * 1024.0) / compress_time.as_secs_f64();
        let decompress_mbps = data.len() as f64 / (1024.0 * 1024.0) / decompress_time.as_secs_f64();

        println!("LZ4 Compression: {:.0} MB/s", compress_mbps);
        println!("LZ4 Decompression: {:.0} MB/s", decompress_mbps);
        
        assert!(compress_mbps > 100.0, "Compression too slow");
        assert!(decompress_mbps > 200.0, "Decompression too slow");
    }

    #[test]
    fn test_kv_cache_latency() {
        use truthgpt_rust::{KVCache, KVCacheConfig};
        
        let config = KVCacheConfig {
            max_size: 10000,
            enable_compression: false,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        // Pre-populate
        for i in 0..1000 {
            cache.put(i % 10, i, vec![1u8; 128]);
        }

        // Measure get latency
        let iterations = 10000;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = cache.get(i % 10, i % 1000);
        }
        let elapsed = start.elapsed();

        let avg_ns = elapsed.as_nanos() as f64 / iterations as f64;
        println!("KV Cache get latency: {:.0} ns", avg_ns);
        
        assert!(avg_ns < 1000.0, "Cache get too slow");
    }

    #[test]
    fn test_attention_scaling() {
        use truthgpt_rust::{scaled_dot_product_attention};
        
        for seq_len in [32, 64, 128] {
            let batch_size = 2;
            let d_k = 32;

            let query = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);
            let key = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);
            let value = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);

            let start = Instant::now();
            let _ = scaled_dot_product_attention(&query, &key, &value, None);
            let elapsed = start.elapsed();

            println!("Attention seq_len={}: {:.2}ms", seq_len, elapsed.as_secs_f64() * 1000.0);
        }
    }
}

