//! Edge Case Tests for TruthGPT Rust Core
//!
//! Tests for boundary conditions, empty inputs, and edge cases.

use truthgpt_rust::*;

// ═══════════════════════════════════════════════════════════════════════════════
// KV CACHE EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod kv_cache_edge_cases {
    use truthgpt_rust::{KVCache, KVCacheConfig, ConcurrentKVCache};

    #[test]
    fn test_cache_size_one() {
        let config = KVCacheConfig {
            max_size: 1,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        cache.put(0, 0, vec![1]);
        assert_eq!(cache.get(0, 0), Some(&[1u8][..]));

        cache.put(0, 1, vec![2]);
        assert_eq!(cache.get(0, 1), Some(&[2u8][..]));
        assert!(cache.get(0, 0).is_none(), "Old entry should be evicted");
    }

    #[test]
    fn test_empty_data_storage() {
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config);

        cache.put(0, 0, vec![]);
        let result = cache.get(0, 0);
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_large_key_values() {
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config);

        // Large layer and position indices
        cache.put(usize::MAX - 1, usize::MAX - 1, vec![42]);
        assert_eq!(cache.get(usize::MAX - 1, usize::MAX - 1), Some(&[42u8][..]));
    }

    #[test]
    fn test_overwrite_existing_key() {
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config);

        cache.put(0, 0, vec![1, 2, 3]);
        cache.put(0, 0, vec![4, 5, 6]); // Overwrite

        assert_eq!(cache.get(0, 0), Some(&[4, 5, 6][..]));
    }

    #[test]
    fn test_remove_nonexistent() {
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config);

        let result = cache.remove(999, 999);
        assert!(result.is_none());
    }

    #[test]
    fn test_clear_empty_cache() {
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config);

        cache.clear(); // Should not panic
        assert!(cache.is_empty());
        assert_eq!(cache.hit_rate(), 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPRESSION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod compression_edge_cases {
    use truthgpt_rust::{compress, decompress, compress_with_stats, compress_zstd_level, CompressionAlgorithm, StreamingCompressor};

    #[test]
    fn test_compress_empty_data() {
        let data: Vec<u8> = vec![];
        
        let compressed_lz4 = compress(&data, &CompressionAlgorithm::LZ4).unwrap();
        let decompressed_lz4 = decompress(&compressed_lz4, &CompressionAlgorithm::LZ4).unwrap();
        assert_eq!(data, decompressed_lz4);

        let compressed_zstd = compress(&data, &CompressionAlgorithm::Zstd).unwrap();
        let decompressed_zstd = decompress(&compressed_zstd, &CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed_zstd);
    }

    #[test]
    fn test_compress_single_byte() {
        let data = vec![42u8];
        
        let compressed = compress(&data, &CompressionAlgorithm::LZ4).unwrap();
        let decompressed = decompress(&compressed, &CompressionAlgorithm::LZ4).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compress_incompressible_data() {
        // Random-looking data that doesn't compress well
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let data: Vec<u8> = (0..1000u64)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                h.finish() as u8
            })
            .collect();

        let compressed = compress(&data, &CompressionAlgorithm::LZ4).unwrap();
        let decompressed = decompress(&compressed, &CompressionAlgorithm::LZ4).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compression_stats_empty_data() {
        let data: Vec<u8> = vec![];
        let (_, stats) = compress_with_stats(&data, &CompressionAlgorithm::LZ4).unwrap();
        
        assert_eq!(stats.original_size, 0);
        assert_eq!(stats.compression_ratio(), 0.0);
        assert_eq!(stats.space_savings(), 1.0);
    }

    #[test]
    fn test_streaming_compressor_partial_chunk() {
        let mut compressor = StreamingCompressor::new(CompressionAlgorithm::LZ4, 100);

        // Write less than chunk size
        let chunks = compressor.write(&[1, 2, 3]).unwrap();
        assert!(chunks.is_empty()); // Not enough data for a chunk

        let final_chunk = compressor.flush().unwrap();
        assert!(final_chunk.is_some());
    }

    #[test]
    fn test_zstd_level_clamping() {
        let data = b"test data".to_vec();
        
        // Level below minimum
        let compressed_low = compress_zstd_level(&data, -100).unwrap();
        let decompressed = decompress(&compressed_low, &CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed);

        // Level above maximum
        let compressed_high = compress_zstd_level(&data, 100).unwrap();
        let decompressed = decompress(&compressed_high, &CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod attention_edge_cases {
    use truthgpt_rust::{scaled_dot_product_attention, flash_attention_block, sparse_attention, create_causal_mask, softmax_1d};
    use ndarray::Array3;

    #[test]
    fn test_attention_batch_size_one() {
        let query = Array3::from_elem((1, 4, 8), 0.1f32);
        let key = Array3::from_elem((1, 4, 8), 0.1f32);
        let value = Array3::from_elem((1, 4, 8), 0.1f32);

        let output = scaled_dot_product_attention(&query, &key, &value, None);
        assert_eq!(output.dim(), (1, 4, 8));
    }

    #[test]
    fn test_attention_seq_len_one() {
        let query = Array3::from_elem((2, 1, 8), 0.1f32);
        let key = Array3::from_elem((2, 1, 8), 0.1f32);
        let value = Array3::from_elem((2, 1, 8), 0.1f32);

        let output = scaled_dot_product_attention(&query, &key, &value, None);
        assert_eq!(output.dim(), (2, 1, 8));
    }

    #[test]
    fn test_attention_dim_one() {
        let query = Array3::from_elem((2, 4, 1), 0.1f32);
        let key = Array3::from_elem((2, 4, 1), 0.1f32);
        let value = Array3::from_elem((2, 4, 1), 0.1f32);

        let output = scaled_dot_product_attention(&query, &key, &value, None);
        assert_eq!(output.dim(), (2, 4, 1));
    }

    #[test]
    fn test_attention_with_all_zeros() {
        let query = Array3::zeros((2, 4, 8));
        let key = Array3::zeros((2, 4, 8));
        let value = Array3::from_elem((2, 4, 8), 1.0f32);

        let output = scaled_dot_product_attention(&query, &key, &value, None);
        
        // With zero Q and K, all scores are equal, so output should be uniform average of values
        assert_eq!(output.dim(), (2, 4, 8));
    }

    #[test]
    fn test_softmax_empty() {
        let input: Vec<f32> = vec![];
        let output = softmax_1d(&input);
        assert!(output.is_empty());
    }

    #[test]
    fn test_softmax_single_element() {
        let input = vec![5.0];
        let output = softmax_1d(&input);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_extreme_values() {
        // Large values should not overflow
        let input = vec![100.0, 200.0, 300.0];
        let output = softmax_1d(&input);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Negative values
        let input = vec![-100.0, -200.0, -300.0];
        let output = softmax_1d(&input);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_causal_mask_size_one() {
        let mask = create_causal_mask(1);
        assert_eq!(mask.dim(), (1, 1));
        assert_eq!(mask[[0, 0]], 0.0);
    }

    #[test]
    fn test_sparse_attention_window_larger_than_seq() {
        let query = Array3::from_elem((1, 8, 4), 1.0f32);
        let key = Array3::from_elem((1, 8, 4), 1.0f32);
        let value = Array3::from_elem((1, 8, 4), 1.0f32);

        // Window larger than sequence - should still work
        let output = sparse_attention(&query, &key, &value, 100, 0);
        assert_eq!(output.dim(), (1, 8, 4));
    }

    #[test]
    fn test_flash_attention_small_block() {
        let query = Array3::from_elem((2, 16, 8), 0.1f32);
        let key = Array3::from_elem((2, 16, 8), 0.1f32);
        let value = Array3::from_elem((2, 16, 8), 0.1f32);

        // Block size of 1
        let output = flash_attention_block(&query, &key, &value, 1);
        assert_eq!(output.dim(), (2, 16, 8));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTIZATION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod quantization_edge_cases {
    use truthgpt_rust::{QuantizationType, QuantizationParams, quantize_int8, dequantize_int8, quantize_int4, dequantize_int4, quantize_fp16, dequantize_fp16, quantize_bf16, dequantize_bf16, quantize_grouped_int8, dequantize_grouped_int8};

    #[test]
    fn test_quantize_single_value() {
        let data = vec![1.0f32];
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT8);
        
        let quantized = quantize_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);
        
        assert_eq!(dequantized.len(), 1);
    }

    #[test]
    fn test_quantize_all_zeros() {
        let data = vec![0.0f32; 100];
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT8);
        
        let quantized = quantize_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);
        
        for val in &dequantized {
            assert!((val - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_quantize_same_value() {
        let data = vec![5.0f32; 50];
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT8);
        
        let quantized = quantize_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);
        
        for val in &dequantized {
            assert!((val - 5.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_int4_odd_length() {
        let data = vec![1.0f32, 2.0, 3.0]; // Odd length
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT4);
        
        let quantized = quantize_int4(&data, &params);
        let dequantized = dequantize_int4(&quantized, &params, data.len());
        
        assert_eq!(dequantized.len(), 3);
    }

    #[test]
    fn test_fp16_special_values() {
        let data = vec![0.0f32, f32::MIN_POSITIVE, 65504.0]; // Max FP16 value approx
        
        let quantized = quantize_fp16(&data);
        let dequantized = dequantize_fp16(&quantized);
        
        assert_eq!(dequantized.len(), 3);
    }

    #[test]
    fn test_empty_quantization() {
        let data: Vec<f32> = vec![];
        
        let fp16 = quantize_fp16(&data);
        assert!(fp16.is_empty());
        
        let bf16 = quantize_bf16(&data);
        assert!(bf16.is_empty());
    }

    #[test]
    fn test_grouped_quantization_uneven_groups() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let group_size = 32; // 100 / 32 = 3 groups with remainder

        let (quantized, params) = quantize_grouped_int8(&data, group_size);
        let dequantized = dequantize_grouped_int8(&quantized, &params, group_size);

        assert_eq!(dequantized.len(), data.len());
        assert_eq!(params.len(), 4); // ceil(100/32) = 4
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROPE EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod rope_edge_cases {
    use truthgpt_rust::{RoPE, RoPEConfig, RoPEScaling, ALiBi};
    use ndarray::Array3;

    #[test]
    fn test_rope_position_at_max() {
        let config = RoPEConfig {
            dim: 8,
            max_seq_len: 16,
            base: 10000.0,
            scaling: RoPEScaling::None,
        };
        let rope = RoPE::new(config);

        let q = Array3::ones((1, 4, 8));
        let k = Array3::ones((1, 4, 8));

        // Start position near max
        let (q_rot, k_rot) = rope.apply(&q, &k, 14);
        assert_eq!(q_rot.dim(), (1, 4, 8));
        assert_eq!(k_rot.dim(), (1, 4, 8));
    }

    #[test]
    fn test_rope_dim_two() {
        let config = RoPEConfig {
            dim: 2, // Minimum valid
            max_seq_len: 32,
            base: 10000.0,
            scaling: RoPEScaling::None,
        };
        let rope = RoPE::new(config);

        let q = Array3::ones((1, 4, 2));
        let k = Array3::ones((1, 4, 2));

        let (q_rot, k_rot) = rope.apply(&q, &k, 0);
        assert_eq!(q_rot.dim(), (1, 4, 2));
    }

    #[test]
    fn test_alibi_single_head() {
        let alibi = ALiBi::new(1);
        let bias = alibi.compute_bias(4);

        assert_eq!(bias.dim(), (1, 4, 4));
    }

    #[test]
    fn test_alibi_large_seq_len() {
        let alibi = ALiBi::new(4);
        let bias = alibi.compute_bias(256);

        assert_eq!(bias.dim(), (4, 256, 256));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGED ATTENTION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod paged_attention_edge_cases {
    use truthgpt_rust::{BlockManager, BlockTable, BLOCK_SIZE};

    #[test]
    fn test_allocate_single_token() {
        let manager = BlockManager::new(100, false);
        manager.allocate_sequence(1, 1).unwrap();
        
        let table = manager.get_block_table(1).unwrap();
        assert_eq!(table.num_blocks(), 1);
    }

    #[test]
    fn test_allocate_exact_block_boundary() {
        let manager = BlockManager::new(100, false);
        manager.allocate_sequence(1, BLOCK_SIZE).unwrap();
        
        let table = manager.get_block_table(1).unwrap();
        assert_eq!(table.num_blocks(), 1);

        manager.allocate_sequence(2, BLOCK_SIZE + 1).unwrap();
        let table2 = manager.get_block_table(2).unwrap();
        assert_eq!(table2.num_blocks(), 2);
    }

    #[test]
    fn test_append_no_new_block_needed() {
        let manager = BlockManager::new(100, false);
        manager.allocate_sequence(1, 1).unwrap();
        
        let free_before = manager.stats().free_blocks;
        manager.append_tokens(1, BLOCK_SIZE - 2).unwrap();
        let free_after = manager.stats().free_blocks;

        assert_eq!(free_before, free_after);
    }

    #[test]
    fn test_free_nonexistent_sequence() {
        let manager = BlockManager::new(100, false);
        manager.free_sequence(999); // Should not panic
    }

    #[test]
    fn test_get_nonexistent_block_table() {
        let manager = BlockManager::new(100, false);
        assert!(manager.get_block_table(999).is_none());
    }

    #[test]
    fn test_block_table_empty() {
        let table = BlockTable::new(1);
        
        assert_eq!(table.num_blocks(), 0);
        assert!(table.get_block(0).is_none());
        assert!(!table.last_block_has_space());
    }

    #[test]
    fn test_allocate_all_blocks() {
        let manager = BlockManager::new(10, false);
        
        // Allocate all blocks
        manager.allocate_sequence(1, BLOCK_SIZE * 10).unwrap();
        
        // Should not be able to allocate more
        let result = manager.allocate_sequence(2, 1);
        assert!(result.is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH INFERENCE EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod batch_inference_edge_cases {
    use truthgpt_rust::{InferenceRequest, BatchScheduler, BatchConfig, Priority, ContinuousBatcher};

    #[test]
    fn test_empty_input_tokens() {
        let req = InferenceRequest::new(vec![]);
        assert_eq!(req.input_len(), 0);
    }

    #[test]
    fn test_scheduler_no_pending() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        assert!(scheduler.get_batch().is_none());
    }

    #[test]
    fn test_cancel_nonexistent_request() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        assert!(!scheduler.cancel(999999));
    }

    #[test]
    fn test_continuous_batcher_full() {
        let mut batcher = ContinuousBatcher::new(2);
        
        assert!(batcher.add(1, &[1, 2]).is_some());
        assert!(batcher.add(2, &[3, 4]).is_some());
        assert!(batcher.add(3, &[5, 6]).is_none()); // Full
    }

    #[test]
    fn test_request_with_zero_max_tokens() {
        let req = InferenceRequest::new(vec![1, 2, 3])
            .with_max_tokens(0);
        assert_eq!(req.max_new_tokens, 0);
    }

    #[test]
    fn test_temperature_clamping() {
        let req_low = InferenceRequest::new(vec![1])
            .with_temperature(-5.0);
        assert!(req_low.temperature >= 0.0);

        let req_high = InferenceRequest::new(vec![1])
            .with_temperature(100.0);
        assert!(req_high.temperature <= 2.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPECULATIVE DECODING EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod speculative_edge_cases {
    use truthgpt_rust::{SpeculativeDecoder, SpeculativeConfig, DraftResult};

    #[test]
    fn test_empty_draft() {
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            adaptive_length: false,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);

        let draft = DraftResult {
            tokens: vec![],
            draft_logprobs: vec![],
            confidences: vec![],
        };

        let target_logprobs: Vec<Vec<f32>> = vec![];
        let result = decoder.verify(&draft, &target_logprobs);

        assert_eq!(result.accepted_count, 0);
    }

    #[test]
    fn test_tree_speculation_empty_logprobs() {
        use truthgpt_rust::TreeSpeculation;
        let mut tree = TreeSpeculation::new(2, 3);
        tree.build_tree(&[]);
        
        // Should have just root node
        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn test_kl_divergence_identical() {
        use truthgpt_rust::kl_divergence;
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        
        let kl = kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn test_speculation_length_minimum() {
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            adaptive_length: true,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);

        // Force very low acceptance
        for _ in 0..100 {
            decoder.update_stats(0, 4);
        }

        // Should not go below 1 (use public API)
        let stats = decoder.stats();
        assert!(stats.current_spec_length >= 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILS EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod utils_edge_cases {
    use truthgpt_rust::{Timer, AtomicCounter, Histogram, f32_to_bytes, bytes_to_f32, RingBuffer, format_bytes, format_duration, ExponentialMovingAverage};
    use std::time::Duration;

    #[test]
    fn test_timer_no_laps() {
        let timer = Timer::new("test");
        let summary = timer.summary();
        assert!(summary.contains("test"));
    }

    #[test]
    fn test_counter_overflow_handling() {
        let counter = AtomicCounter::with_value(u64::MAX - 1);
        counter.increment();
        assert_eq!(counter.get(), u64::MAX);
        
        // Overflow wraps around
        counter.increment();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_histogram_single_value() {
        let hist = Histogram::new();
        hist.record(42);
        
        let stats = hist.stats();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.min_us, 42);
        assert_eq!(stats.max_us, 42);
    }

    #[test]
    fn test_histogram_empty() {
        let hist = Histogram::new();
        let stats = hist.stats();
        
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_empty_bytes_conversion() {
        let original: Vec<f32> = vec![];
        let bytes = f32_to_bytes(&original);
        let recovered = bytes_to_f32(&bytes);
        
        assert!(bytes.is_empty());
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_ring_buffer_single_element() {
        let buffer: RingBuffer<i32, 2> = RingBuffer::new();
        
        assert!(buffer.push(1));
        assert!(!buffer.push(2)); // Capacity is N-1 = 1
        
        assert_eq!(buffer.pop(), Some(1));
    }

    #[test]
    fn test_format_bytes_edge_values() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1), "1.00 B");
        
        // Very large values
        let petabyte = 1024u64 * 1024 * 1024 * 1024 * 1024;
        let result = format_bytes(petabyte);
        assert!(result.contains("PB"));
    }

    #[test]
    fn test_format_duration_microseconds() {
        let result = format_duration(Duration::from_micros(1));
        assert!(result.contains("µs"));
    }

    #[test]
    fn test_ema_zero_alpha() {
        let ema = ExponentialMovingAverage::new(0.0);
        ema.update(100.0);
        // With alpha=0, value should stay at 0
        assert_eq!(ema.get(), 0.0);
    }

    #[test]
    fn test_ema_one_alpha() {
        let ema = ExponentialMovingAverage::new(1.0);
        ema.update(100.0);
        // With alpha=1, value should be exactly the new value
        assert_eq!(ema.get(), 100.0);
    }
}

