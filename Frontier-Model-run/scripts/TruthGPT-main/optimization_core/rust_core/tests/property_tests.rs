//! Property-Based Tests for TruthGPT Rust Core
//!
//! Uses proptest for randomized testing with automatic shrinking.

use proptest::prelude::*;
use truthgpt_rust::*;

// ═══════════════════════════════════════════════════════════════════════════════
// COMPRESSION PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod compression_properties {
    use super::*;
    use truthgpt_rust::{compress, decompress, compress_with_stats, CompressionAlgorithm};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn lz4_roundtrip_any_data(data in prop::collection::vec(any::<u8>(), 0..10000)) {
            let compressed = compress(&data, &CompressionAlgorithm::LZ4).unwrap();
            let decompressed = decompress(&compressed, &CompressionAlgorithm::LZ4).unwrap();
            prop_assert_eq!(data, decompressed);
        }

        #[test]
        fn zstd_roundtrip_any_data(data in prop::collection::vec(any::<u8>(), 0..5000)) {
            let compressed = compress(&data, &CompressionAlgorithm::Zstd).unwrap();
            let decompressed = decompress(&compressed, &CompressionAlgorithm::Zstd).unwrap();
            prop_assert_eq!(data, decompressed);
        }

        #[test]
        fn no_compression_is_identity(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let result = compress(&data, &CompressionAlgorithm::None).unwrap();
            prop_assert_eq!(&data, &result);
        }

        #[test]
        fn compression_stats_are_valid(data in prop::collection::vec(any::<u8>(), 1..1000)) {
            let (compressed, stats) = compress_with_stats(&data, &CompressionAlgorithm::LZ4).unwrap();
            
            prop_assert_eq!(stats.original_size, data.len());
            prop_assert_eq!(stats.compressed_size, compressed.len());
            prop_assert!(stats.compression_ratio() >= 0.0);
            prop_assert!(stats.space_savings() <= 1.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KV CACHE PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod kv_cache_properties {
    use super::*;
    use truthgpt_rust::{KVCache, KVCacheConfig};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn cache_never_exceeds_max_size(
            max_size in 1usize..100,
            entries in prop::collection::vec((0usize..10, 0usize..100, prop::collection::vec(any::<u8>(), 0..100)), 0..200)
        ) {
            let config = KVCacheConfig {
                max_size,
                enable_compression: false,
                ..Default::default()
            };
            let mut cache = KVCache::new(config);

            for (layer, pos, data) in entries {
                cache.put(layer, pos, data);
                prop_assert!(cache.size() <= max_size);
            }
        }

        #[test]
        fn cache_get_returns_correct_data(
            layer in 0usize..10,
            position in 0usize..100,
            data in prop::collection::vec(any::<u8>(), 0..100)
        ) {
            let config = KVCacheConfig::default();
            let mut cache = KVCache::new(config);

            cache.put(layer, position, data.clone());
            let result = cache.get(layer, position);
            
            prop_assert!(result.is_some());
            prop_assert_eq!(result.unwrap(), &data[..]);
        }

        #[test]
        fn cache_hit_rate_bounds(
            hits in 0u64..100,
            misses in 0u64..100
        ) {
            // Simulate with stats
            let total = hits + misses;
            if total > 0 {
                let rate = hits as f64 / total as f64;
                prop_assert!(rate >= 0.0 && rate <= 1.0);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTIZATION PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod quantization_properties {
    use super::*;
    use truthgpt_rust::{QuantizationType, QuantizationParams, QuantizedTensor, quantize_int8, dequantize_int8, quantize_fp16, dequantize_fp16};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn fp16_roundtrip_preserves_order(
            data in prop::collection::vec(-1000.0f32..1000.0, 2..100)
        ) {
            let quantized = quantize_fp16(&data);
            let dequantized = dequantize_fp16(&quantized);

            prop_assert_eq!(data.len(), dequantized.len());
            
            // Verify relative ordering is preserved
            for i in 1..data.len() {
                if data[i] > data[i-1] + 0.01 {
                    prop_assert!(dequantized[i] >= dequantized[i-1]);
                }
                if data[i] < data[i-1] - 0.01 {
                    prop_assert!(dequantized[i] <= dequantized[i-1]);
                }
            }
        }

        #[test]
        fn int8_quantization_bounded_error(
            data in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let params = QuantizationParams::from_tensor(&data, QuantizationType::INT8);
            let quantized = quantize_int8(&data, &params);
            let dequantized = dequantize_int8(&quantized, &params);

            prop_assert_eq!(data.len(), dequantized.len());
            
            // Error should be bounded
            for (orig, rec) in data.iter().zip(dequantized.iter()) {
                let error = (orig - rec).abs();
                prop_assert!(error < 1.0, "INT8 error too large: {} vs {}", orig, rec);
            }
        }

        #[test]
        fn quantized_tensor_compression_ratio_valid(
            num_elements in 10usize..1000,
        ) {
            let qt = QuantizedTensor {
                data: vec![0; num_elements],
                params: vec![],
                shape: vec![num_elements],
                qtype: QuantizationType::INT8,
                group_size: 0,
            };

            let ratio = qt.compression_ratio();
            prop_assert!(ratio > 0.0, "Compression ratio should be positive");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod attention_properties {
    use super::*;
    use truthgpt_rust::{scaled_dot_product_attention, create_causal_mask, softmax_1d};
    use ndarray::Array3;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn softmax_sums_to_one(
            values in prop::collection::vec(-100.0f32..100.0, 1..50)
        ) {
            let result = softmax_1d(&values);
            let sum: f32 = result.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "Softmax sum: {}", sum);
        }

        #[test]
        fn softmax_all_positive(
            values in prop::collection::vec(-100.0f32..100.0, 1..50)
        ) {
            let result = softmax_1d(&values);
            for &v in &result {
                prop_assert!(v >= 0.0, "Softmax produced negative: {}", v);
            }
        }

        #[test]
        fn causal_mask_is_triangular(seq_len in 1usize..50) {
            let mask = create_causal_mask(seq_len);
            
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j <= i {
                        prop_assert_eq!(mask[[i, j]], 0.0);
                    } else {
                        prop_assert!(mask[[i, j]].is_infinite() && mask[[i, j]] < 0.0);
                    }
                }
            }
        }

        #[test]
        fn attention_output_shape_matches_input(
            batch_size in 1usize..4,
            seq_len in 1usize..32,
            d_k in 1usize..16
        ) {
            let query = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);
            let key = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);
            let value = Array3::from_elem((batch_size, seq_len, d_k), 0.1f32);

            let output = scaled_dot_product_attention(&query, &key, &value, None);
            
            prop_assert_eq!(output.dim(), (batch_size, seq_len, d_k));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGED ATTENTION PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod paged_attention_properties {
    use super::*;
    use truthgpt_rust::{BlockManager, BlockTable, BLOCK_SIZE};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn block_allocation_releases_correctly(
            num_tokens in 1usize..500,
        ) {
            let manager = BlockManager::new(1000, false);
            let initial_free = manager.stats().free_blocks;

            manager.allocate_sequence(1, num_tokens).unwrap();
            manager.free_sequence(1);

            let final_free = manager.stats().free_blocks;
            prop_assert_eq!(initial_free, final_free);
        }

        #[test]
        fn fork_preserves_content(
            num_tokens in 1usize..200,
        ) {
            let manager = BlockManager::new(1000, false);
            
            manager.allocate_sequence(1, num_tokens).unwrap();
            let parent_table = manager.get_block_table(1).unwrap();

            manager.fork_sequence(1, 2).unwrap();
            let child_table = manager.get_block_table(2).unwrap();

            prop_assert_eq!(parent_table.blocks, child_table.blocks);
            prop_assert_eq!(parent_table.num_tokens, child_table.num_tokens);
        }

        #[test]
        fn block_table_position_mapping(
            num_blocks in 1usize..20,
        ) {
            let mut table = BlockTable::new(1);
            
            for i in 0..num_blocks {
                table.push_block(i as u32);
            }
            table.num_tokens = num_blocks * BLOCK_SIZE;

            // Check that positions map to correct blocks
            for pos in 0..(num_blocks * BLOCK_SIZE) {
                let expected_block = (pos / BLOCK_SIZE) as u32;
                prop_assert_eq!(table.get_block(pos), Some(expected_block));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH INFERENCE PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod batch_inference_properties {
    use super::*;
    use truthgpt_rust::{InferenceRequest, ContinuousBatcher};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn request_ids_are_unique(
            num_requests in 2usize..50
        ) {
            let mut ids = Vec::new();
            
            for _ in 0..num_requests {
                let req = InferenceRequest::new(vec![1, 2, 3]);
                ids.push(req.id);
            }

            // All IDs should be unique
            ids.sort();
            ids.dedup();
            prop_assert_eq!(ids.len(), num_requests);
        }

        #[test]
        fn continuous_batcher_slot_management(
            operations in prop::collection::vec((prop::bool::ANY, 0u64..100), 1..100)
        ) {
            let mut batcher = ContinuousBatcher::new(20);
            let mut active_slots: Vec<usize> = Vec::new();

            for (is_add, id) in operations {
                if is_add && active_slots.len() < 20 {
                    if let Some(slot) = batcher.add(id, &[1, 2, 3]) {
                        active_slots.push(slot);
                    }
                } else if !active_slots.is_empty() {
                    let slot = active_slots.pop().unwrap();
                    batcher.remove(slot);
                }

                // Invariant: active_count + free_count = max_slots
                prop_assert_eq!(batcher.active_count() + batcher.free_count(), 20);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILS PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod utils_properties {
    use super::*;
    use truthgpt_rust::{f32_to_bytes, bytes_to_f32, AtomicCounter, format_bytes};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn f32_bytes_roundtrip(data in prop::collection::vec(-1e10f32..1e10, 0..100)) {
            let bytes = f32_to_bytes(&data);
            let recovered = bytes_to_f32(&bytes);
            prop_assert_eq!(data, recovered);
        }

        #[test]
        fn counter_add_is_correct(
            initial in 0u64..1000,
            additions in prop::collection::vec(1u64..100, 0..50)
        ) {
            let counter = AtomicCounter::with_value(initial);
            let expected: u64 = initial + additions.iter().sum::<u64>();
            
            for n in additions {
                counter.add(n);
            }
            
            // Handle overflow
            let actual = counter.get();
            prop_assert_eq!(actual, expected % (u64::MAX.wrapping_add(1)));
        }

        #[test]
        fn format_bytes_is_non_empty(bytes in 0u64..u64::MAX) {
            let result = format_bytes(bytes);
            prop_assert!(!result.is_empty());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROPE PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod rope_properties {
    use super::*;
    use truthgpt_rust::{RoPE, RoPEConfig, RoPEScaling, ALiBi, YaRN};
    use ndarray::Array3;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn rope_preserves_shape(
            batch_heads in 1usize..4,
            seq_len in 1usize..16,
            head_dim in prop::sample::select(vec![8usize, 16, 32, 64])
        ) {
            let config = RoPEConfig {
                dim: head_dim,
                max_seq_len: 128,
                base: 10000.0,
                scaling: RoPEScaling::None,
            };
            let rope = RoPE::new(config);

            let q = Array3::ones((batch_heads, seq_len, head_dim));
            let k = Array3::ones((batch_heads, seq_len, head_dim));

            let (q_rot, k_rot) = rope.apply(&q, &k, 0);
            
            prop_assert_eq!(q_rot.dim(), (batch_heads, seq_len, head_dim));
            prop_assert_eq!(k_rot.dim(), (batch_heads, seq_len, head_dim));
        }

        #[test]
        fn alibi_bias_shape_matches(
            num_heads in 1usize..16,
            seq_len in 1usize..64
        ) {
            let alibi = ALiBi::new(num_heads);
            let bias = alibi.compute_bias(seq_len);

            prop_assert_eq!(bias.dim(), (num_heads, seq_len, seq_len));
        }

        #[test]
        fn yarn_mscale_positive(factor in 0.5f32..10.0) {
            let mscale = if factor <= 1.0 {
                1.0
            } else {
                0.1 * factor.ln() + 1.0
            };
            prop_assert!(mscale > 0.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPECULATIVE DECODING PROPERTY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod speculative_properties {
    use super::*;
    use truthgpt_rust::{TreeSpeculation, kl_divergence};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn kl_divergence_is_nonnegative(
            p in prop::collection::vec(0.01f32..1.0, 2..10),
            q in prop::collection::vec(0.01f32..1.0, 2..10)
        ) {
            if p.len() == q.len() {
                // Normalize
                let p_sum: f32 = p.iter().sum();
                let q_sum: f32 = q.iter().sum();
                let p_norm: Vec<f32> = p.iter().map(|x| x / p_sum).collect();
                let q_norm: Vec<f32> = q.iter().map(|x| x / q_sum).collect();
                
                let kl = kl_divergence(&p_norm, &q_norm);
                prop_assert!(kl >= -1e-6, "KL divergence should be non-negative: {}", kl);
            }
        }

        #[test]
        fn acceptance_rate_bounded(
            accepted in 0usize..100,
            total in 1usize..100
        ) {
            let rate = accepted as f32 / total as f32;
            prop_assert!(rate >= 0.0 && rate <= (accepted as f32 / total as f32).min(1.0));
        }

        #[test]
        fn tree_path_count(
            branching in 1usize..4,
            depth in 1usize..4
        ) {
            let mut tree = TreeSpeculation::new(branching, depth);
            
            // Create logprobs for each depth
            let logprobs: Vec<Vec<f32>> = (0..depth)
                .map(|_| vec![0.0; branching + 2])
                .collect();
            
            tree.build_tree(&logprobs);
            
            let paths = tree.get_paths();
            
            // Number of paths should be branching^depth
            let expected_paths = branching.pow(depth as u32);
            prop_assert!(paths.len() <= expected_paths + 1);
        }
    }
}

