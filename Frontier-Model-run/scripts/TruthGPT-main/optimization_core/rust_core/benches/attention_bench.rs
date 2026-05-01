//! Attention Benchmarks
//!
//! Benchmarks for attention mechanisms.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ndarray::Array3;
use truthgpt_rust::attention::{
    scaled_dot_product_attention, flash_attention_block, sparse_attention, create_causal_mask,
};

fn bench_scaled_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaled_dot_product_attention");
    
    let batch_size = 1;
    let d_k = 64;
    
    for seq_len in [64, 128, 256, 512, 1024].iter() {
        let query = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        let key = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        let value = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        
        group.throughput(Throughput::Elements((*seq_len * *seq_len) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("seq_{}", seq_len)),
            seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(scaled_dot_product_attention(&query, &key, &value, None))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");
    
    let batch_size = 1;
    let d_k = 64;
    
    for seq_len in [128, 256, 512, 1024, 2048].iter() {
        let query = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        let key = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        let value = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        
        group.throughput(Throughput::Elements((*seq_len) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("seq_{}", seq_len)),
            seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(flash_attention_block(&query, &key, &value, 64))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_flash_block_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_block_sizes");
    
    let batch_size = 1;
    let seq_len = 512;
    let d_k = 64;
    
    let query = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
    let key = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
    let value = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
    
    for block_size in [32, 64, 128, 256].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("block_{}", block_size)),
            block_size,
            |b, &block_size| {
                b.iter(|| {
                    black_box(flash_attention_block(&query, &key, &value, block_size))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_sparse_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_attention");
    
    let batch_size = 1;
    let d_k = 64;
    let local_window = 64;
    let global_tokens = 4;
    
    for seq_len in [256, 512, 1024, 2048].iter() {
        let query = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        let key = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        let value = Array3::from_elem((batch_size, *seq_len, d_k), 1.0f32);
        
        group.throughput(Throughput::Elements(*seq_len as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("seq_{}", seq_len)),
            seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(sparse_attention(&query, &key, &value, local_window, global_tokens))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_causal_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("causal_mask");
    
    for seq_len in [128, 256, 512, 1024, 2048].iter() {
        group.throughput(Throughput::Elements((*seq_len * *seq_len) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("seq_{}", seq_len)),
            seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    black_box(create_causal_mask(seq_len))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_attention_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_comparison");
    
    let batch_size = 1;
    let seq_len = 512;
    let d_k = 64;
    
    let query = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
    let key = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
    let value = Array3::from_elem((batch_size, seq_len, d_k), 1.0f32);
    
    group.bench_function("standard", |b| {
        b.iter(|| {
            black_box(scaled_dot_product_attention(&query, &key, &value, None))
        })
    });
    
    group.bench_function("flash_64", |b| {
        b.iter(|| {
            black_box(flash_attention_block(&query, &key, &value, 64))
        })
    });
    
    group.bench_function("sparse_64_4", |b| {
        b.iter(|| {
            black_box(sparse_attention(&query, &key, &value, 64, 4))
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_scaled_dot_product,
    bench_flash_attention,
    bench_flash_block_sizes,
    bench_sparse_attention,
    bench_causal_mask,
    bench_attention_comparison,
);

criterion_main!(benches);












