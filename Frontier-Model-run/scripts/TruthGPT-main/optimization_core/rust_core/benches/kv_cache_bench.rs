//! KV Cache Benchmarks
//!
//! Benchmarks for the high-performance KV Cache implementation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use truthgpt_rust::{KVCache, KVCacheConfig, EvictionStrategy};

fn bench_kv_cache_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_get");
    
    let config = KVCacheConfig {
        max_size: 8192,
        eviction_strategy: EvictionStrategy::LRU,
        enable_compression: false,
        compression_threshold: 1024,
    };
    
    let mut cache = KVCache::new(config);
    
    for i in 0..1000 {
        cache.put(0, i, vec![0u8; 128]);
    }
    
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("hit", |b| {
        b.iter(|| {
            black_box(cache.get(0, 500))
        })
    });
    
    group.bench_function("miss", |b| {
        b.iter(|| {
            black_box(cache.get(0, 9999))
        })
    });
    
    group.finish();
}

fn bench_kv_cache_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_put");
    
    for size in [64, 256, 1024, 4096].iter() {
        let config = KVCacheConfig {
            max_size: 8192,
            eviction_strategy: EvictionStrategy::LRU,
            enable_compression: false,
            compression_threshold: 1024,
        };
        
        let mut cache = KVCache::new(config);
        let data = vec![0u8; *size];
        
        group.throughput(Throughput::Bytes(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}B", size)),
            size,
            |b, _| {
                let mut pos = 0usize;
                b.iter(|| {
                    cache.put(0, pos, data.clone());
                    pos += 1;
                })
            },
        );
    }
    
    group.finish();
}

fn bench_eviction_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction_strategy");
    
    let strategies = [
        ("LRU", EvictionStrategy::LRU),
        ("LFU", EvictionStrategy::LFU),
        ("FIFO", EvictionStrategy::FIFO),
        ("Adaptive", EvictionStrategy::Adaptive),
    ];
    
    for (name, strategy) in strategies.iter() {
        let config = KVCacheConfig {
            max_size: 1000,
            eviction_strategy: *strategy,
            enable_compression: false,
            compression_threshold: 1024,
        };
        
        let mut cache = KVCache::new(config);
        let data = vec![0u8; 128];
        
        group.bench_function(*name, |b| {
            let mut pos = 0usize;
            b.iter(|| {
                cache.put(0, pos, data.clone());
                pos += 1;
            })
        });
    }
    
    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_compression");
    
    for compression in [false, true].iter() {
        let config = KVCacheConfig {
            max_size: 8192,
            eviction_strategy: EvictionStrategy::LRU,
            enable_compression: *compression,
            compression_threshold: 512,
        };
        
        let mut cache = KVCache::new(config);
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        
        let name = if *compression { "compressed" } else { "uncompressed" };
        
        group.bench_function(name, |b| {
            let mut pos = 0usize;
            b.iter(|| {
                cache.put(0, pos, data.clone());
                pos += 1;
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_kv_cache_get,
    bench_kv_cache_put,
    bench_eviction_strategies,
    bench_compression,
);

criterion_main!(benches);












